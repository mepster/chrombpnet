import math

import numpy as np
import pandas as pd
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import tensorflow as tf
from chrombpnet.helpers.misc import get_strategy
#import chrombpnet.prediction.bigwig_helper as bigwig_helper
import chrombpnet.training.utils.losses as losses
import chrombpnet.training.utils.one_hot as one_hot
import h5py

schemas = {'ALLVAR': ["chr", "start", "end", "species", "peak_name", "6", "7", "8", "9", "10", "sequence"],
           'SINGLEVAR': ["chr_hum", "start_hum", "end_hum", "SNP_hum_chimp", "species", "chr_chimp", "start_chimp", "end_chimp", "peak_name", "10", "11", "12", "13", "14", "sequence"]}

class Predictions_h5():
    def __init__(self, output_prefix):
        print("init")
        self.fname = "{}_predictions.h5".format(output_prefix)
        self.h5_file = h5py.File(self.fname, mode="w")
        self.first_batch = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        print("close")
        self.h5_file.close()

    @staticmethod
    def write_dset(ds, data):
        ds.resize((ds.shape[0] + data.shape[0]), axis=0)
        ds[-data.shape[0]:] = data

    def write_batch(self, profiles, logcounts, names):
        print("write_batch")
        if self.first_batch:
            # this is the first batch; create file, groups, datasets, and write the data
            dt = h5py.special_dtype(vlen=str)
            self.h5_file.create_dataset("names", dtype=dt, data=names, compression="gzip", chunks=True, maxshape=(None,))

            pred_group = self.h5_file.create_group("predictions")
            pred_group.create_dataset("profiles", dtype=float, data=profiles, compression="gzip", chunks=True, maxshape=(None, None))
            pred_group.create_dataset("logcounts", dtype=float, data=logcounts, compression="gzip", chunks=True, maxshape=(None, ))

            self.first_batch = False
        else:
            # this is a new batch; append the data
            Predictions_h5.write_dset(self.h5_file['names'], np.array(names))
            Predictions_h5.write_dset(self.h5_file['predictions/profiles'], profiles)
            Predictions_h5.write_dset(self.h5_file['predictions/logcounts'], logcounts)


def softmax(x, temp=1):
    norm_x = x - np.mean(x,axis=1, keepdims=True)
    return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)

def load_model_wrapper(model_hdf5):
    # read .h5 model
    custom_objects={"multinomial_nll":losses.multinomial_nll, "tf": tf}    
    get_custom_objects().update(custom_objects)    
    model=load_model(model_hdf5, compile=False)
    print(f"got the model from {model_hdf5}")
    model.summary()
    return model

def main(args):
    # get tf strategy to either run on single, or multiple GPUs
    strategy = get_strategy(args)
    with strategy.scope():
        # load model
        model_chrombpnet_nb = load_model_wrapper(model_hdf5=args.chrombpnet_model_nb)
        inputlen = int(model_chrombpnet_nb.input_shape[1])
        outputlen = int(model_chrombpnet_nb.output_shape[0][1])

    # prepare data frames
    schema = schemas[args.schema] # ALLVAR or SINGLEVAR
    regions_df = pd.read_csv(args.input_bed_file, sep='\t', names=schema)
    print("regions:")
    print(regions_df)

    target_chunk_size = 100000
    print("target_chunk_size:", target_chunk_size)
    num_chunks = math.ceil(regions_df.shape[0] / target_chunk_size)
    print("num_chunks:", num_chunks)

    # predict, in chunks
    with Predictions_h5(args.output_prefix + "_chrombpnet_nobias") as file:
        regions_chunks = np.array_split(regions_df, num_chunks)
        for regions_chunk in regions_chunks:
            chunk_size = regions_chunk.shape[0]
            print("chunk size:", chunk_size)

            # get names for this chunk
            if args.schema == 'ALLVAR':
                fields_chunk = regions_chunk.iloc[:, 0:5]
            elif args.schema == 'SINGLEVAR':
                fields_chunk = regions_chunk.iloc[:, 0:9]
            else:
                assert(False)
            print("fields:")
            print(fields_chunk)
            names = ["/".join(map(str,x)) for x in fields_chunk.values.tolist()]
            assert(len(names) == chunk_size)

            # get seqs for this chunk
            seqs_chunk = regions_chunk['sequence']
            print("seqs:")
            print(seqs_chunk)
            #print(seqs_chunk.shape)
            one_hot_seqs_chunk = one_hot.dna_to_one_hot(np.array(seqs_chunk))
            #print(one_hot_seqs_chunk)
            #print(one_hot_seqs_chunk.shape)
            assert(one_hot_seqs_chunk.shape[0] == chunk_size)
            assert(one_hot_seqs_chunk.shape[1] == inputlen)

            # predict this chunk
            pred_logits_wo_bias, pred_logcts_wo_bias = model_chrombpnet_nb.predict([one_hot_seqs_chunk],
                                              batch_size = args.batch_size, # GPU batch size
                                              verbose=True)
            #print("pred_logits_wo_bias.shape:", pred_logits_wo_bias.shape)
            #print("pred_logcts_wo_bias.shape:", pred_logcts_wo_bias.shape)
            assert(pred_logits_wo_bias.shape[0] == chunk_size)
            assert(pred_logcts_wo_bias.shape[0] == chunk_size)

            pred_logits_wo_bias = np.squeeze(pred_logits_wo_bias)

            profile_probs = softmax(pred_logits_wo_bias)
            counts_sum = np.squeeze(pred_logcts_wo_bias)
            #print("profile_probs.shape:", profile_probs.shape)
            #print("counts_sum.shape:", counts_sum.shape)
            assert(profile_probs.shape[0] == chunk_size)
            assert(counts_sum.shape[0] == chunk_size)

            # I think absolute profiles should be computed like this:
            # abs_profiles = profile_probs * np.expand_dims(np.exp(counts_sum),axis=1)

            # write named predictions
            file.write_batch(profile_probs, counts_sum, names)

    # workaround to explicitly close strategy. https://github.com/tensorflow/tensorflow/issues/50487
    import atexit
    atexit.register(strategy._extended._collective_ops._pool.close)  # type: ignore


if __name__=="__main__":
    print("running from command line not implemented")
    # args = parse_args
    # main(args)
