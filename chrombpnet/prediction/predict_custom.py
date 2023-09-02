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

def write_predictions_h5py(output_prefix, profile, logcts, names):
    # open h5 file for writing predictions
    output_h5_fname = "{}_predictions.h5".format(output_prefix)
    h5_file = h5py.File(output_h5_fname, "w")

    # create groups
    pred_group = h5_file.create_group("predictions")

    # create the "names"  dataset
    names_data =  [str(name) for name in names]
    dt = h5py.special_dtype(vlen=str)

    names_dset = h5_file.create_dataset(
        "names",
        data=np.array(names_data, dtype=dt),
        dtype=dt,
        compression="gzip")

    # create the "predictions" group datasets
    profs_dset = pred_group.create_dataset(
        "profs",
        data=profile,
        dtype=float,
        compression="gzip")
    logcounts_dset = pred_group.create_dataset(
        "logcounts",
        data=logcts,
        dtype=float,
        compression="gzip")

    # close hdf5 file
    h5_file.close()

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

    # load data
    schema = schemas[args.schema] # ALLVAR or SINGLEVAR
    regions_df = pd.read_csv(args.input_bed_file, sep='\t', names=schema) # have to add an argument to choose the schema
    print(regions_df.head())

    seqs = regions_df['sequence']
    print(seqs)
    one_hot_seqs = one_hot.dna_to_one_hot(seqs)
    print(one_hot_seqs)

    with strategy.scope():
        # load model
        model_chrombpnet_nb = load_model_wrapper(model_hdf5=args.chrombpnet_model_nb)
        inputlen = int(model_chrombpnet_nb.input_shape[1])
        outputlen = int(model_chrombpnet_nb.output_shape[0][1])

    assert(len(seqs[0]) == inputlen)

    # predict
    pred_logits_wo_bias, pred_logcts_wo_bias = model_chrombpnet_nb.predict([one_hot_seqs],
                                      batch_size = args.batch_size,
                                      verbose=True)

    pred_logits_wo_bias = np.squeeze(pred_logits_wo_bias)

    # bigwig_helper.write_bigwig(softmax(pred_logits_wo_bias) * (np.expand_dims(np.exp(pred_logcts_wo_bias)[:,0],axis=1)),
    #                             regions,
    #                             gs,
    #                             args.output_prefix + "_chrombpnet_nobias.bw",
    #                             outstats_file=args.output_prefix_stats,
    #                             use_tqdm=args.tqdm)

    profile_probs_predictions = softmax(pred_logits_wo_bias)
    counts_sum_predictions = np.squeeze(pred_logcts_wo_bias)

    # compose the name for each row - it's made of the first 5 fields joined with a "/"
    fields = regions_df.iloc[:, 0:5] #0:5]
    print(fields)
    names = ["/".join(map(str,x)) for x in fields.values.tolist()]

    # write named predictions
    write_predictions_h5py(args.output_prefix + "_chrombpnet_nobias", profile_probs_predictions, counts_sum_predictions, names)

    # workaround to explicitly close strategy. https://github.com/tensorflow/tensorflow/issues/50487
    import atexit
    atexit.register(strategy._extended._collective_ops._pool.close)  # type: ignore


if __name__=="__main__":
    print("running from command line not implemented")
    # args = parse_args
    # main(args)
