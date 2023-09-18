import tensorflow as tf
import tensorflow_probability as tfp


#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values
      logits: predicted logit values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                         logits=logits)

    # This tf.cond() is here because, on multiple GPUs (--multiGPU flag), sometimes an empty batch is passed to the loss function.
    # In our case below, an empty batch produces a NaN, because tf.shape(true_counts)[0] will be 0.
    # This is a longstanding bug apparently, see:
    # https://stackoverflow.com/questions/54283937/training-on-multiple-gpus-causes-nan-validation-errors-in-keras
    # https://github.com/tensorflow/tensorflow/issues/36224
    loss = tf.cond(tf.shape(true_counts)[0] == 0,
                    lambda: 0.,
                    lambda: -tf.reduce_sum(dist.log_prob(true_counts)) /
                             tf.cast(tf.shape(true_counts)[0], dtype=tf.float32)
                    )
    return loss
