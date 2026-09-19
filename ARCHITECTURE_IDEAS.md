# ChromBPNet architecture experiment ideas (notes for later)

Context: keep all data-wrangling, preprocessing, inference, and evaluation code
as-is. Only touch the model-definition functions. Retrain from scratch to test.

## Where the architecture actually lives

- TF (signal) submodel: `chrombpnet/training/models/chrombpnet_with_bias_model.py`
  -> local `bpnet_model(filters, n_dil_layers, sequence_len, out_pred_len)`
  function (layer names prefixed `wo_bias_`). This is wrapped as a named
  sub-model (`model_wo_bias`) inside the full compiled model.
- Bias-only submodel: `chrombpnet/training/models/bpnet_model.py`
  -> `getModelGivenModelOptionsAndWeightInits(args, model_params)`. Used only
  when running `chrombpnet bias pipeline` to train `bias.h5` from scratch.
- These are two independent copies of the same code, not a shared function.
  Editing one does not affect the other.

Current shared architecture (both files, same shape):
`Conv1D(filters, kernel=21, relu, valid)` -> N x `{Conv1D(filters, kernel=3,
dilation=2^i, relu, valid) + symmetric-crop residual add}` -> profile head
(`Conv1D(1, kernel=75, valid)` -> crop -> flatten) and counts head
(`GlobalAvgPool1D` -> `Dense(1)`). No normalization layers anywhere
(deliberate — keeps DeepSHAP/TF-MoDISco attributions clean).

Typical hyperparams: TF model `filters=512, n_dil_layers=8`; bias model much
smaller (`filters~128, n_dil_layers~4`) and trained only on non-peak/
background regions.

## How the TF model and bias model combine (chrombpnet_with_bias_model.py)

```python
bias_output = bias_model(inp)              # frozen (trainable=False on load)
output_wo_bias = bpnet_model_wo_bias(inp)   # the model actually being trained

profile_out = Add(...)([output_wo_bias[0], bias_output[0]])       # logits added
count_out = logsumexp([output_wo_bias[1], bias_output[1]])        # counts added in linear space
```

- Profile logits are added before the softmax/multinomial-NLL (log-linear
  mixture). Counts are combined via `logsumexp`, equivalent to
  `total_counts ≈ bias_counts + tf_counts` in linear space.
- Bias model is frozen — no gradients flow into it during ChromBPNet training.
  It's a fixed additive term; the TF submodel learns to explain whatever is
  left over after the bias model's contribution.
- Because combination happens only via output addition (not structural
  composition), the two submodels' internal architectures are fully
  decoupled — only need matching output shapes (`(None, out_pred_len)`
  profile, `(None, 1)` counts) and self-consistent valid-conv/crop arithmetic.

## Candidate no-brainer changes, ranked (for the TF submodel)

1. **ReLU -> GELU/Swish.** One-line activation swap in the dilated stack.
   Free, essentially no risk, no shape/receptive-field changes.
2. **WaveNet-style gated activation unit** in place of the single ReLU conv
   per dilated block. Same receptive field / crop arithmetic, richer
   within-block expressiveness. Sketch:

   ```python
   for i in range(1, n_dil_layers + 1):
       conv_layer_name = 'bpnet_{}conv'.format(layer_names[i-1])
       conv_out = Conv1D(2 * filters, kernel_size=3, padding='valid',
                          dilation_rate=2**i, name=conv_layer_name)(x)
       conv_filter, conv_gate = tf.split(conv_out, 2, axis=-1)
       conv_x = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

       x_len = int_shape(x)[1]
       conv_x_len = int_shape(conv_x)[1]
       assert (x_len - conv_x_len) % 2 == 0
       x = Cropping1D((x_len - conv_x_len) // 2,
                       name="bpnet_{}crop".format(layer_names[i-1]))(x)
       x = add([conv_x, x])
   ```
3. **Squeeze-and-excitation block** per residual block (GAP over current
   sequence length -> small FC -> sigmoid channel gate). Cheap, usually a
   small free win in genomics CNNs — but see caveat below re: bias model.
4. Multi-branch/parallel dilations at a few depths (Inception-style) instead
   of strictly sequential doubling, for a more multi-scale receptive field.
5. Replace profile head's large-kernel conv with something that avoids
   checkerboard artifacts if a transposed conv is ever introduced (currently
   it's a plain valid conv, so this is only relevant if the head changes).
6. Stochastic depth / block-level dropout on the residual tower — worth
   trying since ChromBPNet is usually data-limited (peak counts, not
   compute), more likely to help than plain dropout.

Skip (or treat as a bigger, riskier project): BatchNorm/LayerNorm directly
on the trunk (degrades attribution quality — this is why it's absent from
the original architecture); full transformer/self-attention block (real
option long-term, but not "no-brainer," bigger training-dynamics changes).

## Applying changes to the bias model specifically — revised take

Original instinct was "don't give the bias model more capacity, it might
absorb real TF signal." On reflection this was overstated for most of the
above changes. More precise version:

- **Primary safeguard against bias-model contamination is the training
  data** (non-peak/background regions with little real accessibility
  signal), not the architecture per se.
- **Receptive field (controlled by `n_dil_layers`, i.e. total dilation
  depth) is the architectural safeguard that actually matters.** Tn5/DNase
  sequence bias is short-range (~10bp-scale mono/dinucleotide preference),
  so keeping the bias model's dilation depth small is a deliberate,
  meaningful constraint — don't casually increase `n_dil_layers` or add
  mechanisms that extend effective range.
- **Local nonlinearity changes that don't touch the dilation schedule**
  (GELU swap, gated activation units) don't expand receptive field at all —
  low risk on the bias model too, no strong reason to withhold them there.
- **Global-context mechanisms are the real risk**: squeeze-excite's "squeeze"
  step does a GAP over the *entire current sequence length*, injecting
  global context at every position regardless of the local dilation-limited
  receptive field — this can leak longer-range structure in through the
  back door even with nominal receptive field unchanged. Avoid SE (and
  definitely avoid attention, which gives unlimited-range interactions
  irrespective of any dilation cap) on the bias model specifically.
- None of this is from a specific published ablation I've verified — it's
  reasoned from the architecture's evident design intent. If experimenting
  with bias model architecture, validate empirically (e.g. inspect bias-only
  model's learned motifs/PWMs via the existing `pwm_from_input.png` /
  bias_metrics.json outputs before vs. after) rather than trusting this
  purely on priors.

## Suggested order of operations when picking this back up

1. Implement GELU + gated activation unit in
   `chrombpnet_with_bias_model.py`'s local `bpnet_model()` (TF submodel
   only). Leave `bpnet_model.py` (bias-only training) untouched initially —
   existing `bias.h5` files remain usable as-is since combination is just
   output addition.
2. Retrain TF model with existing frozen bias model, compare
   `chrombpnet_metrics.json` / profile JSD / counts pearsonr against
   baseline.
3. If that's a clear win, consider SE blocks as a follow-up (TF submodel
   only), same comparison.
4. Only then, if desired, experiment with the bias model itself (GELU/gating
   only, not SE/attention, not deeper dilation), retraining bias.h5 via
   `chrombpnet bias pipeline`, and re-run the full pipeline with the new
   bias model — inspect bias-model-only diagnostics for signs of TF-signal
   leakage before trusting it.
