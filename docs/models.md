# Models

All temporal models receive 1536-dimensional frozen DINOv2 features. A LayerNorm
without affine parameters and a linear 1536-to-64 projection precede the temporal
block. A linear readout maps the last valid state to the target angle or angles.
Hidden state resets for each window. The model also accepts another feature
dimension, but reproducing the reported setup requires 1536.

| Model | Temporal block | Trainable temporal parameters |
|---|---|---:|
| LSTM | One 64-unit LSTM layer, input is 64 features concatenated with elapsed time | 33,536 |
| CfC | Handwritten default-mode cell, input 64, hidden 64, no backbone | 33,024 |
| LTC | Fully connected ncps cell, input 64, hidden 64 | 33,216 |
| Transformer | One pre-norm encoder layer, width 64, four heads, feed-forward width 128, GELU, no dropout | 33,472 |
| Single-frame | Linear 64-to-256, GELU, linear 256-to-64 on the current frame | 33,088 |

The shared projection has 98,368 trainable parameters. The readout has 65 for
current-angle prediction or 260 for the four anticipation horizons. LTC additionally
has 8,192 frozen mask entries; these are excluded from trainable counts.

## Elapsed time and missing frames

`models.SteeringModel` receives features `(B,T,D)`, elapsed times `(B,T)`, a valid
mask `(B,T)` and relative timestamps `(B,T)`. Valid observations must be packed
at the beginning in chronological order. Padding follows the target frame.

`models.forward_masked` accepts unpadded windows and absolute timestamps, randomly
retains history frames and always keeps the target frame. It computes elapsed
time from the retained original timestamps, with the first elapsed time zero.
No compacted-frame timing replaces actual gaps.

The high-level training and inference APIs use seconds (`dt_unit=1.0`). LTC clamps
elapsed time to at least 0.001 s inside its solver. Its reversal-potential seed
matches the training seed, and its parameters are constrained after every optimiser
step. `TrainConfig` uses 24 ODE integration substeps, matching the reported runs.
The low-level cell constructor retains its default of six; pass `ode_unfolds=24`
explicitly when constructing a reported-model configuration yourself.

## Transformer

For retained frame time `t_i` and target time `t_target`, the default encoding uses
`p_i = (t_i - t_target) / 0.1`. Dimensions `2j` and `2j+1` are
`sin(p_i / 10000**(2j/64))` and `cos(p_i / 10000**(2j/64))`.
This code is added to projected features without trainable parameters. Attention
is causal and excludes padding; there is no final encoder LayerNorm. Inference
uses the same attention path as training.

The low-level `time_code="index"` and `time_code="none"` options support timing
ablations. The training interface uses `"real"`. These options do not change
parameter counts.

## CfC

The candidate states and interpolation gate are:

```text
candidate_h = tanh(W_h [x,h] + b_h)
candidate_g = tanh(W_g [x,h] + b_g)
gate = sigmoid((W_f [x,h] + b_f) * dt + W_o [x,h] + b_o)
h_new = (1 - gate) * candidate_h + gate * candidate_g
```

Candidate weights use Xavier initialisation and zero biases. The optional backbone
and mask interfaces remain available on `CfCCell`. The tests compare forward
outputs and gradients against the ncps default cell after explicit weight transfer,
and independently check the equations and initialisation.
