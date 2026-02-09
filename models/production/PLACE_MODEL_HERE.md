# Place Your Trained Model Here

The inference server loads models from this directory.

## Required Files

Copy these files from your DeepBiocharge training output:

```
models/production/
├── best_model_<epoch>.pt      # Model weights (e.g., best_model_8.pt)
└── model_config.json           # Model architecture config
```

## How to Copy Your Existing Model

If you have a trained model from DeepBiocharge, copy it like this:

```bash
# Example: copy from a DeepBiocharge checkpoint
cp /path/to/DeepBiocharge/checkpoints/<date>/best_model_8.pt models/production/
cp /path/to/DeepBiocharge/checkpoints/<date>/model_config.json models/production/
```

## model_config.json Format

The inference server reads this to reconstruct the model architecture:

```json
{
  "model_type": "mlp",
  "input_dim": 14,
  "hidden_dim": 128,
  "num_layers": 3,
  "dropout": 0.1,
  "norm_type": "layer"
}
```

The key fields used by the inference server are:
- `input_dim` - Number of input features
- `hidden_dim` or `hidden` - Hidden layer size
- `num_layers` or `layers` - Number of hidden layers
- `dropout` - Dropout rate
- `norm_type` or `norm` - Normalization type ("batch" or "layer")

## After Placing the Model

Restart the inference service:

```bash
docker compose restart inference
# or
make serve
```

Test it:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
```
