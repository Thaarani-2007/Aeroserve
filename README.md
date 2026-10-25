# Registering a Model with AeroServe

## 1. Overview

AeroServe is an ML inference serving platform that helps developers register their machine learning models, profile their performance, and serve predictions through an API.

As an ML developer, you provide your model's inference implementation, sample inputs, and required model assets. AeroServe profiles the model, generates a configuration, and registers the model for serving.

## 2. Prerequisites

Before registering a model, ensure you have:

- Python installed with the dependencies required by your model.
- Docker and Docker Compose installed.
- The AeroServe CLI and platform services available.
- A model implementation compatible with the AeroServe processor interface.
- Representative sample inputs for testing and profiling.

**Important:** AeroServe must support the framework and dependencies required by your model. Specifying a framework does not automatically install its dependencies.

## 3. Prepare Your Model Package

Create a directory containing your inference processor, test inputs, and model assets.

Example:

```text
my_model/
├── processor.py
├── test_inputs.json
├── weights/
│   └── model_weights.bin
└── tokenizer/
    └── tokenizer_config.json
```

Your package may contain different files depending on your model. The weights and tokenizer shown above are examples.

### A. Implement `processor.py`

Your processor connects your model to AeroServe's inference interface.

For a custom processor, inherit from `AeroServeCustomProcessor` and implement `execute()`.

```python
from models.aeroserve_preprocessor import AeroServeCustomProcessor

class SentimentProcessor(AeroServeCustomProcessor):
    def __init__(self):
        # Load your model and required assets here.
        pass

    def execute(self, payload: dict) -> dict:
        # Preprocess input, run inference,
        # and format the prediction.
        return {
            "prediction": "positive"
        }
```

This is an illustrative example. Replace the placeholder with your actual model-loading and inference logic.

Your implementation must return a dictionary that can be serialized to JSON. The processor must follow the interface and batching requirements supported by your AeroServe version.

### B. Create `test_inputs.json`

Provide a non-empty JSON array containing representative inputs that your model can process.

```json
[
  {
    "text": "The product is excellent."
  },
  {
    "text": "The service was disappointing."
  }
]
```

Each object must match the input format expected by your processor. AeroServe uses these examples to validate the model interface and profile inference performance.

Do not include confidential or sensitive production data in test inputs.

### C. Include model assets

Include the files needed to load and run your model, such as model weights, tokenizer files, configuration files, or other assets.

Ensure your processor loads assets using paths that work in the deployed model directory.

## 4. Register Your Model Using the CLI

Run the following command from the AeroServe project root:

```bash
python cli.py submit \
  --model-dir ./my_model \
  --name sentiment_analyzer \
  --framework transformers \
  --processor-class SentimentProcessor
```

Replace the example values with your own.

| Argument | Description |
|---|---|
| `--model-dir` | Directory containing `processor.py`, `test_inputs.json`, and required model assets. |
| `--name` | Unique registry name used to identify the model. |
| `--framework` | Supported inference runtime or framework selected for your model. |
| `--processor-class` | Exact processor class name defined in `processor.py`. Optional if AeroServe can uniquely identify the processor. |
| `--sla` | Optional target latency in milliseconds used during profiling. |

Use a framework choice supported by your AeroServe runtime. The CLI's current choices are `transformers`, `pytorch`, `tensorflow`, `onnx`, and `custom`.

## 5. What AeroServe Does During Registration

When you submit your model, AeroServe is intended to perform the following steps:

1. **Validate the package:** Check that the processor and test inputs exist and that the test inputs contain valid JSON.
2. **Load the processor:** Import the selected processor class and instantiate it.
3. **Validate inference:** Execute test inputs and check that outputs follow the required format.
4. **Profile performance:** Measure inference latency, throughput, and memory usage across supported batch sizes.
5. **Generate configuration:** Create `model_config.json` containing the selected batch size, batching timeout, latency estimate, priority timeouts, and worker limits.
6. **Register the model:** Copy the model package into AeroServe's model registry.
7. **Verify availability:** Check that the gateway responds. A successful gateway health check alone does not guarantee that the submitted model is loaded or that inference succeeds.

**Registration status:** The current CLI implementation should be treated as a prototype. Before claiming full deployment verification, AeroServe should confirm that the specific model is registered, loaded by a worker, and able to return a successful test prediction.

## 6. Send a Prediction Request

Once the model is registered and available, a client application can send a prediction request to the AeroServe gateway.

Example request:

```http
POST http://localhost:8000/predict
Content-Type: application/json
```

Illustrative request body:

```json
{
  "model_name": "sentiment_analyzer",
  "input": {
    "text": "The product is excellent."
  }
}
```

The exact request fields and endpoint must match the prediction schema implemented by your AeroServe gateway.

Example response:

```json
{
  "prediction": "positive"
}
```

The response is illustrative; actual output depends on the registered model and gateway response schema.

## 7. Troubleshooting

- **Missing processor:** Confirm that `processor.py` exists in the model directory.
- **Invalid test inputs:** Ensure `test_inputs.json` contains a non-empty JSON array of inputs accepted by your processor.
- **Processor loading error:** Verify the class name, imports, dependencies, and inheritance from the required AeroServe base class.
- **Profiling failure:** Check model initialization, available memory, device compatibility, and whether the requested latency target can be met.
- **Model unavailable after registration:** Confirm that the gateway and workers can access the registered model directory and that the runtime refreshes or reloads its model registry.
- **Prediction failure:** Verify the gateway's request schema, model name, processor output format, and worker logs.

## 8. Important Notes

- Register only models and code that you are authorized to deploy.
- Model code executes during loading and inference. Only register trusted code; profiling should be isolated from the platform in a production deployment.
- Model profiling provides estimates based on the tested hardware and inputs. It does not guarantee production latency or throughput.
- Keep your model package and dependencies compatible with the AeroServe runtime.
