# Serving codegen-350M-mono-gptj with Triton

## Requirements
As for https://huggingface.co/moyix/codegen-350M-mono-gptj ,
- Set up the client
- Set up a server

## How to Run

### Client
TBD

### Server: Docker (Option 1)
```bash
make model   # Download a model repository for triton.
make triton  # Run a triton server.
```

### Server: Kubernetes (Option 2)
TBD

## Artifacts
- TritonServer with FasterTransformer: https://gitlab.com/curt-park/tritonserver-ft
- CodeGen-350M-mono-gptj: https://huggingface.co/curt-park/codegen-350M-mono-gptj

## For Developer
```bash
make setup      # Install packages for execution.
make setup-dev  # Install packages for development.
make format     # Format the code.
make lint       # Lint the code.
```

## References
- https://github.com/NVIDIA/FasterTransformer
- https://github.com/triton-inference-server/fastertransformer_backend
- https://github.com/triton-inference-server/python_backend
