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

## Experiments: Load Test

Device Info:
- CPU: AMD EPYC Processor (with IBPB)
- GPU: A100-SXM4-80GB x 1
- RAM: 1.857TB

Experimental Setups:
- Single Triton instance.
- Dynamic batching.
- Triton docker server.
- Output Length: 8 vs 32 vs 128 vs 512

### Output Length: 8
![](assets/loadtest_output_len_08_00.png)
![](assets/loadtest_output_len_08_01.png)
```bash
# metrics
nv_inference_count{model="ensemble",version="1"} 391768
nv_inference_count{model="postprocessing",version="1"} 391768
nv_inference_count{model="codegen-350M-mono-gptj",version="1"} 391768
nv_inference_count{model="preprocessing",version="1"} 391768

nv_inference_exec_count{model="ensemble",version="1"} 391768
nv_inference_exec_count{model="postprocessing",version="1"} 391768
nv_inference_exec_count{model="codegen-350M-mono-gptj",version="1"} 20439
nv_inference_exec_count{model="preprocessing",version="1"} 391768

nv_inference_compute_infer_duration_us{model="ensemble",version="1"} 6368616649
nv_inference_compute_infer_duration_us{model="postprocessing",version="1"} 51508744
nv_inference_compute_infer_duration_us{model="codegen-350M-mono-gptj",version="1"} 6148437063
nv_inference_compute_infer_duration_us{model="preprocessing",version="1"} 168281250
```

- RPS (Response per Second) reaches around 1,715.
- The average response time is 38 ms.
- The metric shows dynamic batching works (`nv_inference_count` vs `nv_inference_exec_count`)
- Preprocessing spends 2.73% of the model inference time.
- Postprocessing spends 0.83% of the model inference time.

### Output Length: 32
![](assets/loadtest_output_len_32_00.png)
![](assets/loadtest_output_len_32_01.png)
```bash
# metrics
nv_inference_count{model="ensemble",version="1"} 118812
nv_inference_count{model="codegen-350M-mono-gptj",version="1"} 118812
nv_inference_count{model="postprocessing",version="1"} 118812
nv_inference_count{model="preprocessing",version="1"} 118812

nv_inference_exec_count{model="ensemble",version="1"} 118812
nv_inference_exec_count{model="codegen-350M-mono-gptj",version="1"} 6022
nv_inference_exec_count{model="postprocessing",version="1"} 118812
nv_inference_exec_count{model="preprocessing",version="1"} 118812

nv_inference_compute_infer_duration_us{model="ensemble",version="1"} 7163210716
nv_inference_compute_infer_duration_us{model="codegen-350M-mono-gptj",version="1"} 7090601211
nv_inference_compute_infer_duration_us{model="postprocessing",version="1"} 18416946
nv_inference_compute_infer_duration_us{model="preprocessing",version="1"} 54073590
```
- RPS (Response per Second) reaches around 500.
- The average response time is 122 ms.
- The metric shows dynamic batching works (`nv_inference_count` vs `nv_inference_exec_count`)
- Preprocessing spends 0.76% of the model inference time.
- Postprocessing spends 0.26% of the model inference time.

### Output Length: 128
![](assets/loadtest_output_len_128_00.png)
![](assets/loadtest_output_len_128_01.png)
```bash
nv_inference_count{model="ensemble",version="1"} 14286
nv_inference_count{model="codegen-350M-mono-gptj",version="1"} 14286
nv_inference_count{model="preprocessing",version="1"} 14286
nv_inference_count{model="postprocessing",version="1"} 14286

nv_inference_exec_count{model="ensemble",version="1"} 14286
nv_inference_exec_count{model="codegen-350M-mono-gptj",version="1"} 1121
nv_inference_exec_count{model="preprocessing",version="1"} 14286
nv_inference_exec_count{model="postprocessing",version="1"} 14286

nference_compute_infer_duration_us{model="ensemble",version="1"} 4509635072
nv_inference_compute_infer_duration_us{model="codegen-350M-mono-gptj",version="1"} 4498667310
nv_inference_compute_infer_duration_us{model="preprocessing",version="1"} 7348176
nv_inference_compute_infer_duration_us{model="postprocessing",version="1"} 3605100
```
- RPS (Response per Second) reaches around 65.
- The average response time is 620 ms.
- The metric shows dynamic batching works (`nv_inference_count` vs `nv_inference_exec_count`)
- Preprocessing spends 0.16% of the model inference time.
- Postprocessing spends 0.08% of the model inference time.

### Output Length: 512
![](assets/loadtest_output_len_512_00.png)
![](assets/loadtest_output_len_512_01.png)
```bash
nv_inference_count{model="ensemble",version="1"} 7183
nv_inference_count{model="codegen-350M-mono-gptj",version="1"} 7183
nv_inference_count{model="preprocessing",version="1"} 7183
nv_inference_count{model="postprocessing",version="1"} 7183

nv_inference_exec_count{model="ensemble",version="1"} 7183
nv_inference_exec_count{model="codegen-350M-mono-gptj",version="1"} 465
nv_inference_exec_count{model="preprocessing",version="1"} 7183
nv_inference_exec_count{model="postprocessing",version="1"} 7183

nv_inference_compute_infer_duration_us{model="ensemble",version="1"} 5764391176
nv_inference_compute_infer_duration_us{model="codegen-350M-mono-gptj",version="1"} 5757320649
nv_inference_compute_infer_duration_us{model="preprocessing",version="1"} 3678517
nv_inference_compute_infer_duration_us{model="postprocessing",version="1"} 3384699
```
- RPS (Response per Second) reaches around 40.
- The average response time is 1,600 ms.
- The metric shows dynamic batching works (`nv_inference_count` vs `nv_inference_exec_count`)
- Preprocessing spends 0.06% of the model inference time.
- Postprocessing spends 0.06% of the model inference time.

## References
- https://github.com/NVIDIA/FasterTransformer
- https://github.com/triton-inference-server/fastertransformer_backend
- https://github.com/triton-inference-server/python_backend
