from vllm import LLM, SamplingParams

llm = LLM(model="TheBloke/mixtral-8x7b-v0.1-AWQ", 
          gpu_memory_utilization=0.95, 
          quantization="AWQ",  
          #kv_cache_dtype="fp8", 
          enforce_eager=True,
          enable_chunked_prefill=True, 
          max_num_batched_tokens=8192
         )

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
