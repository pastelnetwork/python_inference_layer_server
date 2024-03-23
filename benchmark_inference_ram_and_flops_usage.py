import time
import psutil
import os
from typing import Callable

def benchmark_inference(inference_func: Callable, *args, **kwargs) -> dict:
    """
    Benchmark an inference task and measure FLOPS and RAM usage.

    Args:
        inference_func (Callable): The inference function to benchmark.
        *args: Positional arguments to pass to the inference function.
        **kwargs: Keyword arguments to pass to the inference function.

    Returns:
        dict: A dictionary containing the benchmark results.
    """
    # Get the process ID of the current Python process
    pid = os.getpid()
    process = psutil.Process(pid)
    # Measure the start time and initial memory usage
    start_time = time.time()
    initial_memory = process.memory_info().rss
    # Run the inference function
    result = inference_func(*args, **kwargs)
    # Measure the end time and final memory usage
    end_time = time.time()
    final_memory = process.memory_info().rss
    # Calculate the execution time and memory usage
    execution_time = end_time - start_time
    memory_used = final_memory - initial_memory
    # Calculate the average memory usage
    avg_memory = (initial_memory + final_memory) / 2
    # Estimate the number of FLOPS (floating-point operations)
    # This is a rough estimate based on the execution time and assumes a certain number of FLOPS per second
    # Adjust the FLOPS_PER_SECOND value based on your system's specifications and the specific inference task
    FLOPS_PER_SECOND = 1e9  # Assuming 1 GFLOPS (adjust this value based on your system)
    estimated_flops = execution_time * FLOPS_PER_SECOND
    # Create a dictionary with the benchmark results
    benchmark_results = {
        "execution_time": execution_time,
        "memory_used": memory_used,
        "avg_memory": avg_memory,
        "estimated_flops": estimated_flops,
        "result": result
    }
    return benchmark_results

# Example usage with llama_cpp
def llama_cpp_inference(prompt: str, model_path: str, n_threads: int = 4, n_predict: int = 128, temp: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> str:
    import llama_cpp
    # Load the model
    model = llama_cpp.Llama(model_path, n_threads=n_threads)
    # Set the inference parameters
    model.set_n_predict(n_predict)
    model.set_temperature(temp)
    model.set_top_k(top_k)
    model.set_top_p(top_p)
    # Run the inference
    output = model.generate(prompt, max_tokens=n_predict)
    # Clean up the generated output
    result = output.strip()
    return result

# Example usage with Huggingface Transformers
def huggingface_inference(prompt: str, model_name: str = "facebook/opt-1.3b", max_length: int = 128, num_beams: int = 4, no_repeat_ngram_size: int = 2, early_stopping: bool = True) -> str:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Run the inference
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping
    )
    # Decode the generated output
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result


# Example usage of the benchmark_inference function
prompt = "The meaning of life is"

llama_model_path = "/path/to/llama/model"
llama_cpp_benchmark = benchmark_inference(llama_cpp_inference, prompt, model_path=llama_model_path, n_threads=4, n_predict=128, temp=0.8, top_k=40, top_p=0.9)
print("llama_cpp benchmark results:")
print(llama_cpp_benchmark)

huggingface_model_name = "facebook/opt-1.3b"
huggingface_benchmark = benchmark_inference(huggingface_inference, prompt, model_name=huggingface_model_name, max_length=128, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
print("Huggingface Transformers benchmark results:")
print(huggingface_benchmark)