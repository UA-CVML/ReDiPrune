import re
import argparse

def extract_text_len(line):
    MEM_RE = re.compile(r'\[TIMING\]\s*text length is:\s*(\d+)')
    match = MEM_RE.search(line)
    if match:
        return int(match.group(1))
    return None

def extract_after_generation_memory(line):
    MEM_RE = re.compile(r'\[TIMING\]\s*after generation memory:\s*(\d+)')
    match = MEM_RE.search(line)
    if match:
        return int(match.group(1))
    return None

def extract_generation_latency_time(line):
    LAT_RE = re.compile(r'\[TIMING\]\s*Generation latency time is:\s*([\d.]+)')
    match = LAT_RE.search(line)
    if match:
        return float(match.group(1))
    return None

def extract_generation_prefill_time(line):
    LAT_PRE = re.compile(r'\[TIMING\]\s*prefill time is:\s*([\d.]+)')
    match = LAT_PRE.search(line)
    if match:
        return float(match.group(1))
    return None

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    feature_size_array = []
    after_generation_memory_array = []
    generation_latency_array = []
    generation_prefill_array = []
    generation_text_len_array = []
    i = 0
    while i < len(lines):

        after_generation_memory = extract_after_generation_memory(lines[i])
        if after_generation_memory:
            after_generation_memory_array.append(after_generation_memory)

        generation_latency_time = extract_generation_latency_time(lines[i])
        if generation_latency_time:
            generation_latency_array.append(generation_latency_time)

        generation_prefill_time = extract_generation_prefill_time(lines[i])
        if generation_prefill_time:
            generation_prefill_array.append(generation_prefill_time)

        generation_text_len = extract_text_len(lines[i])
        if generation_text_len:
            generation_text_len_array.append(generation_text_len)

        i += 1
    
    return generation_prefill_array, after_generation_memory_array,generation_latency_array,generation_text_len_array

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="input path arg")
    parser.add_argument('--path', type=str, help='path to the log', default = './log_eval.log')
    args = parser.parse_args()

    prefill_array, memoery_array, latency_array, text_len_array = process_file(args.path)
    print(f"Average prefill (s): {float(sum(prefill_array))/len(prefill_array)/1000} Sec")
    print(f"Average E2E latency: {float(sum(latency_array))/len(latency_array)/1000} Sec" )
    print(f"Average max memory: {float(sum(memoery_array))/len(memoery_array)/(1024**3)} GB")
    print(f"Average text length: {float(sum(text_len_array))/len(text_len_array)} tokens over {len(text_len_array)} samples")