import json
from tqdm import tqdm
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

def transform_format(input_data):
    output_data = [
        {
            "conversation": [
                {
                    "system": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": item.get("output", "")
                }
            ]
        }
        for item in input_data
    ]

    return output_data

def extend_output_to_1k(output_text, target_token_count=8000):
    # Calculate the number of tokens in the current output
    token_count = len(tokenizer.encode(output_text))
    
    # Calculate the number of repetitions needed
    repetitions = target_token_count // token_count + 1
    
    # Repeat the text and trim to exactly 64k tokens
    extended_output = (output_text + " ") * repetitions  # Concatenate the text with a space separator
    tokens = tokenizer.encode(extended_output)[1:target_token_count+1]
    extended_output = tokenizer.decode(tokens)
    return extended_output

def process_json_file(input_file, output_file, target_token_count):
    # Load JSON data from file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extend each instruction's output to 64k tokens with progress bar
    for item in tqdm(data, desc="Processing instructions"):
        item['output'] = extend_output_to_1k(item['output'], target_token_count - len(item['instruction']))
    
    data = transform_format(data)
    
    # Save the modified data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Define input and output file paths
input_file = "/root/xtuner/llama3-8b-lorasft-8k/alpaca_en_demo.json"
output_file_8k = "/root/xtuner/llama3-8b-lorasft-8k/alpaca_en_demo_8k.json"  # replace with your output file path

# Run the script
process_json_file(input_file, output_file_8k, target_token_count=8000)
