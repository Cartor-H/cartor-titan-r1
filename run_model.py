# Import necessary classes from qwen2 source code
from qwen2.tokenization_qwen2 import Qwen2Tokenizer
from qwen2.modeling_qwen2 import Qwen2ForCausalLM

local_model_path = "DeepSeek-R1-Distill-Qwen-1.5B"

# Initialize the tokenizer and model from the local path
tokenizer = Qwen2Tokenizer.from_pretrained(local_model_path)
model = Qwen2ForCausalLM.from_pretrained(local_model_path)

# Encode the input text
input_text = "What are you?\n"
inputs = tokenizer(input_text, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Generate text
print("\nGenerating text...\n")
output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)

# Decode the generated text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nOutput Text:\n", output_text)

"""
R1 is based on qwen2 (path = `qwen2`), I downloaded the init, configuration, modeling,
modular, and tokenization python files for qwen2. Can you get the model and tokenizer
directly from the qwen2 source code (using the R1 pretrain)
"""