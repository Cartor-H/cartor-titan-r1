from pathlib import Path

# Import necessary classes from qwen2 source code
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer

local_model_path = "DeepSeek-R1-Distill-Qwen-1.5B"

# Load the tokenizer using tokenizer_config.json and tokenizer.json
tokenizer_config_file = str(Path(local_model_path) / "tokenizer_config.json")
tokenizer_file = str(Path(local_model_path) / "tokenizer.json")

# tokenizer = Qwen2Tokenizer.from_pretrained(local_model_path, vocab_file=tokenizer_config_file, merges_file=tokenizer_file)
# tokenizer = LlamaTokenizer.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
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