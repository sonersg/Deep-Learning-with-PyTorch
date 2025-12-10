import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil
from pathlib import Path
import os

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model name
model_name = "microsoft/Phi-3-mini-4k-instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model (this will use cached files if available)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=False,
)

print("Model loaded successfully!")
print(f"Model size: {model.get_memory_footprint() / 1e9:.2f} GB")


def generate_text(prompt, max_length=200):
    messages = [{"role": "user", "content": prompt}]
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(device)
    except:
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Test the model
print("\n" + "="*50)
# prompt = "What is machine learning?"
prompt = "What is ur name?"
print(f"Prompt: {prompt}")
print("="*50)
response = generate_text(prompt)
print(f"Response: {response}")
print("="*50)


# Now copy from cache to local directory
print("\n📁 Copying model from cache to local directory...")

# Find the cache directory
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_cache = None

# Look for the Phi-3 model in cache
for item in cache_dir.glob("models--microsoft--Phi-3-mini-4k-instruct*"):
    if item.is_dir():
        model_cache = item / "snapshots"
        # Get the latest snapshot (there might be multiple versions)
        snapshots = list(model_cache.glob("*"))
        if snapshots:
            model_cache = snapshots[0]  # Use the first (or only) snapshot
            break

if model_cache and model_cache.exists():
    destination = Path("./phi3-mini-local")
    
    print(f"Source: {model_cache}")
    print(f"Destination: {destination}")
    
    # Remove destination if it exists
    if destination.exists():
        print("Removing existing destination...")
        shutil.rmtree(destination)
    
    # Copy the entire directory
    print("Copying files... (this may take a minute)")
    shutil.copytree(model_cache, destination)
    
    print(f"✅ Model successfully copied to {destination}")
    print(f"📊 Directory size: {sum(f.stat().st_size for f in destination.rglob('*') if f.is_file()) / 1e9:.2f} GB")
else:
    print("❌ Could not find model in cache. Make sure the model loaded successfully first.")

print("\n✅ Done! You can now use the model from ./phi3-mini-local")
print("\n💡 To load from local directory in the future:")
print("model = AutoModelForCausalLM.from_pretrained('./phi3-mini-local', trust_remote_code=False)")