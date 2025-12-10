# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers.cache_utils import DynamicCache
# import torch

# # Patch for transformers >=4.41
# if not hasattr(DynamicCache, "get_max_length"):
#     DynamicCache.get_max_length = lambda self: None

# model = AutoModelForCausalLM.from_pretrained(
#     "./phi3-base",
#     device_map="auto",
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
#     attn_implementation="eager"
# )
# tokenizer = AutoTokenizer.from_pretrained("./phi3-base", trust_remote_code=True)




# print(f"Model device: {model.device}")




# # Start conversation history
# messages = []

# print("💬 Chat with Phi-3 Mini! Type 'q' to exit.")

# while True:
#     user_input = input("\nYou: ").strip()
#     if user_input.lower() == "q":
#         break

#     # Add user message
#     messages.append({"role": "user", "content": user_input})

#     # Format full conversation with prompt for assistant reply
#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         return_tensors="pt",
#         add_generation_prompt=True
#     ).to(model.device)

#     # Generate response
#     with torch.no_grad():
#         output = model.generate(
#             input_ids,
#             max_new_tokens=256,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id
#         )

#     # Decode and extract assistant reply
#     full_response = tokenizer.decode(output[0], skip_special_tokens=True)
#     assistant_reply = full_response.split("<|assistant|>")[-1].strip()

#     print(assistant_reply)
    
#     # Add assistant reply to history
#     messages.append({"role": "assistant", "content": assistant_reply})

#     print(f"\nPhi-3: {assistant_reply}")
    
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model name
model_name = "microsoft/Phi-3-mini-4k-instruct"

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

    input_length = inputs.shape[1]  # <-- record input length

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Only decode the newly generated tokens
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response

# Test the model
# print("\n" + "="*50)
# prompt = "What is machine learning?"
# prompt = "What is ur name?"
# print(f"Prompt: {prompt}")
# print("="*50)
# response = generate_text(prompt)
# print(f"Response: {response}")
# print("="*50)

messages = []  # Store conversation history as structured messages

while True:
    print("\n\n" + "="*50 + "\n\n")
    prompt = input("Prompt: ")

    if prompt == "q": break
    if not prompt: continue

    # Add user message to history
    messages.append({"role": "user", "content": prompt})

    # Generate response using full history
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(device)

    input_len = inputs.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract ONLY the new response
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    # Add assistant response to history
    messages.append({"role": "assistant", "content": response})
    
    print("\n\n" + "="*50 + "\n\n")
    print(f"Phi-3: {response}")



    # response = generate_text(conversation)
    # # response = input("Response: ")

    # conversation += f" Response: {response} "
    # # print("\n\n" + conversation + "\n\n")

    # print(f"Response: {response}")


