import os
from openai import OpenAI
from fasttok import Tokenizer

# --- CONFIGURATION ---
# Replace with your actual key or set the environment variable
OPENAI_API_KEY = "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" 
MODEL = "gpt-4.1-mini"

def run_real_test():
    client = OpenAI(api_key=OPENAI_API_KEY)
    tok = Tokenizer.from_pretrained(MODEL)
    
    # Example long, verbose prompt
    prompt = (
        "For your information, I am working on the application development "
        "and the progress is currently approximately seventy percent. "
        "In addition to this, we need to check the reference requirements "
        "as soon as possible because the deadline is approaching. "
        "Thanks for your assistance regarding this library."
    )
    
    print("--- STEP 1: Compression ---")
    compressed_prompt = tok.compress(prompt)
    
    orig_tokens = tok.count(prompt)
    comp_tokens = tok.count(compressed_prompt)
    
    print(f"Original Tokens:   {orig_tokens}")
    print(f"Compressed Tokens: {comp_tokens}")
    print(f"Savings:           {orig_tokens - comp_tokens} tokens")
    
    print("\n--- STEP 2: Sending to OpenAI ---")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": compressed_prompt}
            ]
        )
        
        print("\n[OpenAI Response]")
        print(response.choices[0].message.content)
        
        print("\n--- STEP 3: Actual API Usage ---")
        print(f"Prompt tokens used: {response.usage.prompt_tokens}")
        print(f"Total tokens used:  {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        print("\nNote: Please make sure to replace 'your-api-key-here' with a valid OpenAI API key.")

if __name__ == "__main__":
    run_real_test()
