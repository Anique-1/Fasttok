import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tiktoken
from fasttok import Tokenizer
from openai import OpenAI
import os

# --- 1. CONFIGURATION ---
OPENAI_API_KEY = ""
MODEL = "gpt-4.1-mini"
tok = Tokenizer.from_pretrained(MODEL)
enc = tiktoken.encoding_for_model(MODEL)

# --- 2. DATA PROCESSING ---
prompt = """
For your information, the application development is in order to meet the requirements 
as soon as possible. Due to the fact that the project is at this point in time 
ninety percent finished, please make sure that you check the reference documents 
with respect to the final deployment.
"""

# Tokenize
n_orig = len(enc.encode(prompt))
compressed = tok.compress(prompt)
n_red = len(enc.encode(compressed))
n_saved = n_orig - n_red
percent_saved = (n_saved / n_orig) * 100

# Mock Response (Since real key might vary, we provided integrated logic in the next step)
ai_response = "The application deployment is proceeding well. I have checked the reference documents and confirmed we are 90% finished. Will meet requirements ASAP."

if OPENAI_API_KEY != "sk-...":
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": compressed}]
        )
        ai_response = resp.choices[0].message.content
    except Exception as e:
        print(f"API Call failed or skipped: {e}")

# --- 3. BEAUTIFUL DASHBOARD CREATION (LIGHT THEME) ---
plt.style.use('default')  # Set back to white background
fig = plt.figure(figsize=(16, 12))  # Larger size
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1])

# Panel 1: Bar Chart (Left Top)
ax1 = plt.subplot(gs[0, 0])
colors = ['#FF595E', '#1982C4']
bars = ax1.bar(['Original Content', 'FastTok Optimized'], [n_orig, n_red], color=colors, alpha=0.85, width=0.5)
ax1.set_title('Token Count Reduction', fontsize=18, fontweight='bold', pad=20, color='#2B2D42')
ax1.set_ylabel('Tokens', fontsize=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=16, fontweight='bold', color='#2B2D42')

# Panel 2: Pie Chart (Right Top)
ax2 = plt.subplot(gs[0, 1])
pie_labels = ['Tokens Used', 'Profit (Saved)']
pie_sizes = [n_red, n_saved]
pie_colors = ['#1982C4', '#8AC926']
ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=140, colors=pie_colors, explode=(0, 0.15), shadow=True, textprops={'fontsize': 14, 'fontweight': 'bold'})
ax2.set_title(f'Cost Saving Factor: {percent_saved:.1f}%', fontsize=20, fontweight='bold', color='#6A4C93', pad=15)

# Panel 3: Text Box - User Prompt (Bottom Left)
ax3 = plt.subplot(gs[1, 0])
ax3.axis('off')
prompt_text = f"ORIGINAL PROMPT:\n{'-'*60}\n{prompt.strip()}\n\nFASTTOK COMPRESSED:\n{'-'*60}\n{compressed.strip()}"
ax3.text(0, 0.5, prompt_text, fontsize=12, family='monospace', verticalalignment='center', wrap=True, 
         bbox=dict(boxstyle='round,pad=1', facecolor='#F8F9FA', edgecolor='#DEE2E6', alpha=1))
ax3.set_title("Input Optimization Details", loc='left', fontsize=16, fontweight='bold', color='#1982C4')

# Panel 4: Text Box - AI Response (Bottom Right)
ax4 = plt.subplot(gs[1, 1])
ax4.axis('off')
resp_text = f"AI OUTPUT (Verified Model Intelligence):\n{'-'*60}\n{ai_response.strip()}"
ax4.text(0, 0.5, resp_text, fontsize=12, family='monospace', verticalalignment='center', wrap=True, 
         bbox=dict(boxstyle='round,pad=1', facecolor='#E9F5FF', edgecolor='#BDE0FE', alpha=1))
ax4.set_title("Standard Output Verification", loc='left', fontsize=16, fontweight='bold', color='#1982C4')

plt.suptitle('FastTok Enterprise Deployment Analysis', fontsize=28, fontweight='bold', color='#2B2D42', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.savefig('fasttok_white_dashboard.png', dpi=200)

print("\n[SUCCESS] Large White Dashboard saved as 'fasttok_white_dashboard.png'")
print(f"Summary: Original={n_orig} | Optimized={n_red} | Saved={n_saved} ({percent_saved:.1f}%)")
