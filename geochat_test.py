import os
import csv
import torch
import numpy as np
from PIL import Image
import pandas as pd

from geochat.conversation import conv_templates, Chat
from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import get_model_name_from_path

# === CONFIGURATION ===
MODEL_PATH = "/content/merged/out_geochat_7b"
IMAGE_DIR = "/content/GeoChat/prompt_prep/images"
OUTPUT_CSV = "/content/results.csv"
DATA_CSV = "/content/test_split.csv"
DEVICE = "cuda"

# === Load metadata CSV ===
df = pd.read_csv(DATA_CSV)

# Questions
QUESTIONS = [
    "Is this region rural or urban?",
    "How many buildings are present in this image?",
    "Estimate the population density class of this region. Use 2^n classes from n=0 to n=16.",
    "Estimate the approximate number of people in this image"
]

# === Load model ===
print("Loading model...")
model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, None, model_name, False, False, device=DEVICE
)
model = model.eval()
chat = Chat(model, image_processor, tokenizer, device=DEVICE)
print("Model loaded!")

# === Function to process image and get answers ===
def process_image_questions(chat, pil_img, questions, origine="unknown"):
    """
    Process an image and get answers to multiple questions
    """
    # Initialize fresh conversation state for this image
    chat_state = conv_templates["llava_v1"].copy()
    img_list = []
    
    try:
        # Upload image - this should handle the preprocessing
        llm_message = chat.upload_img(pil_img, chat_state, img_list)
        print(f"    Image uploaded successfully")
        
        responses = []
        for q_idx, question in enumerate(questions):
            try:
                # For the first question, optionally inject origine information
                if q_idx == 0 and origine != "unknown":
                    full_question = f"The image is from {origine}. {question}"
                else:
                    full_question = question
                
                # Ask the question
                chat.ask(full_question, chat_state)
                
                # Ensure image is encoded if needed
                if len(img_list) > 0:
                    if not isinstance(img_list[0], torch.Tensor):
                        chat.encode_img(img_list)
                
                # Get streaming response
                streamer = chat.stream_answer(
                    conv=chat_state,
                    img_list=img_list,
                    temperature=0.6,
                    max_new_tokens=500,
                    max_length=2000
                )
                
                # Collect full response
                output = ""
                for chunk in streamer:
                    output += chunk
                
                # Update conversation state
                if len(chat_state.messages) > 0 and len(chat_state.messages[-1]) > 1:
                    chat_state.messages[-1][1] = output.strip()
                
                responses.append(output.strip())
                print(f"    Q{q_idx+1}: {full_question}")
                print(f"    A{q_idx+1}: {output.strip()[:150]}...")
                if len(output.strip()) > 150:
                    print("    [truncated]")
                print()
                
            except Exception as e:
                print(f"    Error answering Q{q_idx+1}: {str(e)}")
                responses.append(f"ERROR: {str(e)}")
        
        return responses
        
    except Exception as e:
        print(f"    Error processing image: {str(e)}")
        return [f"ERROR: {str(e)}"] * len(questions)

# === Alternative approach if upload_img fails ===
def process_image_questions_alt(chat, pil_img, questions, origine="unknown"):
    """
    Alternative approach using direct image preprocessing
    """
    try:
        # Convert PIL to numpy array (as expected by some parts of the code)
        img_array = np.array(pil_img)
        
        # Initialize conversation state
        chat_state = conv_templates["llava_v1"].copy()
        img_list = []
        
        # Try to manually add the image
        img_list.append(pil_img)  # Add PIL image directly
        
        # Encode the image
        chat.encode_img(img_list)
        
        responses = []
        for q_idx, question in enumerate(questions):
            try:
                # For the first question, optionally inject origine information
                if q_idx == 0 and origine != "unknown":
                    full_question = f"The image is from {origine}. {question}"
                else:
                    full_question = question
                
                # Ask the question
                chat.ask(full_question, chat_state)
                
                # Get streaming response
                streamer = chat.stream_answer(
                    conv=chat_state,
                    img_list=img_list,
                    temperature=0.6,
                    max_new_tokens=500,
                    max_length=2000
                )
                
                # Collect full response
                output = ""
                for chunk in streamer:
                    output += chunk
                
                # Update conversation state
                if len(chat_state.messages) > 0 and len(chat_state.messages[-1]) > 1:
                    chat_state.messages[-1][1] = output.strip()
                
                responses.append(output.strip())
                print(f"    Q{q_idx+1}: {full_question}")
                print(f"    A{q_idx+1}: {output.strip()[:150]}...")
                print()
                
            except Exception as e:
                print(f"    Error answering Q{q_idx+1}: {str(e)}")
                responses.append(f"ERROR: {str(e)}")
        
        return responses
        
    except Exception as e:
        print(f"    Error in alternative processing: {str(e)}")
        return [f"ERROR: {str(e)}"] * len(questions)

# === Main processing loop ===
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    header = ["filename", "origine"] + [f"question_{i+1}" for i in range(len(QUESTIONS))]
    writer.writerow(header)

    # Process each row
    for idx, row in df.iterrows():
        filename = row["filename"]
        origine = row.get("origine", "unknown")

        image_path = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(image_path):
            print(f"Skipping {filename} (not found)")
            continue

        print(f"[{idx+1}/{len(df)}] Processing: {filename} | Origine: {origine}")

        try:
            # Load image
            pil_img = Image.open(image_path).convert("RGB")
            print(f"  Loaded image: {pil_img.size}")
            
            # Try primary approach first
            responses = process_image_questions(chat, pil_img, QUESTIONS, origine)
            
            # If primary approach failed, try alternative
            if all("ERROR:" in resp for resp in responses):
                print("  Primary approach failed, trying alternative...")
                responses = process_image_questions_alt(chat, pil_img, QUESTIONS, origine)

            # Write to CSV
            writer.writerow([filename, origine] + responses)
            
        except Exception as e:
            print(f"  Complete failure processing {filename}: {str(e)}")
            # Write error row
            error_responses = [f"ERROR: {str(e)}"] * len(QUESTIONS)
            writer.writerow([filename, origine] + error_responses)

        # Optional: Clear GPU memory periodically
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            print(f"  Processed {idx + 1} images, cleared GPU cache")

print(f"Done! Results saved to {OUTPUT_CSV}")

# Final cleanup
torch.cuda.empty_cache()
print("GPU memory cleared")