Mistral_AI.ipynb

!unzip /content/test.zip

pip install jiwer

import os

os.environ["MISTRAL_API_KEY"] = "2MLijoYL7VQceYLedrVoKjGGZGDB4F53"

print("Key set:", os.environ.get("MISTRAL_API_KEY") is not None)

import os
import requests
from jiwer import wer
import editdistance

# =========================
# CONFIG
# =========================
BASE_URL = "https://api.mistral.ai/v1"
TEST_DIR = "/content/test"   # change if needed

API_KEY = os.environ.get("MISTRAL_API_KEY")
if not API_KEY:
    raise RuntimeError("MISTRAL_API_KEY environment variable not set!")

print("API key loaded:", bool(API_KEY))
print("Files in test dir:", os.listdir(TEST_DIR))

# METRICS

def cer(gt, pred):
    return editdistance.eval(gt, pred) / max(1, len(gt))


# STEP 1: Upload file

def upload_file_for_ocr(image_path):
    url = f"{BASE_URL}/files"

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"purpose": "ocr"}

        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        print("[UPLOAD ERROR]", response.status_code)
        print(response.text)
        response.raise_for_status()

    result = response.json()
    file_id = result["id"]

    return file_id

# STEP 2: Run OCR

def mistral_ocr_with_file_id(file_id):
    url = f"{BASE_URL}/ocr"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "file",
            "file_id": file_id
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print("[OCR ERROR]", response.status_code)
        print(response.text)
        response.raise_for_status()

    result = response.json()

    # Robust text extraction
    texts = []

    if "pages" in result:
        for p in result["pages"]:
            if "text" in p:
                texts.append(p["text"])
    elif "markdown" in result:
        texts.append(result["markdown"])
    elif "text" in result:
        texts.append(result["text"])
    else:
        print("[WARN] Unknown OCR response format:", result)

    full_text = "\n".join(texts)
    return full_text.strip()


# Combined OCR function
def mistral_ocr_image(image_path):
    file_id = upload_file_for_ocr(image_path)
    pred_text = mistral_ocr_with_file_id(file_id)
    return pred_text


# MAIN EVALUATION LOOP

total_cer = []
total_wer = []
num_samples = 0

for fname in sorted(os.listdir(TEST_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(TEST_DIR, fname)
    txt_path = os.path.splitext(img_path)[0] + ".txt"

    if not os.path.exists(txt_path):
        print(f"[WARN] Missing GT for {fname}")
        continue

    #  Read GT
    with open(txt_path, "r", encoding="utf-8") as f:
        gt_text = f.read().strip()

    try:
        # Run Mistral OCR 
        pred_text = mistral_ocr_image(img_path)
    except Exception as e:
        print(f"[ERROR] OCR failed for {fname}: {e}")
        continue

    # Metrics 
    sample_cer = cer(gt_text, pred_text)
    sample_wer = wer(gt_text, pred_text)

    total_cer.append(sample_cer)
    total_wer.append(sample_wer)
    num_samples += 1

    print("=" * 70)
    print(f"File: {fname}")
    print(f"CER = {sample_cer:.4f} | WER = {sample_wer:.4f}")
    print("GT  :", gt_text[:300])
    print("PRED:", pred_text[:300])


# FINAL RESULTS

print("\n===================================")
if num_samples > 0:
    avg_cer = sum(total_cer) / num_samples
    avg_wer = sum(total_wer) / num_samples

    print(f"Mistral OCR Average CER: {avg_cer:.4f}")
    print(f"Mistral OCR Average WER: {avg_wer:.4f}")
    print("Evaluated samples:", num_samples)
else:
    print("No valid image-GT pairs found!")

