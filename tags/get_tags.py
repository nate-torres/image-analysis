from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from keybert import KeyBERT            # keyword extractor built on SBERT
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()

def get_tags(img_path):
    image    = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_length=60)
    caption  = processor.decode(out_ids[0], skip_special_tokens=True)

    # print(f"\nGenerated caption:\n{caption}\n")

    kw_model = KeyBERT(model="all-MiniLM-L6-v2")   
    keywords = kw_model.extract_keywords(
                caption,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=10
            )
    tags = [phrase.lower().strip() for phrase, _ in keywords]
    # print(f"Raw tags (ranked): {tags}")

    clean_tags = []
    for t in tags:
        if t not in clean_tags:
            clean_tags.append(t)

    # print(f"\nFinal tag list:\n{clean_tags}")
    return clean_tags