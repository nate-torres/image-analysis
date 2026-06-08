# from fastapi import FastAPI

import os 
import tempfile
import requests
from PIL import Image
from io import BytesIO

from tags.get_tags import get_tags
from genre.predict_genre_model import predict_genre
from composition.predict_composition_model import predict_composition
from color.predict_color_harmony import extract_base_palette,find_color_harmonies,best_harmony
from lighting.type.predict_lighting_type_model import predict_light_type
from lighting.quality.predict_lighting_quality_model import predict_light_quality
from lighting.time.predict_lighting_time_model import predict_light_time
from lighting.pattern.predict_lighting_pattern_model import predict_light_pattern

# app = FastAPI()

def is_url(path):
    return path.startswith("http://images") or path.startswith("https://images")

# @app.get("/analyze/")
def analyze(img: str):
    if is_url(img):
        response = requests.get(img)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(tmp.name)
            img = tmp.name

    tags = get_tags(img)

    print("Tags:", tags)
    
    genre_prediction = predict_genre(img, threshold=0.3)

    composition_prediction = predict_composition(img, threshold=0.4)

    print("\nPredicted genre labels:", genre_prediction)
    print("\nPredicted composition labels:", composition_prediction)

    color_palette = extract_base_palette(img)
    color_match = find_color_harmonies(color_palette)
    color_harmony = best_harmony(color_match)

    print("Detected base colours:", color_palette)
    print("Main harmony:", color_harmony)

    light_type = predict_light_type(img, threshold=0.9)
    light_quality = predict_light_quality(img, threshold=0.5)
    light_time = predict_light_time(img, threshold=0.8)
    light_pattern = predict_light_pattern(img, threshold=0.5)

    print("Light type:", light_type)
    print("Light quality:", light_quality)
    print("Light time:", light_time)
    print("Light pattern:", light_pattern)

    if is_url(img):
        os.remove(img)

    results = {
        "tags": tags,
        "genre": genre_prediction,
        "composition": composition_prediction,
        "color": color_harmony,
        "light-type": light_type,
        "light-quality": light_quality,
        "light-time": light_time,
    }

# 2. Check if "headshot" is in the genre list and conditionally add light_pattern
    if "headshot" in genre_prediction:
        light_pattern = predict_light_pattern(img, threshold=0.5)
        results["light-pattern"] = light_pattern
        print("Light pattern (Headshot detected):", light_pattern)

    return results;


if __name__ == "__main__":
    analyze("https://images.unsplash.com/photo-1773332611628-9e1bdce4881b?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDF8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");