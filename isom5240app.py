
# import part
from transformers import pipeline
from PIL import Image
import streamlit as st

# function part
def age_classifier():
    age_classifier = pipeline("image-classification",
                          model="prithivMLmods/Age-Classification-SigLIP2")

image_name = "middleagedMan.jpg"
image_name = Image.open(image_name).convert("RGB")

# Classify age
age_predictions = age_classifier(image_name)
print(age_predictions)
age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

def main():
    st.header("Title: Age Classification using ViT")
    age_classifier()
    st.write(age_predictions[0]['label'])
  
# main part
if __name__ == "__main__":
    main()
