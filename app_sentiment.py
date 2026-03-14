import streamlit as st
from transformers import pipeline

def main():
    sentiment_pipeline = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    st.title("Sentiment Analysis with HuggingFace Spaces")
    st.write("Enter a sentence to analyze its sentiment:")

    user_input = st.text_input("")
    # Testing with the saved model
    model3 = AutoModelForSequenceClassification.from_pretrained("isom5240/2026Spring5240L1",
                                                            num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Tokenized testing data
    text = user_input
    #"dr. goldberg offers everything i look for in a general practitioner. he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-notch hospital (nyu) which my parents have explained to me is very important in case something happens and you need surgery; and you can get referrals to see specialists without having to see him first. really, what more do you need? i'm sitting here trying to think of any complaints i have about him, but i'm really drawing a blank."
    inputs = tokenizer(text,
                   padding=True,
                   truncation=True,
                   return_tensors='pt')

    outputs = model3(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()

    # Get the index of the largest output value
    max_index = np.argmax(predictions)

    st.write(f"Prediction: {max_index}")

if __name__ == "__main__":
    main()
