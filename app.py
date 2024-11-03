# Import necessary libraries
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer from Hugging Face
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # Consider switching to a health-focused model if available
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# Set up the Streamlit app UI
st.title("Symptom Checker and Health Guidance")
st.write("Enter your symptoms below to receive AI-generated insights. Please note that this is not a substitute for professional medical advice.")

# User input for symptoms
symptom_input = st.text_area("Describe your symptoms here (e.g., 'I have a sore throat and a mild fever.')")

# Generate response on button click
if st.button("Analyze Symptoms"):
    if symptom_input:
        # Add a disclaimer prompt to clarify this is not medical advice
        disclaimer_prompt = (
            "You are an AI language model trained to assist with health-related questions. Provide insights "
            "based on common health information but avoid diagnosing. Remind the user that consulting a "
            "medical professional is essential."
        )
        
        # Combine disclaimer prompt with user input
        input_text = disclaimer_prompt + "\n\nUser: " + symptom_input + "\nAI:"
        
        # Generate a response
        response = generator(input_text, max_length=150, num_return_sequences=1)[0]['generated_text']
        
        # Post-process the response to exclude the disclaimer and input prompt
        response_text = response.split("AI:")[-1].strip()
        
        # Display the AI response
        st.write("### AI's Response")
        st.write(response_text)
        
        # Disclaimer at the bottom
        st.warning("**Disclaimer:** This AI tool is not a substitute for professional medical advice. "
                   "Please consult a healthcare provider for an accurate diagnosis and treatment.")
    else:
        st.error("Please enter your symptoms for analysis.")

# Deploy on Streamlit or run locally
# Use the command `streamlit run your_script.py` to run this app
