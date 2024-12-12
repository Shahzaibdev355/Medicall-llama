# Save this script as `app.py` in Colab
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Define configurations for 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Define model and tokenizer IDs
base_model_id = "ShahzaibDev/Biomistral_Model_weight_files"
peft_model_id = "ShahzaibDev/biomistral-medqa-finetune"

@st.cache_resource
def load_model_and_tokenizer():
    st.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    st.info("Loading base model with 4-bit precision...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=bnb_config
    )

    st.info("Loading fine-tuned PEFT model...")
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.eval()
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Streamlit UI
st.title("Medical Question Answering")
st.markdown("Powered by BioMistral and PEFT fine-tuning.")

# Input section
question = st.text_area("Enter your question:", "")
question_type = st.text_input("Enter question type (optional):")

# Submit button
if st.button("Get Answer"):
    if question.strip() == "":
        st.error("Please enter a question.")
    else:
        with st.spinner("Generating response..."):
            if question_type:
                eval_prompt = f"""From the MedQuad MedicalQA Dataset: Given the following medical question and question type, provide an accurate answer:

### Question type:
{question_type}

### Question:
{question}

### Answer:
"""
            else:
                eval_prompt = question

            inputs = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300)

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success("Response generated:")
            st.write(answer)
