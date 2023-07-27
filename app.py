# streamlit run app.py
# https://chatbotslife.com/conversational-chatbot-using-transformers-and-streamlit-73d621afde9

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


#@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast:
# hash}, suppress_st_warning=True)

def load_data():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
    return tokenizer, model



def generate(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer.encode(f"{query}", return_tensors="pt")
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

if __name__=='__main__':
    st.title('Chatbot GODEL : an conversational agent')
    st.write("Welcome to the Chatbot. I am still learning, please be patient")

    tokenizer, model = load_data()

    instruction = st.text_input("Instruction", "Instruction: given a dialog context, you need to respond empathically.")
    knowledge = st.text_input("Knowledge", "")
    dialog = st.text_area("Dialog", value="Does money buy happiness?\nIt is a question. Money buys you a lot of things, but not enough to buy happiness.\nWhat is the best way to buy happiness?")
    dialog = dialog.split("\n")

    if st.button("Generate Response"):
        response = generate(instruction, knowledge, dialog)
        st.text("Generated Response:")
        st.write(response)
