import streamlit as st
from transformers import BertTokenizer, BertModel
import time
from model import BERTScoring

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
bert_model = BertModel.from_pretrained('indobenchmark/indobert-base-p2')
modelForScoring = BERTScoring(bert_model, tokenizer)

# Streamlit app starts here
st.title("Automated Essay Scoring")

# Text area for input: Reference answer
reference_answer = st.text_area("Kunci Jawaban:", height=200)

# Text area for input: Student answer
student_answer = st.text_area("Jawaban Siswa:", height=200)

# Predict button
if st.button('Submit'):
    # Validate input
    if reference_answer.strip() == '' or student_answer.strip() == '':
        st.error('Tolong isi kunci jawaban dan jawaban siswa!')
    else:
        # Perform prediction
        with st.spinner('Melakukan koreksi...'):
            t1 = time.time()
            predictionResult = modelForScoring.predict(reference_answer, student_answer)
            time_cost = time.time() - t1

        # Determine score class based on prediction
        score = predictionResult[0]
        if score >= 0.75:
            score_class = 'Tinggi'
        elif 0.5 <= score < 0.75:
            score_class = 'Sedang'
        else:
            score_class = 'Rendah'

        score *= 100 # to make score between range 0 - 100

        # Display results
        st.subheader("Hasil")
        st.write(f"Nilai : {score:.2f} ({score_class})")
        st.write(f"Waktu dibutuhkan: {time_cost:.2f} detik")
