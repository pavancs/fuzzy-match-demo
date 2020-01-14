# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from screens.screen_1 import load_data, text_preprocess, n_grams
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



def render():
    """

    """
    st.markdown("It is clear from the results that TF-IDF method performs superior to fuzzywuzzy package in terms of"
                "both efficiency/time and accuracy.")

    st.markdown("Let's see how the td-idf score varies,")

    reference, query = load_data()
    query['text'] = query.apply(text_preprocess, axis=1)
    text = np.concatenate((reference['text'], query["text"]), axis=None)
    text = np.unique(text)
    vectorizer = TfidfVectorizer(min_df=1, analyzer=n_grams)
    tf_idf_vector = vectorizer.fit(text)
    reference_text = tf_idf_vector.transform(reference['text'])
    query_text = tf_idf_vector.transform(query['text'])

    # get best match
    reference_id = []
    score = []
    for q in query_text:
        reference_id.append(cosine_similarity(q, reference_text).argmax() + 1)
        score.append(cosine_similarity(q, reference_text).max())

    query['reference_id'] = reference_id
    query['score'] = score

    query.score.hist()
    st.pyplot()

    st.markdown("It's clear that there are some low score value. Fune-tuning cut-off score to allow fuzzy "
                "match could result in better solution. Below we can experiment with it,")

    value = st.slider('Score cut-off', min_value=0.1, max_value=1.0, value=0.6)

    for i, row in query[query['score'] <= value].iterrows():
        st.markdown("*Query row where score is less than %s,*"%value)
        st.dataframe(row.loc[['id', 'name', 'address', 'city', 'cuisine']])

        matched_text = reference[reference['id'] == row['reference_id']]
        matched_text = matched_text.iloc[0]
        st.markdown("*TF-IDF matched reference row*")
        st.dataframe(matched_text.loc[['id', 'name', 'address', 'city', 'cuisine']])
        st.markdown("---")

    return
