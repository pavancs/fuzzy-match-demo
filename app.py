# import libraries
import streamlit as st
import time
import re
import numpy as np
import pandas as pd
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# main function
def main():
    """

    :return:
    """

    # Dashboard Title
    st.title("Fuzzy Record Matching Demo!")

    # create select box on sidebar
    activities = ['Introduction', 'Solution', 'Parameter Fine-tuning']
    selected_screen = st.sidebar.selectbox("Select Screen", activities)

    #
    # render screens
    if selected_screen == "Introduction":
        st.sidebar.markdown('---')
        # reading markdown
        st.markdown(open("./home.md").read())
    elif selected_screen == "Solution":
        screen1_render()
    elif selected_screen == "Parameter Fine-tuning":
        screen2_render()
    
    #
    # about on side bar
    st.sidebar.markdown("---")
    st.sidebar.markdown("# About \n This app has been developed by [Pavan] (https://github.com/pavancs) \
     using [Streamlit](https://streamlit.io/).")


def screen1_render():
    """
    render page
    """
    algo = st.radio("Select the algorithm to run fuzzy matching",
                    ('Fuzzywuzzy', 'TF-IDF', 'Both'))

    if st.button("Run"):
        #load data
        reference,query = load_data()

        if algo == 'Fuzzywuzzy' or algo == 'Both':
            with st.spinner('Running Fuzzywuzzy..'):
                t = time.time()
                reference_id = []
                score = []
                for text in query.text:
                    temp = process.extractOne(text, reference.text)
                    reference_id.append(temp[2] + 1)
                    score.append(temp[1])

                query['reference_id'] = reference_id
                query['score'] = score
                query = query.drop(['text'], axis=1)
                query_fuzzywuzzy = query.copy()
            st.success('Took %s secs to finish fuzzywuzzy model.' % str(time.time()-t))
        if algo == 'TF-IDF' or algo == 'Both':
            with st.spinner('Running TF-IDF..'):
                t = time.time()
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
                query = query.drop(['text'], axis=1)
                query_tfidf = query
            st.success('Took %s secs to finish TF-IDF model.' % str(time.time() - t))



        st.markdown("# Results Summary")
        st.markdown('---')
        if algo == "Both":
            st.markdown("*Percentage match between both solution :* **%s**" % str(
                (sum(query_fuzzywuzzy['reference_id'] == query_tfidf['reference_id'])) * 100 /
                len(query_fuzzywuzzy['reference_id'])))

        if algo == "Both" or algo == 'Fuzzywuzzy':
            st.markdown("*FuzzyWuzzy, average score:* **%s**" % np.average(query_fuzzywuzzy['score']))
            st.markdown("*FuzzyWuuzy, number of times score > 70:* **%s / %s**" % (
            sum(query_fuzzywuzzy['score'] > 70), len(query_fuzzywuzzy['score'])))

        if algo == "Both" or algo == 'TF-IDF':
            st.markdown("*TF-IDF, average score:* **%s**" % np.average(query_tfidf['score']))
            st.markdown("*TF-IDF, number of times score > 0.6:* **%s / %s**" % (
            sum(query_tfidf['score'] > 0.6), len(query_tfidf['score'])))
        st.markdown('---')

        # random results
        st.markdown("** Sample random results, **")
        for i in [np.random.randint(len(query)) for t in range(5)]:
            row = query.iloc[i]
            st.markdown("*Query row,*")
            st.dataframe(row.loc[['id', 'name', 'address', 'city', 'cuisine']])

            if algo == "Both" or algo == 'TF-IDF':
                matched_text = reference[reference['id'] == row['reference_id']]
                matched_text = matched_text.iloc[0]
                st.markdown("*TF-IDF matched reference row*")
                st.dataframe(matched_text.loc[['id', 'name', 'address', 'city', 'cuisine']])
            if algo == "Both" or algo == 'Fuzzywuzzy':
                matched_text = reference[reference['id'] == query_fuzzywuzzy.loc[i, 'reference_id']]
                matched_text = matched_text.iloc[0]
                st.markdown("*Fuzzywuzzy matched reference row*")
                st.dataframe(matched_text.loc[['id', 'name', 'address', 'city', 'cuisine']])
            st.markdown('---')
    return



def load_data():
    """
    load data and pre-process
    """
    reference = pd.read_csv("data/reference.csv")
    query = pd.read_csv("data/query.csv")

    query = query.drop(['reference_id', 'score(optional)'], axis=1)
    reference['text'] = reference.apply(text_preprocess, axis=1)
    query['text'] = query.apply(text_preprocess, axis=1)

    return reference,query


def text_preprocess(row):
    """
    funtion to pre-process with following steps,
        * Handle special characters like, " . " and " ' "
        * Handle text inside ()
        * Handle entity synonyms in column city

    args,
        row: dataframe row
    """

    entity_synonyms = {'new york city': 'new york'}

    text = row['name'] + row['address'] + row['city'] + row['cuisine']
    text = re.sub(r'\([^)]*\)', '', text)
    text = text.replace('.', '')
    text = text.replace("'", '')
    text = text.replace("`", '')
    text = text.replace(',', ' ')
    text = text.replace('&', 'and')

    # entity resolution
    for key, value in entity_synonyms.items():
        if key in text:
            text = text.replace(key, value)

    text = re.sub(' +', ' ', text)
    text = text.strip()

    return text


def n_grams(text, n=3):
    """
    function to character level ngrams
     * add padding to text
     * create character level ngrams


    args,
        text: text to convert to ngrams
        n : ngrams 'n' value
    """

    text = " " + text + " "
    ngrams = zip(*[text[i:] for i in range(n)])

    return [''.join(ngram) for ngram in ngrams]


def screen2_render():
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


if __name__ == '__main__':
    main()
