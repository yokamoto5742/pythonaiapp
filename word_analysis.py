import streamlit as st
import spacy
import pandas as pd

nlp = spacy.load('ja_ginza')
pos_dic = {'名詞': 'NOUN', '代名詞': 'PRON', '固有名詞': 'PROPN', '動詞': 'VERB'}

uploaded_file = st.file_uploader('csvファイルをアップロードしてください', type='csv')
select_pos = st.sidebar.multiselect(
    '品詞選択:',
    list(pos_dic.keys()),
    ['名詞']
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    tg_col = st.sidebar.selectbox(
        '対象列を選択してください',
        list(data.columns)
    )
    if tg_col is not None:
        data = data.dropna()
        input_text = data[tg_col]
        input_text = ' '.join(str(item) for item in input_text)

        if st.button('実行'):
            doc = nlp(input_text)
            output_word = []
            tg_pos = [pos_dic[x] for x in select_pos]
            for comment in data['comment']:
                doc = nlp(comment)
                for token in doc:
                    if token.pos_ in tg_pos:
                        output_word.append(token.lemma_)

            output_df = pd.DataFrame({"Word": output_word})
            output_df = output_df.groupby('Word').size().reset_index(name='Count')
            output_df.sort_values(by='Count', ascending=False, inplace=True)

            st.dataframe(output_df)
            st.bar_chart(data=output_df.head(5).set_index('Word'))
