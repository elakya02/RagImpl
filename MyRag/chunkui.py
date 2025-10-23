import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

def dude_fun(query_context , number_of_similarity):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    data_db = 'chunkdb.db'
    conn = sqlite3.connect(data_db)
    print("connection successfull")
    curr = conn.cursor()

    statement = "Select Context , encoded_matrix from Chunking"

    curr.execute(statement)
    vector = curr.fetchall()

    context = [row[0] for row in vector]

    embeddings = [row[1] for row in vector]
    print(type(embeddings[0]))

    # print(context)
    # print(embeddings)

    #To get the dimension of the embeddings
    dim = model.get_sentence_embedding_dimension()
    print(dim)

    #Convert the type of embeddings
    num_embed = [np.frombuffer(row , dtype=np.float32).reshape(1 , -1) for row in embeddings]

    num_embed = np.vstack(num_embed)
    print(num_embed.shape)


    #Query Embedding
    Query = query_context
    query_embed = model.encode(Query)
    #Faced problem here
    query_embed_2 = query_embed.reshape(1 , -1)
    print(query_embed_2.shape)
    print(num_embed.shape)
    similar_search = cosine_similarity(query_embed_2 , num_embed)[0]
    print(len(similar_search))
    print(similar_search)

    argument_sorting = np.argsort(similar_search)[::-1][:number_of_similarity]
    print(argument_sorting)
    list1 = [context[value] for value in argument_sorting]
    return list1
st.title("Finding Similar Chunks")
Query = st.text_input("Enter your Query to find the similarity")
num = st.text_input("Enter no of similarity")

if st.button("Find Similar Chunks"):
    if Query.strip():
        with st.spinner("Finding similar chunks..."):
            result_list = dude_fun(Query, int(num))
        st.success("Here are the most similar chunks:")
        i = 1
        for value in enumerate(result_list):
            st.markdown(f"{i}. {value}")
            i = i+1
    else:
        st.warning("Please enter a valid query before clicking the button.")