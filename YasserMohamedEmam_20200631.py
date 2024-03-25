from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.preprocessing import normalize
from documentGenerator import generate_documents
import pandas as pd
import re


def preprocess_text(text):

    # convert text to lowercases
    text = text.lower()

    # Remove Special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenizartion
    tokens = word_tokenize(text)

    # Remove Stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatizer_token = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return lemmatizer_token


def tfidfFunction(documents):
    # # This is for the TF part
    count_vectorizer = CountVectorizer()
    tf_matrix = count_vectorizer.fit_transform(documents)

    # This is for the IDF part
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_matrix = tfidf_transformer.fit_transform(tf_matrix)

    # We multiply the TF and IDF parts to generate the TF-IDF matrix
    tfidf_matrix = tf_matrix.multiply(tfidf_matrix)

    # Get Normalized TFIDF
    tfidf_normalized = normalize(tfidf_matrix, norm="l2", axis=1)

    # Print TFIDF matrix
    print("TF-IDF Matrix:")
    print(tfidf_normalized.toarray())
    print(tfidf_normalized.shape)

    feature_names = count_vectorizer.get_feature_names_out()

    # get tfidf vector for first document
    first_document_vector = tfidf_normalized[0]

    # print the scores
    df = pd.DataFrame(
        first_document_vector.T.todense(),
        index=feature_names,
        columns=["tfidf"],
    )
    df.sort_values(by=["tfidf"], ascending=False)

    print(df)

    return tfidf_matrix.toarray()


def main():

    # generate documents using OpenAI
    topics = ["Programming", "AI", "Technology"]

    documents = []

    for i in range(len(topics)):
        documents.append(generate_documents(topics[i]))

        prep_doc = preprocess_text(documents[i])

        print("Pre Processed: ")
        print(prep_doc)

        unique_words = set(prep_doc)
        print("Unique Words:")
        print(unique_words)

    # Using Built in
    tfidfFunction(documents)


if __name__ == "__main__":
    main()


#     documents = [
#         """Programming is the process of creating instructions that can be executed by a computer to perform specific tasks. Programmers use languages such as
# Python, Java, and C++ to write these instructions and create software applications, websites, and more. Programming requires problem-solving skills, logical thinking, and attention to detail to effectively code and debug programs. It is a versatile skill that is in high demand in various industries as technology continues to advance.""",
#         """AI, short for Artificial Intelligence, refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI technologies are designed to perform tasks such as speech recognition, decision-making, visual perception, and language translation. These technologies have the ability to analyze large amounts of data, recognize patterns, and make predictions. AI has the potential to revolutionize various
# industries, improve efficiency, and drive innovation in the coming years.""",
#         """Technology has become an integral part of our daily lives, shaping the way we communicate, work, and live. From smartphones and computers to advanced medical equipment and self-driving cars, technology continues to advance at a rapid pace, driving innovation and changing the way we interact with the world. With the evolution of artificial intelligence, virtual reality, and other emerging technologies, the possibilities for the future are endless. As we continue to embrace and adapt to new technologies, it's important to consider the impact they have on society and how we can harness their potential for the greater good.""",
#     ]
