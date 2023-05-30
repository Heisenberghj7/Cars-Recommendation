import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

New_Cars=pd.read_csv("Ready_Cars_Data.csv")
choosen_columns = ['More_Info', 'Moteur', 'Energie', 'Architecture', 'Cylindrée', 'Puissance fiscale', 'Puissance maxi.', 'Couple maxi.']
New_Cars['features'] = New_Cars[choosen_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
french_stop_words = [
    'a', 'à', 'afin', 'alors', 'après', 'au', 'aucun', 'aussi', 'autre', 'avec',
    'avoir', 'avant', 'c', 'ce', 'cela', 'ces', 'ceux', 'chaque', 'ci', 'comme'
]
tfv = TfidfVectorizer(min_df=10,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = french_stop_words)


tfv_matrix = tfv.fit_transform(New_Cars['features'])
similarity = cosine_similarity(tfv_matrix)

def get_recommendations(name, cosine_similarities=similarity, data=New_Cars, top_n=10):
    # Get the index of the car with the given name
    idx = data[data['name'] == name].index[0]
    
    # Get the similarity scores for all cars
    similarity_scores = list(enumerate(cosine_similarities[idx]))
    
    # Sort the cars based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N most similar cars
    top_similar_cars = [data.iloc[i[0]]['name'] for i in similarity_scores[1:top_n+1]]
    
    return top_similar_cars



