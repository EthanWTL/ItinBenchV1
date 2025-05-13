from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import json
from typing import List, Dict

class Nearby:
    def __init__(self):
        pass
    def run(self, notebook):
        #self.notebook = notebook
        allbusinesses = []
        hotel_substrings = ["accommodation","hotel"]
        attraction_substrings = ['attraction']
        for cat in notebook:
            for sub in hotel_substrings:
                #print(sub)
                #print(cat['Description'])
                if sub in cat['Description'].lower():
                    cols = ['name','latitude','longitude']
                    hotels = cat['Content'][cols].values.tolist()
                    break

            for sub in attraction_substrings:
                if sub in cat['Description'].lower():
                    cols = ['name','latitude','longitude']
                    attractions = cat['Content'][cols].values.tolist()

        allbusinesses.extend(attractions)
        allbusinesses.extend(hotels)
        #print(allbusinesses)
        allbusiness_names = []
        allbusiness_cordinates = []

        for business in allbusinesses:
            allbusiness_cordinates.append([business[1],business[2]])
            allbusiness_names.append([business[0]])
        coordinates = np.array(allbusiness_cordinates)
        #k = int(prompt['day'][0].strip()[0]) 0.23 1.24
        # k=10 0.27 1.3
        k = int(len(coordinates)/5)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(coordinates)
        # Cluster labels
        allBusiness_clusterNumbers = kmeans.labels_

        clusterInfo = pd.DataFrame({'business': allbusiness_names, 'cluster number': allBusiness_clusterNumbers})
        clusterInfo['cluster number'] = clusterInfo['cluster number'].apply(lambda x: 'Cluster_' + str(x))
        clusterInfo = clusterInfo.sort_values('cluster number')

        cluster_dict = clusterInfo.groupby('cluster number')['business'].apply(list).to_dict()
        return cluster_dict