#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

from numpy import dot
from numpy.linalg import norm


# In[ ]:


st.header('Artificial Intelligence Prediction of rice applications')
st.write('made by RDA and Sejong Univ.')
st.write(' ')
st.write('(쌀가루의 특성을 토대로 응용 제품군을 추천합니다)')

uploaded_file = st.file_uploader("Choose an excel file", type="xlsx")

if uploaded_file:
    sample = pd.read_excel(uploaded_file)

    st.write('[입력 데이터] ')
    st.dataframe(sample)
    
    df = pd.read_excel('rda-rice-applications.xlsx', header=0)
    
    df_filter = df[[
    'protein',
    'amylose',
    'lipid',
    'peak', 'trough', 'breakdown', 'final', 'setback', 'peak_time', 'pasting_temp',
    'labels',
    'label',
        ]]
    
    df_filter = df_filter.dropna()

    df1 = df_filter.drop(["labels", "label"], axis = 1)
    
    y_data = df_filter['labels']
    y_data_label = df_filter['label']
    
    scaler = MinMaxScaler()
    
    x_data = scaler.fit_transform(df1.values)
    
    sample_normal = scaler.transform(sample.values)
    
    # 저장된 모델을 불러옴
    with open('pca_model.pkl', 'rb') as file:
        loaded_pca = pickle.load(file)

    # 불러온 모델을 사용하여 변환 수행
    pca_result = loaded_pca.transform(sample_normal)
    
    # pca 저장 엑셀화일 불러오기 
    df_pca = pd.read_excel('df_pca.xlsx', header=0)
    
    # 유클리드 거리 계산
    df_pca1 = df_pca[["Component_1", "Component_2"]]
    df_pca1 = np.array(df_pca1, dtype=np.float32)
    
    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))
    
    # 결과 출력
    distance_list = []
    cosine_list = []
    for j in range(len(df_pca1)):
        distance = np.linalg.norm(df_pca1[j]- pca_result[0])
        distance_list.append([distance, df_pca['label'][j]])
        cosine = cos_sim(df_pca1[j], pca_result[0])
        cosine_list.append([cosine, df_pca['label'][j]])
    
    #results = pd.DataFrame(distance_list, columns = ['distance', 'label'])
    results = pd.DataFrame(distance_list, columns = ['유사도분석(euclidean distance)', '추천 제품군'])
    
    a1 = results.sort_values('유사도분석(euclidean distance)', ascending=True)
    
    recommend = a1.reset_index().drop(['index'], axis=1).head(5)

    st.write(' ')
    st.write('[결과] ')
    st.dataframe(recommend)
    st.write(' ')
    

