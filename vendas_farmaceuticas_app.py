# Importação das bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import textwrap
import requests
import warnings
import folium
from geopy.geocoders import Nominatim
import unicodedata
import time
import json
from scipy import stats
from streamlit_folium import st_folium
import openai
import streamlit as st
import os
from streamlit_chat import message as msg
import streamlit_embedcode
from io import StringIO
import time

warnings.filterwarnings("ignore")


# Defina a variável de ambiente STREAMLIT_SERVER_MAX_UPLOAD_SIZE
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1500"


@st.cache_data
def ajusta_dados(df):

    import pandas as pd
    import numpy as np
    
    # Deletando a coluna CID10
    df.drop('CID10', axis=1, inplace=True)

    # Imputando os valores para a feature PRINCIPIO_ATIVO
    moda_princ = df['PRINCIPIO_ATIVO'].mode()[0]
    df['PRINCIPIO_ATIVO'].fillna(moda_princ , inplace=True)

    # Imputando os valores para a feature SEXO
    moda_sexo = df['SEXO'].mode()[0]
    df['SEXO'].fillna(moda_sexo , inplace=True)

    # Imputando os valores para as features IDADE e UNIDADE_IDADE
    moda_uni_idade = df.loc[df['IDADE']==1, 'IDADE'].mode()[0]  #calcula a moda
    df['UNIDADE_IDADE'].fillna(1, inplace = True)               #preenche com o valor a categoria 1 (anos)
    df['IDADE'].fillna(moda_uni_idade, inplace = True)          #preenche com o valor da moda da idade em anos

    df['IDADE_ANOS'] = df['IDADE']
    df['IDADE_ANOS'] = np.where(df['UNIDADE_IDADE'] == 1, df['IDADE_ANOS'], np.floor(df['IDADE_ANOS'] / 12))
    #mantém a idade, se UNIDADE_IDADE for em anos, ou converte, se UNIDADE_IDADE for em meses

    # Excluindo a variável UNIDADE_IDADE
    df.drop('UNIDADE_IDADE', axis=1, inplace=True)
    df.drop('IDADE', axis=1, inplace=True)

    # Convertendo a variável TIPO_RECEITUÁRIO para 'object'
    df['TIPO_RECEITUARIO'] = df['TIPO_RECEITUARIO'].astype(object)  

    # Convertendo a variável SEXO para categorias M e F
    df.loc[df['SEXO']==1,'SEXO'] = 'M'
    df.loc[df['SEXO']==2,'SEXO'] = 'F'
    # global ctrl_ajusta
    ctrl_ajusta = True

    return df


@st.cache_data
def insight_distribuicao(df1, df2, df3):
    # Agrupe df1 por 'IDADE_ANOS' e calcule os percentuais
    df1_percentual = df1['IDADE_ANOS'].value_counts(normalize=True).reset_index()
    df1_percentual.columns = ['IDADE_ANOS', 'Percentual']

    # Agrupe df2 por 'SEXO' e calcule os percentuais
    df2_percentual = df2['SEXO'].value_counts(normalize=True).reset_index()
    df2_percentual.columns = ['SEXO', 'Percentual']

    # Agrupe df3 por 'QTD_VENDIDA' e calcule os percentuais
    df3_percentual = df3['QTD_VENDIDA'].value_counts(normalize=True).reset_index()
    df3_percentual.columns = ['QTD_VENDIDA', 'Percentual']

    # Crie uma string de texto para cada DataFrame de percentuais
    texto_df1 = "\n".join(df1_percentual.astype(str).values.flatten())
    texto_df2 = "\n".join(df2_percentual.astype(str).values.flatten())
    texto_df3 = "\n".join(df3_percentual.astype(str).values.flatten())

    # Mensagens para a API do ChatGPT
    messages = [
        {"role": "system", "content": '''Interprete os dados e gere insights de negócio.
         Os dados são referentes às vendas de medicamentos no Brasil.
         A primeira análise é sobre a distribuição de vendas em relação a 'IDADE'.
         A segunda análise é sobre a distribuição de vendas em relação a 'SEXO'.
         A terceira análise é sobre a distribuição de vendas em relação a 'Quantidade vendida em cada venda'.
         Para cada um, gere os insights de negócio contemplando estratégias de marketing,
         de cross-sell e upsell, em forma de tópicos.
         O texto deve ser escrito na forma de tópicos.
         A análise de cada distribuição deverá ser feita num tópico diferente.
         Tenha atenção para a correta separação das análises, pois o input do chatGPT vai concatenar os três conjuntos de dados."'''},
        {"role": "user", "content": texto_df1},
        {"role": "user", "content": texto_df2},
        {"role": "user", "content": texto_df3}
    ]

    # Chamada da API para gerar interpretação
    retorno_openai = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1000, n=1)

    # Extrai a resposta da API
    resposta = retorno_openai['choices'][0]['message']['content']

    return resposta



@st.cache_data
def insight_metadados(df):
    # Concatenação dos dados do DataFrame em uma única string para o ChatGPT
    texto_para_interpretacao = "\n".join(df.astype(str).values.flatten())

    # Mensagens para a API do ChatGPT
    messages = [
        {"role": "system", "content": '''Interprete os dados como se você estivesse fazendo uma análise exploratória preliminar.
         Para cada variável, comente sobre:
         - valores missing, o que podem significar, como podem afetam na análise e recomedações do que fazer;
         - cardinalidade, o que podem significar, como podem afetam na análise e recomedações do que fazer;
         - tipo de variável (float,int, object, etc), o que podem significar, como podem afetam na análise e recomedações do que fazer.

         Se a variável não possuir valores nulos, não énecessário escrever sobre isso. Caso contrário, escreva e faça sugestão de que isso pode afetar na análise.
         
         Mantenha a boa formatação do texto. Escreve na forma de tópicos e mantenha espaçamento ideal entre as linhas."'''},
        {"role": "user", "content": texto_para_interpretacao},]

    # Chamada da API para gerar interpretação
    retorno_openai = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1000, n=1 )

    # Extrai a resposta da API
    resposta = retorno_openai['choices'][0]['message']['content']

    return resposta

@st.cache_data
def insight_principio_ativo_por_estado(df):
    # Concatenação dos dados do DataFrame em uma única string para o ChatGPT
    texto_para_interpretacao = "\n".join(df.astype(str).values.flatten())

    # Mensagens para a API do ChatGPT
    messages = [
        {"role": "system", "content": '''Interprete os dados do dataframe e gere insights de negócio.
         O dataframe contém dados de vendas de medicamentos, portanto, os insights gerados deverão levar 
         em conta os percentuais de vendas dos três princípios ativos mais vendidos por estado.
         Você deverá gerar alguns poucos insights ou observações mais interessantes, de forma bem resumida, porém,
         escreva na forma de tópicos, com uma boa formatação de texto, se possível, utilizando marckdowns.
         Caso algum medicamento chame a atenção por seu percentual, ou por ser um caso isolado, faça
         um breve comentário sobre ele. Não precisa inserir título antes de escrever o texto.
         Você também não precisa fazernenhum cálculo. No dataset já há as informações de estado,
         princípios ativos mais vendidos e percentual de venda de cada.
         Você também não precisa analisar totais de vendas em um mesmo estado.
         Escreva na forma de tópicos, mantendo espaçamento entre as linhas de cada tópico, de forma que facilite a leitura.'''},
        {"role": "user", "content": texto_para_interpretacao},]

    # Chamada da API para gerar interpretação
    retorno_openai = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1000, n=1 )

    # Extrai a resposta da API
    resposta = retorno_openai['choices'][0]['message']['content']

    return resposta


@st.cache_data
def insight_principio_ativo_por_regiao(df):
    # Concatenação dos dados do DataFrame em uma única string para o ChatGPT
    texto_para_interpretacao = "\n".join(df.astype(str).values.flatten())

    # Mensagens para a API do ChatGPT
    messages = [
        {"role": "system", "content": '''Interprete os dados do dataframe e gere insights de negócio
         com base nos percentuais de vendas dos três princípios ativos mais vendidos por região."'''},
        {"role": "user", "content": texto_para_interpretacao},]

    # Chamada da API para gerar interpretação
    retorno_openai = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1000, n=1 )

    # Extrai a resposta da API
    resposta = retorno_openai['choices'][0]['message']['content']

    return resposta


@st.cache_data
def insight_vendas_por_estado(df):
    # Concatenação dos dados do DataFrame em uma única string para o ChatGPT
    texto_para_interpretacao = "\n".join(df.astype(str).values.flatten())

    # Mensagens para a API do ChatGPT
    messages = [
        {"role": "system", "content": '''Interprete os dados do dataframe e gere insights de negócio.
         O dataframe contém dados de vendas de medicamentos, portanto, os insights 
         deverão levar em conta os percentuais de venda total de medicamentos por estado."'''},
        {"role": "user", "content": texto_para_interpretacao},]

    # Chamada da API para gerar interpretação
    retorno_openai = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1000, n=1 )

    # Extrai a resposta da API
    resposta = retorno_openai['choices'][0]['message']['content']

    return resposta


@st.cache_data
def insight_vendas_por_regiao(df):
    # Concatenação dos dados do DataFrame em uma única string para o ChatGPT
    texto_para_interpretacao = "\n".join(df.astype(str).values.flatten())

    # Mensagens para a API do ChatGPT
    messages = [
        {"role": "system", "content": '''Interprete os dados do dataframe e gere insights de negócio.
         O dataframe contém dados de vendas de medicamentos, portanto, os insights 
         deverão levar em conta os percentuais de venda total de medicamentos por região."'''},
        {"role": "user", "content": texto_para_interpretacao},]

    # Chamada da API para gerar interpretação
    retorno_openai = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1000, n=1 )

    # Extrai a resposta da API
    resposta = retorno_openai['choices'][0]['message']['content']

    return resposta

def mapa_percentual_venda_por_regiao(df):

    # Carregue o arquivo GeoJSON especificando a codificação correta (ISO-8859-1)
    # with open(r"G:\Meu Drive\JACKSON SOUZA CORRÊA\DATA SCIENCE\Scripts VS Code\Data Science\app-vendas-farmaceuticas\uf.json", encoding="ISO-8859-1") as geojson_data:
    with open("uf.json", encoding="ISO-8859-1") as geojson_data:
        geojson = json.load(geojson_data)

    # Função para definir a cor com base na região (usando "REGIAO" como atributo)
    def assign_color_by_region(feature):
        regiao = feature['properties']['REGIAO']  # Atualizado para o nome correto do atributo
        # Defina cores diferentes para cada região
        cores_por_regiao = {
            'Norte': 'blue',
            'Nordeste': 'red',
            'Sudeste': 'green',
            'Sul': 'orange',
            'Centro-Oeste': 'purple'
        }
        return {
            'fillColor': cores_por_regiao.get(regiao, 'gray'),
            'color': 'black',  # Cor da borda
            'weight': 1,       # Espessura da borda
            'fillOpacity': 0.6 # Opacidade do preenchimento
        }

    # Adicione o GeoJSON ao mapa, definindo a função de estilo
    mapa = folium.Map(location=[-14.2350, -47.879], zoom_start=4)  # Defina o zoom e a localização iniciais
    folium.GeoJson(geojson, style_function=assign_color_by_region).add_to(mapa)

    # Carregue o DataFrame 'df' com seus dados
    # Substitua esta parte pelo carregamento real do DataFrame 'df'
    # Exemplo: df = pd.read_csv('seuarquivo.csv')

    # Função para calcular o percentual de vendas por região com base no DataFrame 'df'
    def calcular_percentual_vendas_por_regiao(df):
        # Crie um mapeamento de UF para região (substitua 'UF' e 'REGIAO' pelos nomes corretos de colunas)
        uf_regiao = {
            'AC': 'Norte',
            'AL': 'Nordeste',
            'AP': 'Norte',
            'AM': 'Norte',
            'BA': 'Nordeste',
            'CE': 'Nordeste',
            'DF': 'Centro-Oeste',
            'ES': 'Sudeste',
            'GO': 'Centro-Oeste',
            'MA': 'Nordeste',
            'MT': 'Centro-Oeste',
            'MS': 'Centro-Oeste',
            'MG': 'Sudeste',
            'PA': 'Norte',
            'PB': 'Nordeste',
            'PR': 'Sul',
            'PE': 'Nordeste',
            'PI': 'Nordeste',
            'RJ': 'Sudeste',
            'RN': 'Nordeste',
            'RS': 'Sul',
            'RO': 'Norte',
            'RR': 'Norte',
            'SC': 'Sul',
            'SP': 'Sudeste',
            'SE': 'Nordeste',
            'TO': 'Norte'
        }

        # Adicione uma coluna de região ao DataFrame 'df'
        df['REGIAO'] = df['UF_VENDA'].map(uf_regiao)

        # Agregue as vendas por região
        vendas_por_regiao = df.groupby('REGIAO').size().reset_index(name='TOTAL_VENDAS')

        # Calcule o total geral de vendas
        total_geral_vendas = vendas_por_regiao['TOTAL_VENDAS'].sum()

        # Calcule o percentual de vendas por região
        vendas_por_regiao['PERCENTUAL_VENDAS'] = (vendas_por_regiao['TOTAL_VENDAS'] / total_geral_vendas) * 100

        return vendas_por_regiao

    # Calcule o percentual de vendas por região com base em 'df'
    percentual_vendas_por_regiao = calcular_percentual_vendas_por_regiao(df)

    # Função para determinar as coordenadas do centro de cada região
    def coordenadas_centro_por_regiao():
        coordenadas = {
            'Norte': [-3.4653, -62.2159],
            'Nordeste': [-9.6502, -37.7397],
            'Sudeste': [-21.9969, -43.2067],
            'Sul': [-27.5954, -48.5480],
            'Centro-Oeste': [-15.7801, -47.9292]
        }
        return coordenadas

    # Função para adicionar marcadores com informações de percentual de vendas por região
    def add_markers():
        coordenadas = coordenadas_centro_por_regiao()
        for index, row in percentual_vendas_por_regiao.iterrows():
            regiao = row['REGIAO']
            percentual_vendas = row['PERCENTUAL_VENDAS']
            coords = coordenadas.get(regiao)
            if coords:
                popup_text = f'''<div style=' font-size: 14px;'>REGIÃO: {regiao}<br>
                PERCENTUAL DE VENDAS: {percentual_vendas:.2f}%'''
                folium.Marker(coords, tooltip=regiao, popup=folium.Popup(popup_text , max_width=1000)).add_to(mapa)

    # Adicione os marcadores ao mapa
    add_markers()

    # Exibindo o mapa por percentual de vendas por região
    return mapa, percentual_vendas_por_regiao


def mapa_principio_ativo_por_estado(df):
    # Carregue o arquivo GeoJSON especificando a codificação correta (ISO-8859-1)
    # with open(r"G:\Meu Drive\JACKSON SOUZA CORRÊA\DATA SCIENCE\Scripts VS Code\Data Science\app-vendas-farmaceuticas\uf.json", encoding="ISO-8859-1") as geojson_data:
    with open("uf.json", encoding="ISO-8859-1") as geojson_data:
        geojson = json.load(geojson_data)

    # Função para definir a cor com base em alguma lógica (aqui, estamos usando aleatoriamente)
    def assign_random_color(feature):
        import random
        return {
            'fillColor': f"#{random.randint(0, 0xFFFFFF):06x}",  # Gera uma cor hexadecimal aleatória
            'color': 'black',  # Cor da borda
            'weight': 1,       # Espessura da borda
            'fillOpacity': 0.6 # Opacidade do preenchimento
        }

    # Crie um objeto de mapa centrado nas coordenadas médias do GeoJSON (para visualização)
    mapa_est = folium.Map(location=[-15.700, -47.879], zoom_start=4)  # Defina o zoom e a localização iniciais
    folium.GeoJson(geojson, style_function=assign_random_color).add_to(mapa_est)

    # Função para calcular os três principais princípios ativos mais vendidos por estado com base no DataFrame 'df'
    def calcular_principios_ativos_mais_vendidos_por_estado(df):
        # Agrupe os princípios ativos por estado e conte as ocorrências de cada um
        principios_ativos_por_estado = df.groupby(['UF_VENDA', 'PRINCIPIO_ATIVO'])['PRINCIPIO_ATIVO'].count().reset_index(name='TOTAL_VENDAS')

        # Ordene o DataFrame para cada estado pelos princípios ativos mais vendidos
        principios_ativos_por_estado = principios_ativos_por_estado.sort_values(by=['UF_VENDA', 'TOTAL_VENDAS'], ascending=[True, False])

        # Calcule o total de vendas por estado
        total_vendas_por_estado = principios_ativos_por_estado.groupby('UF_VENDA')['TOTAL_VENDAS'].sum().reset_index(name='TOTAL_VENDAS_ESTADO')

        # Junte o DataFrame dos princípios ativos com o total de vendas por estado
        principios_ativos_por_estado = principios_ativos_por_estado.merge(total_vendas_por_estado, on='UF_VENDA')

        # Calcule o percentual de vendas para cada princípio ativo em relação ao total de vendas por estado
        principios_ativos_por_estado['PERCENTUAL_VENDAS'] = round(
            (principios_ativos_por_estado['TOTAL_VENDAS'] / principios_ativos_por_estado['TOTAL_VENDAS_ESTADO']) * 100 , 2)

        # Crie um novo DataFrame para armazenar os três principais princípios ativos por estado
        top_principios_ativos_por_estado = principios_ativos_por_estado.groupby('UF_VENDA').head(3)

        return top_principios_ativos_por_estado

    # Calcule os três principais princípios ativos mais vendidos por estado com base em 'df'
    top_principios_ativos_por_estado = calcular_principios_ativos_mais_vendidos_por_estado(df)

    # Função para determinar as coordenadas do centro de cada estado
    def coordenadas_centro_por_estado():
        coordenadas = {
            'AC': [-8.77, -70.55],
            'AL': [-9.71, -35.73],
            'AP': [1.41, -51.77],
            'AM': [-3.47, -65.10],
            'BA': [-12.96, -38.50],
            'CE': [-3.71, -38.54],
            'DF': [-15.78, -47.93],
            'ES': [-19.19, -40.34],
            'GO': [-16.64, -49.31],
            'MA': [-2.55, -44.30],
            'MT': [-12.64, -55.42],
            'MS': [-20.51, -54.54],
            'MG': [-18.10, -44.38],
            'PA': [-5.53, -52.29],
            'PB': [-7.06, -35.55],
            'PR': [-24.89, -51.55],
            'PE': [-8.28, -35.07],
            'PI': [-8.28, -43.68],
            'RJ': [-22.84, -43.15],
            'RN': [-5.22, -36.52],
            'RS': [-30.01, -51.22],
            'RO': [-10.83, -63.34],
            'RR': [1.99, -61.33],
            'SC': [-27.59, -48.55],
            'SP': [-23.55, -46.63],
            'SE': [-10.98, -37.07],
            'TO': [-9.96, -47.34]
        }
        return coordenadas

    def add_markers():
        coordenadas = coordenadas_centro_por_estado()
        for estado, principios_ativos in top_principios_ativos_por_estado.groupby('UF_VENDA'):
            coords = coordenadas.get(estado)
            if coords:
                popup_text = f"<div style=' font-size: 14px;'>ESTADO: {estado}<br><br>PRINCÍPIO ATIVO E PERCENTUAL DE VENDA:<br><br>"
                for i, row in principios_ativos.iterrows():
                    popup_text += f"• {row['PRINCIPIO_ATIVO']}: {row['PERCENTUAL_VENDAS']}%<br>"

                folium.Marker(
                    location=coords,
                    popup=folium.Popup(popup_text, max_width=1000),
                    tooltip=estado,
                ).add_to(mapa_est)


    # Adicione os marcadores ao mapa
    add_markers()

    # Exiba o mapa
    return mapa_est, top_principios_ativos_por_estado


def mapa_principio_ativo_por_regiao(df):
    # PRINCIPIO ATIVO POR REGIÃO

    # Carregue o arquivo GeoJSON especificando a codificação correta (ISO-8859-1)
    # with open(r"G:\Meu Drive\JACKSON SOUZA CORRÊA\DATA SCIENCE\Scripts VS Code\Data Science\app-vendas-farmaceuticas\uf.json", encoding="ISO-8859-1") as geojson_data:
    with open("uf.json", encoding="ISO-8859-1") as geojson_data:
        geojson = json.load(geojson_data)

    # Função para definir a cor com base na região (usando "REGIAO" como atributo)
    def assign_color_by_region(feature):
        regiao = feature['properties']['REGIAO']
        # Defina cores diferentes para cada região
        cores_por_regiao = {
            'Norte': 'blue',
            'Nordeste': 'red',
            'Sudeste': 'green',
            'Sul': 'orange',
            'Centro-Oeste': 'purple'
        }
        return {
            'fillColor': cores_por_regiao.get(regiao, 'gray'),
            'color': 'black',  # Cor da borda
            'weight': 1,       # Espessura da borda
            'fillOpacity': 0.6 # Opacidade do preenchimento
        }

    # Adicione o GeoJSON ao mapa, definindo a função de estilo
    mapa = folium.Map(location=[-14.788, -47.879], zoom_start=4)  # Defina o zoom e a localização iniciais
    folium.GeoJson(geojson, style_function=assign_color_by_region).add_to(mapa)


    # Função para calcular o percentual de vendas de cada princípio ativo por região
    def calcular_percentual_de_vendas(df):
        df_copy = df.copy()  # Crie uma cópia do DataFrame original

        # Crie um mapeamento de UF para região (substitua 'UF' e 'REGIAO' pelos nomes corretos de colunas)
        uf_regiao = {
            'AC': 'Norte',
            'AL': 'Nordeste',
            'AP': 'Norte',
            'AM': 'Norte',
            'BA': 'Nordeste',
            'CE': 'Nordeste',
            'DF': 'Centro-Oeste',
            'ES': 'Sudeste',
            'GO': 'Centro-Oeste',
            'MA': 'Nordeste',
            'MT': 'Centro-Oeste',
            'MS': 'Centro-Oeste',
            'MG': 'Sudeste',
            'PA': 'Norte',
            'PB': 'Nordeste',
            'PR': 'Sul',
            'PE': 'Nordeste',
            'PI': 'Nordeste',
            'RJ': 'Sudeste',
            'RN': 'Nordeste',
            'RS': 'Sul',
            'RO': 'Norte',
            'RR': 'Norte',
            'SC': 'Sul',
            'SP': 'Sudeste',
            'SE': 'Nordeste',
            'TO': 'Norte'
        }

        # Adicione uma coluna de região ao DataFrame 'df_copy'
        df_copy['REGIAO'] = df_copy['UF_VENDA'].map(uf_regiao)

        # Calcular o total de vendas por região
        total_vendas_por_regiao = df_copy.groupby('REGIAO').size().reset_index(name='TOTAL_VENDAS')

        # Calcular o percentual de vendas de cada princípio ativo por região
        df_copy = df_copy.groupby(['REGIAO', 'PRINCIPIO_ATIVO']).size().reset_index(name='QUANTIDADE')
        df_copy = df_copy.merge(total_vendas_por_regiao, on='REGIAO')
        df_copy['PERCENTUAL'] = (df_copy['QUANTIDADE'] / df_copy['TOTAL_VENDAS']) * 100

        return df_copy

    # Calcule o percentual de vendas de cada princípio ativo por região
    df_percentual = calcular_percentual_de_vendas(df)

    # Função para determinar as coordenadas do centro de cada região
    def coordenadas_centro_por_regiao():
        coordenadas = {
            'Norte': [-3.4653, -62.2159],
            'Nordeste': [-9.6502, -37.7397],
            'Sudeste': [-21.9969, -43.2067],
            'Sul': [-27.5954, -48.5480],
            'Centro-Oeste': [-15.7801, -47.9292]
        }
        return coordenadas

    # Função para adicionar marcadores com informações dos percentuais de vendas de princípios ativos por região
    def add_markers():
        coordenadas = coordenadas_centro_por_regiao()
        for regiao, group in df_percentual.groupby('REGIAO'):
            coords = coordenadas.get(regiao)
            if coords:
                # Ordenar os princípios ativos por percentual e pegar os 3 maiores
                group = group.sort_values(by='PERCENTUAL', ascending=False).head(3)
                popup_text = f"<div style=' font-size: 14px;'>REGIÃO: {regiao}<br><br>PRINCÍPIO ATIVO E PERCENTUAL DE VENDA:<br><br>"
                for index, row in group.iterrows():
                    principio_ativo = row['PRINCIPIO_ATIVO']
                    percentual = row['PERCENTUAL']
                    popup_text += f"• {principio_ativo}: {percentual:.2f}%<br>"
                popup_text += "</div>"
                folium.Marker(coords,
                            tooltip=regiao,
                            popup=folium.Popup(popup_text,max_width=1000 )).add_to(mapa)

    # Adicione os marcadores ao mapa
    add_markers()

    return mapa, df_percentual


def mapa_venda_por_estado(df):
    # Carregue o arquivo GeoJSON especificando a codificação correta (ISO-8859-1)
    # with open(r"G:\Meu Drive\JACKSON SOUZA CORRÊA\DATA SCIENCE\Scripts VS Code\Data Science\app-vendas-farmaceuticas\uf.json", encoding="ISO-8859-1") as geojson_data:
    with open("uf.json", encoding="ISO-8859-1") as geojson_data:
        geojson = json.load(geojson_data)

    # Função para definir a cor com base em alguma lógica (aqui, estamos usando aleatoriamente)
    def assign_random_color(feature):
        import random
        return {
            'fillColor': f"#{random.randint(0, 0xFFFFFF):06x}",  # Gera uma cor hexadecimal aleatória
            'color': 'black',  # Cor da borda
            'weight': 1,       # Espessura da borda
            'fillOpacity': 0.6 # Opacidade do preenchimento
        }

    # Crie um objeto de mapa centrado nas coordenadas médias do GeoJSON (para visualização)
    mapa_est = folium.Map(location=[-15.788, -47.879], zoom_start=4)  # Defina o zoom e a localização iniciais
    folium.GeoJson(geojson, style_function=assign_random_color).add_to(mapa_est)

    # Função para calcular o percentual de vendas por estado com base no DataFrame 'df'
    def calcular_percentual_vendas_por_estado(df):
        total_vendas_por_estado = df.groupby('UF_VENDA').size().reset_index(name='TOTAL_VENDAS')
        total_vendas_total = total_vendas_por_estado['TOTAL_VENDAS'].sum()
        total_vendas_por_estado['PERCENTUAL'] = (total_vendas_por_estado['TOTAL_VENDAS'] / total_vendas_total) * 100
        return total_vendas_por_estado

    # Calcule o percentual de vendas por estado com base em 'df'
    percentual_vendas_por_estado = calcular_percentual_vendas_por_estado(df)

    # Função para determinar as coordenadas do centro de cada estado
    def coordenadas_centro_por_estado():
        coordenadas = {
            'AC': [-8.77, -70.55],
            'AL': [-9.71, -35.73],
            'AP': [1.41, -51.77],
            'AM': [-3.47, -65.10],
            'BA': [-12.96, -38.50],
            'CE': [-3.71, -38.54],
            'DF': [-15.78, -47.93],
            'ES': [-19.19, -40.34],
            'GO': [-16.64, -49.31],
            'MA': [-2.55, -44.30],
            'MT': [-12.64, -55.42],
            'MS': [-20.51, -54.54],
            'MG': [-18.10, -44.38],
            'PA': [-5.53, -52.29],
            'PB': [-7.06, -35.55],
            'PR': [-24.89, -51.55],
            'PE': [-8.28, -35.07],
            'PI': [-8.28, -43.68],
            'RJ': [-22.84, -43.15],
            'RN': [-5.22, -36.52],
            'RS': [-30.01, -51.22],
            'RO': [-10.83, -63.34],
            'RR': [1.99, -61.33],
            'SC': [-27.59, -48.55],
            'SP': [-23.55, -46.63],
            'SE': [-10.98, -37.07],
            'TO': [-9.96, -47.34]
        }
        return coordenadas

    # Função para adicionar marcadores com informações de percentual de vendas por estado
    def add_markers():
        coordenadas = coordenadas_centro_por_estado()
        for index, row in percentual_vendas_por_estado.iterrows():
            estado = row['UF_VENDA']
            percentual_vendas = row['PERCENTUAL']
            coords = coordenadas.get(estado)
            if coords:
                popup_text = f"<div style=' font-size: 14px;'>ESTADO: {estado}<br>PERCENTUAL DE VENDAS: {percentual_vendas:.2f}%"
                folium.Marker(
                    location=coords,
                    popup=folium.Popup(popup_text, max_width=1000)
                ).add_to(mapa_est)

    # Adicione os marcadores ao mapa
    add_markers()

    # Exiba o mapa por estado
    return mapa_est, percentual_vendas_por_estado


# @st.cache_data
def plot_bar_2var(df, var_x, var_y, figsize, **kwargs):
    df_aux = df.copy()

    df_aux[var_x] = df_aux[var_x].astype(str)
    df_aux[var_y] = df_aux[var_y].astype(float)

    # Adicione os argumentos opcionais "top" e "ascending"
    ascending = kwargs.get('ascending', False)
    top = kwargs.get('top', None)

    # Ordene o DataFrame com base no parâmetro "ascending"
    df_aux = df_aux.sort_values(by=var_y, ascending=ascending)

    # Aplicar o argumento "top" se especificado
    if top is not None:
        df_aux = df_aux.head(top)

    x = df_aux[var_x].tolist()
    y = df_aux[var_y].tolist()

    color = kwargs.get('color', 'deepskyblue')
    decimals = kwargs.get('decimals', 1)
    pad = kwargs.get('pad', 30)
    dist = kwargs.get('dist', 1)
    label_font = kwargs.get('label_height', 9)
    txt_rot = kwargs.get('txt_rot', 'h')
    title = kwargs.get('title', f'Gráfico de Barras de "{var_x}" por "{var_y}"')
    unity = kwargs.get('unity', '')

    if txt_rot == 'h':
        rotation = 0
    elif txt_rot == 'v':
        rotation = 90
    else:
        rotation = txt_rot

    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(x=x, y=y, color=color, alpha=1)

    ax.set_title(f'{title}', pad=pad, fontdict={'fontsize': 12, 'weight': 'bold'})

    ax.set_xlabel(var_x)
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('')
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for i, v in enumerate(y):
        display_value = str(round(v, decimals))
        ax.text(i, v + dist, display_value + unity, ha='center', va='bottom', fontsize=label_font, rotation=rotation)

    plt.xticks(rotation=90)

    plt.show()


@st.cache_data
def plot_bar_h(df, var, figsize, **kwargs):
    df_aux = df.copy()
    df_aux[var] = df_aux[var].astype(str)

    # Adicione os argumentos opcionais "top" e "ascending"
    ascending = kwargs.get('ascending', False)
    top = kwargs.get('top', None)

    df_grouped = df_aux.groupby(var).size().sort_values(ascending=ascending)

    tot = df_grouped.sum()
    x = df_grouped.index

    mode = kwargs.get('mode', 'percent')
    cutoff = kwargs.get('corte', None)  # Novo argumento para o ponto de corte
    unity = kwargs.get('unity', '')

    if cutoff is not None:
        if mode == 'percent':
            cutoff_value = tot * (cutoff / 100)
            small_categories = df_grouped[df_grouped < cutoff_value].index
            df_grouped[f'Outros\n(<{cutoff}%)'] = df_grouped[small_categories].sum()
            df_grouped.drop(small_categories, inplace=True)
            x = df_grouped.index
        elif mode == 'absolute':  # Handle cutoff for 'absolute' mode
            small_categories = df_grouped[df_grouped < cutoff].index
            df_grouped[f'Outros\n(<{cutoff}{unity})'] = df_grouped[small_categories].sum()
            df_grouped.drop(small_categories, inplace=True)
            x = df_grouped.index

    color = kwargs.get('color', 'deepskyblue')
    decimals = kwargs.get('decimals', 1)
    pad = kwargs.get('pad', 30)
    dist = kwargs.get('dist', 1)
    label_font = kwargs.get('label_height', 9)
    txt_rot = kwargs.get('txt_rot', 'h')
    title = kwargs.get('title', f'Variável "{var}"')

    if mode == 'percent':
        # Ajuste das proporções de acordo com o corte
        y = 100 * (df_grouped.values / df_grouped.values.sum())
        ylabel = 'Percentual (%)'
    elif mode == 'absolute':
        y = df_grouped.values
        ylabel = 'Número Absoluto'
    else:
        raise ValueError("O modo de exibição deve ser 'percent' ou 'absolute'.")

    if top is not None:
        # Exibir apenas as 10 primeiras barras, mas ajustar as proporções
        x = x[:top]
        y = y[:top]

    if txt_rot == 'h':
        rotation = 0
    elif txt_rot == 'v':
        rotation = 90
    else:
        rotation = txt_rot

    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=figsize)

    # Use a função textwrap.fill para formatar os rótulos do eixo X
    x = [textwrap.fill(label, 15) for label in x]

    sns.barplot(x=y, y=x, color=color, alpha=1)  # Inverta x e y para gráfico de barras horizontais

    ax.set_xticklabels(ax.get_xticklabels(), visible=False)  # Oculte os rótulos no eixo X
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)  # Defina o tamanho dos rótulos no eixo Y
    ax.set_title(f'{title}', pad=pad, fontdict={'fontsize': 12, 'weight': 'bold'})
    ax.set_xlabel(ylabel)  # Use ylabel para o eixo X
    ax.set_ylabel('')
    plt.gca().set_xticks([])  # Oculte os ticks no eixo X
    plt.gca().set_xlabel('')
    ax.tick_params(axis='y', labelsize=9)  # Defina o tamanho dos rótulos no eixo Y
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)  # Oculte a linha do eixo X

    for i, v in enumerate(y):
        if mode == 'percent':
            display_value = str(round(v, decimals)) + " %"
        elif mode == 'absolute':
            display_value = str(int(v))
        ax.text(v + dist, i, display_value + ' ' + unity, ha='left', va='center', fontsize=label_font, rotation=rotation)

    plt.show()


@st.cache_data
def plot_bar(df, var, figsize, **kwargs):

    import textwrap  # Importe a função textwrap

    df_aux = df.copy()
    df_aux[var] = df_aux[var].astype(str)

    # Adicione os argumentos opcionais "top" e "ascending"
    ascending = kwargs.get('ascending', False)
    top = kwargs.get('top', None)

    df_grouped = df_aux.groupby(var).size().sort_values(ascending=ascending)

    tot = df_grouped.sum()
    x = df_grouped.index

    mode = kwargs.get('mode', 'percent')
    cutoff = kwargs.get('corte', None)  # Novo argumento para o ponto de corte
    unity = kwargs.get('unity', '')

    if cutoff is not None:
        if mode == 'percent':
            cutoff_value = tot * (cutoff / 100)
            small_categories = df_grouped[df_grouped < cutoff_value].index
            df_grouped[f'Outros\n(<{cutoff}%)'] = df_grouped[small_categories].sum()
            df_grouped.drop(small_categories, inplace=True)
            x = df_grouped.index
        elif mode == 'absolute':  # Handle cutoff for 'absolute' mode
            small_categories = df_grouped[df_grouped < cutoff].index
            df_grouped[f'Outros\n(<{cutoff}{unity})'] = df_grouped[small_categories].sum()
            df_grouped.drop(small_categories, inplace=True)
            x = df_grouped.index

    color = kwargs.get('color', 'deepskyblue')
    decimals = kwargs.get('decimals', 1)
    pad = kwargs.get('pad', 30)
    dist = kwargs.get('dist', 1)
    label_font = kwargs.get('label_height', 9)
    txt_rot = kwargs.get('txt_rot', 'h')
    title = kwargs.get('title', f'Variável "{var}"')


    if mode == 'percent':
        # Ajuste das proporções de acordo com o corte
        y = 100 * (df_grouped.values / df_grouped.values.sum())
        ylabel = 'Percentual (%)'
    elif mode == 'absolute':
        y = df_grouped.values
        ylabel = 'Número Absoluto'
    else:
        raise ValueError("O modo de exibição deve ser 'percent' ou 'absolute'.")

    if top is not None:
        # Exibir apenas as 10 primeiras barras, mas ajustar as proporções
        x = x[:top]
        y = y[:top]

    if txt_rot == 'h':
        rotation = 0
    elif txt_rot == 'v':
        rotation = 90
    else:
        rotation = txt_rot

    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=figsize)

    # Use a função textwrap.fill para formatar os rótulos do eixo X
    x = [textwrap.fill(label, 15) for label in x]

    sns.barplot(x=x, y=y, color=color, alpha=1)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), visible=False)
    ax.set_title(f'{title}', pad=pad, fontdict={'fontsize': 12, 'weight': 'bold'})
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('')
    ax.tick_params(axis='x', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for i, v in enumerate(y):
        if mode == 'percent':
            display_value = str(round(v, decimals)) + " %"
        elif mode == 'absolute':
            display_value = str(int(v))
        ax.text(i, v + dist, display_value + ' ' + unity, ha='center', va='bottom', fontsize=label_font, rotation=rotation)

    plt.show()



@st.cache_data
def plot_hist(df, var, figsize, **kwargs):

    # Extraindo os argumentos opcionais e atribuindo os valores default
    color = kwargs.get('color', 'deepskyblue')

    bins = kwargs.get('bins', 30)  # Número de bins no histograma

    pad = kwargs.get('pad', 30)

    label_size = kwargs.get('label_size', 9)

    txt_rot = kwargs.get('txt_rot', 'v')

    stat = kwargs.get('stat', 'frequency')

    grid = kwargs.get('grid', False)

    title = kwargs.get('title', None)


    if txt_rot == 'h':
        rotation = 0
    elif txt_rot == 'v':
        rotation = 90
    else:
        rotation = txt_rot

    # Gráfico
    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(data=df, x=var, bins=bins, color=color, ax=ax)

    if title==None:
        ax.set_title(f'Histograma da variável "{var}"', pad=pad, fontdict={'fontsize': 12, 'weight': 'bold'})
    else:
        ax.set_title(title)
    ax.set_xlabel(var,  fontsize=9)
    ax.set_ylabel(stat, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)
    ax.grid(grid)

    plt.xticks(rotation=rotation)

    plt.show()


@st.cache_data

def replace_outliers(df, **kwargs):
    """
    Substitui outliers por NaN em um DataFrame para cada variável numérica.

    :param df: DataFrame de entrada.
    :param threshold: Limiar para identificar outliers (opcional, padrão é 1.5).
    :return: DataFrame com outliers substituídos por NaN.
    """
    df_copy = df.copy()
    k = kwargs.get('k', 1.5)

    # Itera sobre as colunas numéricas
    for column in df_copy.select_dtypes(include=['float64', 'int64']).columns:
        # Calcula o primeiro e terceiro quartis
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)

        # Calcula o IQR (Intervalo Interquartil)
        IQR = Q3 - Q1

        # Define o limite inferior e superior para identificar outliers
        lower_limit = Q1 - k * IQR
        upper_limit = Q3 + k * IQR

        # Substitui os outliers por NaN
        df_copy[column] = df_copy[column].apply(lambda x: np.nan if x < lower_limit or x > upper_limit else x)

    return df_copy


@st.cache_data
def report_data(df, **kwargs):
    df2 = pd.DataFrame()
    df2['feature'] = list(df.columns)
    df2['tipo'] = df.dtypes.values
    df2['cardinal'] = df.nunique().values
    df2['qtd miss'] = df.isnull().sum().values
    df2['perc miss'] = round(100 * df2['qtd miss'] / df.shape[0] , 1)

    # Calcular quartis e limites de outliers apenas para colunas numéricas
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    Q1 = numeric_cols.quantile(0.25)    # primeiro quartil (percentil 25)
    Q3 = numeric_cols.quantile(0.75)    # terceiro quartil (percentil 75)
    IIQ = Q3 - Q1                      # intervalo interquartil
    k = kwargs.get('k', 1.5)           # Fator multiplicador
    LS = Q3 + k * IIQ                  # Limite superior
    LI = Q1 - k * IIQ                  # Limite inferior


    df2['qtd out sup'] = df[df.select_dtypes(include=['float64', 'int64', 'float32' , 'int32'])>LS].count().values
    df2['perc out sup'] = round(100 * df2['qtd out sup'] / df.shape[0] , 1)

    df2['qtd out inf'] = df[df.select_dtypes(include=['float64', 'int64', 'float32' , 'int32'])<LI].count().values
    df2['perc out inf'] = round(100 * df2['qtd out inf'] / df.count().values , 1)

    df2['qtd outliers'] = df2['qtd out sup'] + df2['qtd out inf']
    df2['perc outliers'] = round(100 * df2['qtd outliers'] / df.count().values , 1)

    df2.drop(['qtd out sup','perc out sup', 'qtd out inf', 'perc out inf'], axis=1, inplace=True)

    return df2



# Variável global para armazenar a resposta da IA
interpretação = None

# @st.cache_data # Use @st.cache para armazenar em cache os dados
# def carrega_dados(data):
#     df = pd.read_csv(data, sep=';' , encoding='latin-1')
#     return df

@st.cache_data
def exibe_sample(df):
    return df.sample(5)


# =====================================================================================
def main():

    # Configurações gerais
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.set_page_config()                                      # Tamanho padrão de página
    st.set_page_config(layout="wide")                           # Tamanho 'large' de página

    # =============================================================================================
    # CABEÇALHO
    # =============================================================================================
    
    # Título
    st.title('Pharmalytics app')
    st.subheader("Automação em Análise de vendas   :bar_chart:   :pill:")
    
    st.markdown("<hr>", unsafe_allow_html=True)                        # Linha separadora

    st.write("""Desenvolvido por Jackson Corrêa - v00 - setembro/2023<br>
    <a href='https://www.linkedin.com/in/jackson-corrêa' target='_blank'>Acesse meu LinkedIn</a>  |
               <a href='https://www.github.com/JacksonSCorrea' target='_blank'>Acesse meu GitHub</a>""" , 
               unsafe_allow_html=True)
    
    openai.api_key = st.text_input("Digite sua API Key da OpenAI", type='password')

    st.markdown("<hr>", unsafe_allow_html=True)                        # Linha separadora


    # =============================================================================================
    # OPÇÕES DE ORIGEM DOS DADOS
    # =============================================================================================

    opc = st.radio('Origem dos dados',['Carregar do portal','Carregar do computador'],
                   captions=['Esta opção pode levar até 5min para ser concluída','Esta opção demanda que o download do arquivo já tenha sido feito para o computador'])

    st.markdown("<hr>", unsafe_allow_html=True)                        # Linha separadora 

    if opc == 'Carregar do portal':

        # =============================================================================================
        # IMPORTAÇÃO DOS DADOS
        # =============================================================================================

        col1, col2, col3= st.columns([1,1,1])   #dentro do parêntes é o tamanho de cada coluna

        with col1:
            ano = st.selectbox('Ano', ('2018','2019','2020','2021'))
            ano=str(ano)

        with col2:
            # Dicionário que mapeia nomes de meses para valores numéricos
            lista_meses = {'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4, 'Maio': 5, 'Junho': 6, 
                    'Julho': 7, 'Agosto': 8, 'Setembro': 9, 'Outubro': 10, 'Novembro': 11, 'Dezembro': 12}

            # Selecionar o mês
            mes_select = st.selectbox('Mês', list(lista_meses.keys()))

            # Obter o valor numérico correspondente ao mês selecionado
            mes = lista_meses[mes_select]
            if mes < 10:
                mes = '0'+str(mes)
            else:
                mes=str(mes)


        # =============================================================================================
        # DEFINIÇÃO DA AMOSTRAGEM
        # =============================================================================================

        with col3:
        
            perc_amostra = st.slider('Percentual da amostra',1,100,10)
            
        # =============================================================================================
        # DOWNLOAD DOS DADOS
        # =============================================================================================

        st.markdown("<hr>", unsafe_allow_html=True)                        # Linha separadora   

        if st.button('Gerar análise'):

            inicio = time.time()

            with st.spinner('Baixando os dados...'):

                st.markdown("<hr>", unsafe_allow_html=True)                        # Linha separadora   
            
                periodo = ano + mes
                url = f'https://dados.anvisa.gov.br/dados/SNGPC/Industrializados/EDA_Industrializados_{periodo}.csv'
                
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Ler o CSV a partir do conteúdo da resposta com encoding e separador personalizados
                    content = response.content.decode('latin-1')
                    df = pd.read_csv(StringIO(content), sep=';')
                    
                else:
                    st.error(f"Erro ao carregar dados. Código de status HTTP: {response.status_code}")
            
            fim1 = time.time()
            tempo1 = fim1 - inicio
            st.write(f'Tempo de download dos dados: {round(tempo1/60 , 2)} min')
            st.success('Dados baixados com sucesso!')

            # =============================================================================================
            # CÓPIA E AJUSTE DOS DADOS
            # =============================================================================================

            st.markdown("<hr>", unsafe_allow_html=True)                        # Linha separadora
            with st.spinner('Ajustando os dados...'):

                # Embaralha e extrai amostra
                df = df.sample(frac=perc_amostra/100)

                # Fazendo cópia
                df_aux = df.copy()

                # # Ajuste dos dados
                df = ajusta_dados(df)

                st.write('Dataset:')

                # Exibe os dados
                st.dataframe(exibe_sample(df))

                # Exibe o tmaanho do dataset
                st.write(f'Tamanho da amostra: {df.shape}')


            # =============================================================================================
            # METADADOS
            # =============================================================================================

            st.markdown("<hr>", unsafe_allow_html=True)
            with st.spinner('Gerando metadados...'):
 
                col1, col2 = st.columns(2)

                with col1:
                    tabs = st.tabs(["Metadados"])                                  # Criando uma única aba
                    with tabs[0]:                                                  # Entrando na aba
                        df_report = report_data(df_aux)
                        st.dataframe(df_report, hide_index=True, height=610)

                with col2:
                    tabs = st.tabs(["Comentários"])                               # Criando uma única aba
                    with tabs[0]:
                        interpretação = insight_metadados(df_report)
                        st.markdown(f''' <div style="max-height: 610px; overflow-y: scroll;">
                            {interpretação}
                            </div> ''', unsafe_allow_html=True)


            st.markdown("<hr>", unsafe_allow_html=True)                             # Linha separadora 
            with st.spinner('Fazendo as análises...'):

                # =============================================================================================
                # DISTRIBUIÇÕES
                # =============================================================================================
                

                st.subheader('Distribuições')

                col1, col2 = st.columns(2) 

                df_hist = replace_outliers(df)                                           # Limpando outliers
                
                with col1:                                                               # Entra na coluna 1

                    tab1, tab2, tab3 = st.tabs(["Idade", "Sexo", "Unidades por venda"])  # Criando as abas

                    with tab1:                                                           # Entrando na aba 1
                        st.pyplot(plot_hist(df_hist, var='IDADE_ANOS',                   # Plota o histograma
                                            figsize=(4,4), bins=50,
                                            grid=True, stat='frequency',
                                            title=''))
                    
                    with tab3:                                                          # Entrando na aba 2
                        st.pyplot(plot_hist(df_hist, var='QTD_VENDIDA',
                                            figsize=(4,4), bins=50,
                                            grid=True, stat='frequency',
                                            title=''))

                    with tab2:                                                          # Entrando na aba 3
                        st.pyplot(plot_bar(df_hist, var='SEXO',figsize=(4,4),
                                        title=''))

                with col2:                                                              # Entra na coluna 2
                    tabs = st.tabs(["Comentários"])                                     # Criando uma única aba
                    with tabs[0]:
                        interpretação = insight_distribuicao(df_hist[['IDADE_ANOS']] , df_hist[['SEXO']] , df_hist[['QTD_VENDIDA']])
                        st.markdown(f''' <div style="max-height: 570px; overflow-y: scroll;">
                            {interpretação}
                            </div> ''', unsafe_allow_html=True)

                # =============================================================================================
                # ANÁLISE GEOGRÁFICA
                # =============================================================================================

                st.markdown("<hr>", unsafe_allow_html=True)                          # Linha separadora 
                st.subheader('Percentual de vendas por região')                      # Título
                mapa, df_mapa = mapa_percentual_venda_por_regiao(df)                 # Chama a função do mapa

                col1, col2 = st.columns(2)                                           # Dividir a tela em duas colunas

                with col1:                                                           # Criando a primeira coluna
                    
                    tab1, tab2 = st.tabs(["Mapa", "Gráfico"])                        # Criando as abas
                    
                    with tab1:                                                       # Primeira aba
                        st_folium(mapa, width=1000 , returned_objects=[]) 
                    
                    with tab2:                                                       # Segunda aba
                        # Dicionário para agrupar as regiões
                        estado_regiao = {
                            'AC': 'Norte', 'AL': 'Nordeste', 'AP': 'Norte', 'AM': 'Norte', 'BA': 'Nordeste', 'CE': 'Nordeste',
                            'DF': 'Centro-Oeste','ES': 'Sudeste','GO': 'Centro-Oeste','MA': 'Nordeste','MT': 'Centro-Oeste',
                            'MS': 'Centro-Oeste','MG': 'Sudeste','PA': 'Norte','PB': 'Nordeste','PR': 'Sul','PE': 'Nordeste',
                            'PI': 'Nordeste','RJ': 'Sudeste','RN': 'Nordeste','RS': 'Sul','RO': 'Norte','RR': 'Norte',
                            'SC': 'Sul','SP': 'Sudeste','SE': 'Nordeste','TO': 'Norte'}

                        # Crie o DataFrame auxiliar df_aux apenas com as colunas "Região" e "Percentual de vendas".
                        df_aux = pd.DataFrame(columns=['Região', '% Vendas'])

                        # Calcule o total de vendas por região.
                        total_por_regiao = df['UF_VENDA'].map(estado_regiao).value_counts().reset_index()
                        total_por_regiao.columns = ['Região', 'Total_Vendas']

                        # Calcule o percentual de vendas por região.
                        total_vendas_total = df.shape[0]
                        total_por_regiao['% Vendas'] = (total_por_regiao['Total_Vendas'] / total_vendas_total) * 100

                        # Atualize o DataFrame df_aux com as informações de vendas por região.
                        df_aux['Região'] = total_por_regiao['Região']
                        df_aux['% Vendas'] = total_por_regiao['% Vendas']

                        # plota o gráfico
                        st.pyplot(plot_bar_2var(df_aux, var_x = 'Região', var_y= '% Vendas', figsize = (8,6), unity='%', dist = 0.1, title='PERCENTUAL DE VENDAS POR REGIÃO'))
                        


                with col2:                                                           # Criando a segunda coluna
                    tabs = st.tabs(["Insights"])                                     # Criando uma única aba
                    with tabs[0]:                                                    # Aba
                        interpretação = insight_vendas_por_regiao(df_mapa)           # Chamando a função do ChatGPT   
                        st.markdown(f'''
                                <div style="max-height: 700px; overflow-y: scroll;">
                                {interpretação}
                                </div> ''', unsafe_allow_html=True)                  # Escreve a interpretação com limite de altura da caixa
                # ------------------------------------------------------------------------------------------------------------------------------------------------
                
                st.markdown("<hr>", unsafe_allow_html=True)                                                             # Linha separadora 
                st.subheader('Princípio ativo por região')                                                              # Título
                mapa, df_mapa = mapa_principio_ativo_por_regiao(df)                                                     # Chama a função do mapa

                col1, col2 = st.columns(2)                                                                              # Dividir a tela em duas colunas

                with col1:                                                                                              # Criando a primeira coluna
                    tabs = st.tabs(["Mapa"])                                                           # Criando as abas
                    with tabs[0]:                                                                                          # Primeira aba
                        st_folium(mapa, width=1000 , returned_objects=[]) 

                with col2:                                                                                              # Criando a segunda coluna
                    tabs = st.tabs(["Insights"])                                                                        # Criando uma única aba
                    with tabs[0]:
                        # Crie um dataset resumo com os três principais princípios ativos por região
                        df_resumo = df_mapa.groupby(['REGIAO', 'PRINCIPIO_ATIVO'])['PERCENTUAL'].sum().reset_index()
                        df_resumo = df_resumo.sort_values(by=['REGIAO', 'PERCENTUAL'], ascending=[True, False])
                        df_resumo = df_resumo.groupby('REGIAO').head(3)                                                 # Aba
                        interpretação = insight_principio_ativo_por_regiao(df_resumo)                                   # Chamando a função do ChatGPT   
                        st.markdown(f''' <div style="max-height: 700px; overflow-y: scroll;">
                                        {interpretação}
                                        </div> ''', unsafe_allow_html=True)                                             # Escreve a interpretação com limite de altura da caixa


                # ------------------------------------------------------------------------------------------------------------------------------------------------

                st.markdown("<hr>", unsafe_allow_html=True)                          # Linha separadora 
                st.subheader('Percentual de vendas por estado')                      # Título
                mapa, df_mapa = mapa_venda_por_estado(df)                            # Chama a função do mapa

                col1, col2 = st.columns(2)                                           # Dividir a tela em duas colunas

                with col1:                                                           # Criando a primeira coluna
                    tab1, tab2 = st.tabs(["Mapa", "Gráfico"])                        # Criando as abas
                    with tab1:                                                       # Primeira aba
                        st_folium(mapa, width=1000 , returned_objects=[]) 
                    with tab2:
                        st.pyplot(plot_bar_h(df, 'UF_VENDA', figsize=(6,5), label_height=7, txt_rot='h', dist=0.3, title='', pad=0))



                with col2:                                                           # Criando a segunda coluna
                    tabs = st.tabs(["Insights"])                                     # Criando uma única aba
                    with tabs[0]:                                                    # Aba
                        interpretação = insight_vendas_por_estado(df_mapa)           # Chamando a função do ChatGPT   
                        st.markdown(f'''
                                <div style="max-height: 700px; overflow-y: scroll;">
                                {interpretação}
                                </div> ''', unsafe_allow_html=True)                  # Escreve a interpretação com limite de altura da caixa
                
                # ------------------------------------------------------------------------------------------------------------------------------------------------
                
                st.markdown("<hr>", unsafe_allow_html=True)                          # Linha separadora 
                st.subheader('Princípio ativo por estado')                           # Título
                mapa, df_mapa = mapa_principio_ativo_por_estado(df)                  # Chama a função do mapa

                col1, col2 = st.columns(2)                                           # Dividir a tela em duas colunas

                with col1:                                                           # Criando a primeira coluna
                    tabs = st.tabs(["Mapa"])                                         # Criando uma única aba
                    with tabs[0]:                                                       # Primeira aba
                        st_folium(mapa, width=1000 , returned_objects=[])

                with col2:                                                           # Criando a segunda coluna
                    tabs = st.tabs(["Insights"])                                     # Criando uma única aba
                    with tabs[0]:                                                    # Aba
                        interpretação = insight_principio_ativo_por_estado(df_mapa)  # Chamando a função do ChatGPT   
                        st.markdown(f'''
                                <div style="max-height: 700px; overflow-y: scroll;">
                                {interpretação}
                                </div> ''', unsafe_allow_html=True)                  # Escreve a interpretação com limite de altura da caixa
                    
                # ----------------------------------------------------------------------------------------------
            
            st.success('Análises concluídas!')
            fim2 = time.time()
            tempo2 = fim2 - inicio
            st.write(f'Tempo total de execução (download + análises): {round(tempo2/60 , 2)} min')



# =======================================================================================================================================
# =======================================================================================================================================
# =======================================================================================================================================
        


    elif opc == 'Carregar do computador':

        # IMPORTAÇÃO DOS DADOS
        # =============================================================================================
    
        inicio = time.time()

        col1, col2, col3= st.columns([1,1,1])

        with col1:

            upload_file = st.file_uploader('Carregar dados', type=['csv'])

        with col2:
            separador = st.selectbox("Escolha o separador", [",", ";", "\t", "|"])
            encoding = st.selectbox("Escolha o encoding", ["latin-1","utf-8", "ISO-8859-1", "cp1252"])


        # =============================================================================================
        # DEFINIÇÃO DA AMOSTRAGEM
        # =============================================================================================

        with col3:
        
            perc_amostra = st.slider('Percentual da amostra',1,100,10)
            
        # =============================================================================================
        # DOWNLOAD DOS DADOS
        # =============================================================================================

        st.markdown("<hr>", unsafe_allow_html=True)                        # Linha separadora   
        
        # Verifica se o dataset foi carregado
        if upload_file:

            if st.button('Gerar análise'):

                df = pd.read_csv(upload_file, encoding=encoding , sep=separador)

                st.success('Dados carregados com sucesso!')

                # =============================================================================================
                # CÓPIA E AJUSTE DOS DADOS
                # =============================================================================================

                st.markdown("<hr>", unsafe_allow_html=True)                        # Linha separadora
                with st.spinner('Ajustando os dados...'):

                    # Embaralha e extrai amostra
                    df = df.sample(frac=perc_amostra/100)

                    # Fazendo cópia
                    df_aux = df.copy()

                    # # Ajuste dos dados
                    df = ajusta_dados(df)

                    st.write('Dataset:')

                    # Exibe os dados
                    st.dataframe(exibe_sample(df))

                    # Exibe o tmaanho do dataset
                    st.write(f'Tamanho da amostra: {df.shape}')


                # =============================================================================================
                # METADADOS
                # =============================================================================================

                st.markdown("<hr>", unsafe_allow_html=True)
                with st.spinner('Gerando metadados...'):
    
                    col1, col2 = st.columns(2)

                    with col1:
                        tabs = st.tabs(["Metadados"])                                  # Criando uma única aba
                        with tabs[0]:                                                  # Entrando na aba
                            df_report = report_data(df_aux)
                            st.dataframe(df_report, hide_index=True, height=610)

                    with col2:
                        tabs = st.tabs(["Comentários"])                               # Criando uma única aba
                        with tabs[0]:
                            interpretação = insight_metadados(df_report)
                            st.markdown(f''' <div style="max-height: 610px; overflow-y: scroll;">
                                {interpretação}
                                </div> ''', unsafe_allow_html=True)


                st.markdown("<hr>", unsafe_allow_html=True)                             # Linha separadora 
                with st.spinner('Fazendo as análises...'):

                    # =============================================================================================
                    # DISTRIBUIÇÕES
                    # =============================================================================================
                    

                    st.subheader('Distribuições')

                    col1, col2 = st.columns(2) 

                    df_hist = replace_outliers(df)                                           # Limpando outliers
                    
                    with col1:                                                               # Entra na coluna 1

                        tab1, tab2, tab3 = st.tabs(["Idade", "Sexo", "Unidades por venda"])  # Criando as abas

                        with tab1:                                                           # Entrando na aba 1
                            st.pyplot(plot_hist(df_hist, var='IDADE_ANOS',                   # Plota o histograma
                                                figsize=(4,4), bins=50,
                                                grid=True, stat='frequency',
                                                title=''))
                        
                        with tab3:                                                          # Entrando na aba 2
                            st.pyplot(plot_hist(df_hist, var='QTD_VENDIDA',
                                                figsize=(4,4), bins=50,
                                                grid=True, stat='frequency',
                                                title=''))

                        with tab2:                                                          # Entrando na aba 3
                            st.pyplot(plot_bar(df_hist, var='SEXO',figsize=(4,4),
                                            title=''))

                    with col2:                                                              # Entra na coluna 2
                        tabs = st.tabs(["Comentários"])                                     # Criando uma única aba
                        with tabs[0]:
                            interpretação = insight_distribuicao(df_hist[['IDADE_ANOS']] , df_hist[['SEXO']] , df_hist[['QTD_VENDIDA']])
                            st.markdown(f''' <div style="max-height: 570px; overflow-y: scroll;">
                                {interpretação}
                                </div> ''', unsafe_allow_html=True)

                    # =============================================================================================
                    # ANÁLISE GEOGRÁFICA
                    # =============================================================================================

                    st.markdown("<hr>", unsafe_allow_html=True)                          # Linha separadora 
                    st.subheader('Percentual de vendas por região')                      # Título
                    mapa, df_mapa = mapa_percentual_venda_por_regiao(df)                 # Chama a função do mapa

                    col1, col2 = st.columns(2)                                           # Dividir a tela em duas colunas

                    with col1:                                                           # Criando a primeira coluna
                        
                        tab1, tab2 = st.tabs(["Mapa", "Gráfico"])                        # Criando as abas
                        
                        with tab1:                                                       # Primeira aba
                            st_folium(mapa, width=1000 , returned_objects=[]) 
                        
                        with tab2:                                                       # Segunda aba
                            # Dicionário para agrupar as regiões
                            estado_regiao = {
                                'AC': 'Norte', 'AL': 'Nordeste', 'AP': 'Norte', 'AM': 'Norte', 'BA': 'Nordeste', 'CE': 'Nordeste',
                                'DF': 'Centro-Oeste','ES': 'Sudeste','GO': 'Centro-Oeste','MA': 'Nordeste','MT': 'Centro-Oeste',
                                'MS': 'Centro-Oeste','MG': 'Sudeste','PA': 'Norte','PB': 'Nordeste','PR': 'Sul','PE': 'Nordeste',
                                'PI': 'Nordeste','RJ': 'Sudeste','RN': 'Nordeste','RS': 'Sul','RO': 'Norte','RR': 'Norte',
                                'SC': 'Sul','SP': 'Sudeste','SE': 'Nordeste','TO': 'Norte'}

                            # Crie o DataFrame auxiliar df_aux apenas com as colunas "Região" e "Percentual de vendas".
                            df_aux = pd.DataFrame(columns=['Região', '% Vendas'])

                            # Calcule o total de vendas por região.
                            total_por_regiao = df['UF_VENDA'].map(estado_regiao).value_counts().reset_index()
                            total_por_regiao.columns = ['Região', 'Total_Vendas']

                            # Calcule o percentual de vendas por região.
                            total_vendas_total = df.shape[0]
                            total_por_regiao['% Vendas'] = (total_por_regiao['Total_Vendas'] / total_vendas_total) * 100

                            # Atualize o DataFrame df_aux com as informações de vendas por região.
                            df_aux['Região'] = total_por_regiao['Região']
                            df_aux['% Vendas'] = total_por_regiao['% Vendas']

                            # plota o gráfico
                            st.pyplot(plot_bar_2var(df_aux, var_x = 'Região', var_y= '% Vendas', figsize = (8,6), unity='%', dist = 0.1, title='PERCENTUAL DE VENDAS POR REGIÃO'))
                            


                    with col2:                                                           # Criando a segunda coluna
                        tabs = st.tabs(["Insights"])                                     # Criando uma única aba
                        with tabs[0]:                                                    # Aba
                            interpretação = insight_vendas_por_regiao(df_mapa)           # Chamando a função do ChatGPT   
                            st.markdown(f'''
                                    <div style="max-height: 700px; overflow-y: scroll;">
                                    {interpretação}
                                    </div> ''', unsafe_allow_html=True)                  # Escreve a interpretação com limite de altura da caixa
                    # ------------------------------------------------------------------------------------------------------------------------------------------------
                    
                    st.markdown("<hr>", unsafe_allow_html=True)                                                             # Linha separadora 
                    st.subheader('Princípio ativo por região')                                                              # Título
                    mapa, df_mapa = mapa_principio_ativo_por_regiao(df)                                                     # Chama a função do mapa

                    col1, col2 = st.columns(2)                                                                              # Dividir a tela em duas colunas

                    with col1:                                                                                              # Criando a primeira coluna
                        tabs = st.tabs(["Mapa"])                                                           # Criando as abas
                        with tabs[0]:                                                                                          # Primeira aba
                            st_folium(mapa, width=1000 , returned_objects=[]) 

                    with col2:                                                                                              # Criando a segunda coluna
                        tabs = st.tabs(["Insights"])                                                                        # Criando uma única aba
                        with tabs[0]:
                            # Crie um dataset resumo com os três principais princípios ativos por região
                            df_resumo = df_mapa.groupby(['REGIAO', 'PRINCIPIO_ATIVO'])['PERCENTUAL'].sum().reset_index()
                            df_resumo = df_resumo.sort_values(by=['REGIAO', 'PERCENTUAL'], ascending=[True, False])
                            df_resumo = df_resumo.groupby('REGIAO').head(3)                                                 # Aba
                            interpretação = insight_principio_ativo_por_regiao(df_resumo)                                   # Chamando a função do ChatGPT   
                            st.markdown(f''' <div style="max-height: 700px; overflow-y: scroll;">
                                            {interpretação}
                                            </div> ''', unsafe_allow_html=True)                                             # Escreve a interpretação com limite de altura da caixa


                    # ------------------------------------------------------------------------------------------------------------------------------------------------

                    st.markdown("<hr>", unsafe_allow_html=True)                          # Linha separadora 
                    st.subheader('Percentual de vendas por estado')                      # Título
                    mapa, df_mapa = mapa_venda_por_estado(df)                            # Chama a função do mapa

                    col1, col2 = st.columns(2)                                           # Dividir a tela em duas colunas

                    with col1:                                                           # Criando a primeira coluna
                        tab1, tab2 = st.tabs(["Mapa", "Gráfico"])                        # Criando as abas
                        with tab1:                                                       # Primeira aba
                            st_folium(mapa, width=1000 , returned_objects=[]) 
                        with tab2:
                            # st.pyplot(plot_bar(df, 'UF_VENDA', figsize=(12,10), label_height=11, txt_rot='v', dist=0.3, title='PERCENTUAL DE VENDAS POR ESTADO'))
                            st.pyplot(plot_bar_h(df, 'UF_VENDA', figsize=(6,5), label_height=7, txt_rot='h', dist=0.3, title='', pad=0))



                    with col2:                                                           # Criando a segunda coluna
                        tabs = st.tabs(["Insights"])                                     # Criando uma única aba
                        with tabs[0]:                                                    # Aba
                            interpretação = insight_vendas_por_estado(df_mapa)           # Chamando a função do ChatGPT   
                            st.markdown(f'''
                                    <div style="max-height: 700px; overflow-y: scroll;">
                                    {interpretação}
                                    </div> ''', unsafe_allow_html=True)                  # Escreve a interpretação com limite de altura da caixa
                    
                    # ------------------------------------------------------------------------------------------------------------------------------------------------
                    
                    st.markdown("<hr>", unsafe_allow_html=True)                          # Linha separadora 
                    st.subheader('Princípio ativo por estado')                           # Título
                    mapa, df_mapa = mapa_principio_ativo_por_estado(df)                  # Chama a função do mapa

                    col1, col2 = st.columns(2)                                           # Dividir a tela em duas colunas

                    with col1:                                                           # Criando a primeira coluna
                        tabs = st.tabs(["Mapa"])                                         # Criando uma única aba
                        with tabs[0]:                                                    # Primeira aba
                            st_folium(mapa, width=1000 , returned_objects=[]) 

                    with col2:                                                           # Criando a segunda coluna
                        tabs = st.tabs(["Insights"])                                     # Criando uma única aba
                        with tabs[0]:                                                    # Aba
                            interpretação = insight_principio_ativo_por_estado(df_mapa)  # Chamando a função do ChatGPT   
                            st.markdown(f'''
                                    <div style="max-height: 700px; overflow-y: scroll;">
                                    {interpretação}
                                    </div> ''', unsafe_allow_html=True)                  # Escreve a interpretação com limite de altura da caixa
                        
                    # ----------------------------------------------------------------------------------------------
                
                st.success('Análises concluídas!')
                fim = time.time()
                tempo = fim - inicio
                st.write(f'Tempo total de execução (upload + análises): {round(tempo/60 , 2)} min')

if __name__ == "__main__":
    main()