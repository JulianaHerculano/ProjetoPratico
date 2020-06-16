# Importando as Bibliotecas
import streamlit as st
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import base64



def main ():

    st.image('logo.png')
    st.title('SISTEMA DE RECOMENDAÇÃO')
    st.subheader('Como o sistema funciona?')
    st.markdown('Você entra com o seu portfólio ("ids" das empresas que hoje fazem parte da sua lista de clientes). Nós iremos analisar seu portfólio e recomendar novas empresas para fazerem parte do seu time! PREPARADO?')

    # Lendo os arquivos
    csv_populacao_tratada = pd.read_csv('dataframe_populacao_tratada.csv')

    # Criando DataFrame para cada um dos arquivos
    df_populacao = pd.DataFrame(csv_populacao_tratada)

    file = st.file_uploader('Entre com seu PORTFÓLIO (.csv): ', type='csv')

    if file is not None:

        #st.markdown('Carregamos seu arquivo.')
        st.markdown('STATUS: Iniciando a análise do seu PORTFÓLIO')
        # DataFrame (df_entrada): é o DataFrame criado a partir da lista de clientes do usuário (upload arquivo)
        # Queremos apenas o valor de "id". As outras colunas "não interessam, pois já temos as informações na população"
        df_entrada_auxiliar = pd.read_csv(file)
        df_entrada = pd.DataFrame(df_entrada_auxiliar['id'])
        #df_entrada = pd.DataFrame(df_portfolio1['id'])

        # Criando a coluna "target" no DataFrame que recebemos de entrada (que contém a lista das empresas (portfolio))
        # E atribuindo o valor target = 1
        df_entrada['target'] = 1

        df_completo = pd.merge(df_populacao, df_entrada, how='outer', on='id')

        if ('Unnamed: 0' in df_completo.columns):
            df_completo.drop(columns='Unnamed: 0', inplace=True)

        # Colocando zeros onde "Target" é NaN (ou seja, != 1)
        df_completo['target'].fillna(value=0, inplace=True)

        # Definindo TREINO e TESTE
        # Salvando os valores de "id" em (ArmazenandoID)
        df_ArmazenandoID = df_completo['id']
        #df_completo.drop(columns = 'id', inplace=True)

        # Definindo a base de TREINO (portfolio): Y_treino, X_treino

        # Salvando em "df_completo_TREINO" as linhas do DataFrame que tem "target = 1"
        df_completo_TREINO = df_completo.loc[(df_completo['target'] == 1)]
        Y_treino = df_completo_TREINO['target']
        X_treino = df_completo_TREINO.drop(columns='target')

        # Definindo a base de TEST: X_teste, Y_predito

        # (população, excluindo as empresas que fazem parte da lista da empresa, ou seja, "target = 0")
        # Salvando em "df_completo_TESTE" as linhas do DataFrame que tem "target = 0"

        df_completo_TESTE = df_completo.loc[(df_completo['target'] == 0)]
        X_teste = df_completo_TESTE.drop(columns='target')
        # Y_predito: vai ser o resultado da previsão do Modelo

        lista_ID_teste = list(X_teste['id'])
        lista_ID_treino = list(X_treino['id'])

        # Salvando os IDs da base se TESTE em um DataFrame
        # Assim vai facilitar para juntar os IDs com o resultado predito
        df_ID_teste = pd.DataFrame({'id': lista_ID_teste})

        # Apagando a coluna ID de treino e teste para treinar e testar o modelo
        X_teste.drop(columns='id', inplace=True)
        X_treino.drop(columns='id', inplace=True)

        # MODELO: SVM.OneClass

        one_class_SVM = OneClassSVM(kernel='rbf')
        resultado_OneClass_FIT = one_class_SVM.fit(X_treino)

        linhas = X_teste.shape[0]
        colunas = X_teste.shape[1]

        resultado_OneClass_PREDICT = one_class_SVM.predict(X_teste)

        # Vamos salvar em (df_resultado) as colunas: "ID" e "resultado_OneClass_PREDICT
        df_resultado = df_ID_teste

        # Criando um dataFrame com apenas uma coluna que contém o resultado predito pelo modelo
        df_resultado_OneClass_PREDICT = pd.DataFrame({'resultado_OneClass_PREDICT': resultado_OneClass_PREDICT})

        # Vamos acrescentar a coluna do resultado predito ao DataFrame (df_resultado)
        df_resultado = df_resultado.join(df_resultado_OneClass_PREDICT)

        # Armazenando de forma separada as recomendações e não recomendações do modelo
        df_auxiliar_NAO = df_resultado[df_resultado['resultado_OneClass_PREDICT'] == -1]
        df_auxiliar_SIM = df_resultado[df_resultado['resultado_OneClass_PREDICT'] == 1]

        # (df_recomendar) contém os IDs das empresas recomendadas pelo modelo
        df_recomendar = df_auxiliar_SIM['id']

        # Retornando o resultado:
        #st.markdown('TERMINAMOS NOSSA ANÁLISE.')
        st.markdown('   ')
        st.header('RESULTADO')
        tamanho_dataframe = df_recomendar.shape[0]

        st.markdown('Quantidade de empresas recomendadas: ')
        st.write(tamanho_dataframe)

        st.slider('Defina a quantidade de empresas que deseja visualizar:', 1, tamanho_dataframe)
        st.write(df_recomendar)

        st.markdown('Deseja fazer o Download da lista de empresas recomendadas?')
        # Dando a opção do usuário fazer Download do arquivo com as empresas recomendadas
        csv = df_recomendar.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}">Download</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
    main()


