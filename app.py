import numpy as np
import streamlit as st
import tensorflow as tf
import os
import cv2

preprocess_input = tf.keras.applications.resnet50.preprocess_input


NOME_MODELO = 'mask_model.h5'
OPCAO_VIDEO = 'Video'
OPCAO_IMAGEM = 'Image'


def carregar_modelo():
    """
        Carrega o modelo a partir do caminho models/mask_model.h5
    """
    caminho = os.path.join('models', NOME_MODELO)
    modelo = tf.keras.models.load_model(caminho)
    return modelo

# Realiza as predições
modelo = carregar_modelo()


def classificar(img):
    """ 
        Classifica uma imagem retornando 1 - com máscara e 0 - sem máscara
    """
    # Realiza o preprocessamento
    img_resize = tf.image.resize(img, [160, 160])
    img_processada  = preprocess_input(img_resize)
    img_processada = tf.expand_dims(img_processada, 0)
    
 
    previsoes = modelo.predict(img_processada)
    previsoes = tf.nn.sigmoid(previsoes)

    previsao = tf.where(previsoes > 0.5, 1, 0)
    return previsao.numpy()[0][0]
    # [[0]] -> [0] -> 0

def main():
    st.header('Com máscara ou sem máscara')

    opcoes = [OPCAO_IMAGEM, OPCAO_VIDEO]
    escolha = st.radio('Qual modo você gostaria?', opcoes)

    if escolha == OPCAO_IMAGEM:
        file = st.file_uploader('Carregue alguma imagem aqui', type=['jpg', 'png', 'jpeg'])
        if file is not None:
            img = file.read() # Lê a imagem realizada do upload

            img = tf.image.decode_image(img, channels=3).numpy() # Codifica de byte a imagem
            
            st.image(img) # Mostra a imagem na tela

            resultado = classificar(img)
            if resultado == 1:
                st.caption("Está com máscara.")
            else:
                st.caption("Está sem máscara.")
        
    elif escolha == OPCAO_VIDEO:
        framewindow = st.image([])
        camera = cv2.VideoCapture(0)
        placeholder = st.empty()
        while escolha == OPCAO_VIDEO :
            _, frame = camera.read()

            if frame is not None :
                frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1)

                resultado = classificar(frame)
                if resultado == 1:
                    cv2.putText(frame, 'Esta com mascara.', (100,350), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale= 1.5, color = (0,225,0))
                    placeholder.text("Está com máscara.")
                else:
                    cv2.putText(frame, 'Esta sem mascara.', (100,350), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale= 1.5, color = (255,0,0))
                    placeholder.text("Está sem máscara.")
                
                framewindow.image(frame)



if __name__ == '__main__': 
    main()