
import numpy as np
import streamlit as st

W1 = np.load('W1_momt.npy')
W2 = np.load('W2_momt.npy')
W3 =  np.load('W_momt.npy')



def sigmoid(z):
    '''In this function, we will compute the sigmoid(z)'''
    # we can use this function in forward and backward propagation

    return (1/(1+np.exp(-z)))

def predict_target(X,W1,W2):
    X1= np.log(X)
    z11,z12,z13 = np.dot(np.transpose(W1),X1.reshape(-1,1))
    o11 = sigmoid(z11)
    o12 = sigmoid(z12)
    o13 = sigmoid(z13)
    y_hat = o11*W2[0] + o12*W2[1] + o13*W2[2]

    return np.exp(y_hat[0])


def perceptron_predict_target(X,W):
    X1 = np.log(X)
    y_hat = np.dot(np.transpose(W),X1.reshape(-1,1))
    return np.exp(y_hat[0])

def main():
    st.text('Author : Ananda Hange')
    st.title("FLUX Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit FLUX Predictior ANN App  </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    X1 = st.number_input("X1 (30-150) :")
    X2 = st.number_input("X2 (1-5) :")
    X3 = st.number_input("X3 (1-5) :")
    X4 = st.number_input("X4 (1-100) :")
    X5 = st.number_input("X5 (1-10) :")
    X6 = st.number_input("X6 (0-600) :")

    X = np.array([X1,X2,X3,X4,X5,X6])
    ann = ["Perceptron", "Multilayered Perceptron (Single Hidden Layer)"]
    opt = st.radio('Select Type Of ANN : ',ann)

    result=""

    if opt == 'Perceptron':
      if st.button("Predict"):
        result = perceptron_predict_target(X,W3)
        #print(result)
      st.success('The Target is : {}'.format(result))

    if opt == 'Multilayered Perceptron (Single Hidden Layer)':
      if st.button("Predict"):
        result = predict_target(X,W1,W2)
      st.success('The Target is : {}'.format(result))
    



if __name__=='__main__':
    main()
