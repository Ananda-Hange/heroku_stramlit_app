
import numpy as np
import streamlit as st

W1 = np.load('W1_momt.npy')
W2 = np.load('W2_momt.npy')
W3 =  np.load('W_momt.npy')



def sigmoid(z):
    '''In this function, we will compute the sigmoid(z)'''
    # we can use this function in forward and backward propagation

    return (1/(1+np.exp(-z)))

def predict_flux(X,W1,W2):
    X1= np.log(X)
    z11,z12,z13 = np.dot(np.transpose(W1),X1.reshape(-1,1))
    o11 = sigmoid(z11)
    o12 = sigmoid(z12)
    o13 = sigmoid(z13)
    y_hat = o11*W2[0] + o12*W2[1] + o13*W2[2]

    return np.exp(y_hat[0])


def perceptron_predict_flux(X,W):
    X1 = np.log(X)
    y_hat = np.dot(np.transpose(W),X1.reshape(-1,1))
    return np.exp(y_hat[0])

def main():
    st.title("FLUX Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit FLUX Prediction ANN  </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    FT = st.number_input("FT :")
    FFR = st.number_input("FFR :")
    CFR = st.number_input("CFR :")
    N = st.number_input("N :")
    W = st.number_input("W :")
    L = st.number_input("L :")

    X = np.array([FT,FFR,CFR,N,W,L])
    ann = ["Perceptron", "Multilayer Perceptron"]
    opt = st.radio('Select Type Of ANN : ',ann)

    result=""

    if opt == 'Perceptron':
      if st.button("Predict"):
        result = perceptron_predict_flux(X,W3)
        #print(result)
      st.success('The FLUX is : {}'.format(result))

    if opt == 'Multilayer Perceptron':
      if st.button("Predict"):
        result = predict_flux(X,W1,W2)
      st.success('The FLUX is : {}'.format(result))
    



if __name__=='__main__':
    main()
