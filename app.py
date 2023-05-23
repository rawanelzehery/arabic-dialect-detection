import joblib
import streamlit as st
from preprocessing import TextPreprocessor

# Load preprocessor
text_preprocessor = TextPreprocessor()

# Load the scikit-learn model from disk using joblib
model = joblib.load('models/svm_model.joblib')

#Define predictions dictionary
counrty_dict = {"EG": "مصري", "LY": "ليبي", "LB": "لباني" , "SD":"سوداني" ,"MA":"مغربي"}

# Define the Streamlit app
def app():
    # Create a text input field for the user to enter a string
    input_string = st.text_input('أدخل نص عربي')
    
    # Create a button to make the prediction
    if st.button('Make prediction'):
        trans_data = text_preprocessor.transform([input_string])
        # Check whether user inputted a valid string
        if trans_data[0] == "" or trans_data[0] == " ":
            text = "من فضلك أدخل جملة عربية صحيحة"
        else:
            # Make prediction using the model
            prediction = model.predict(trans_data)
            for i in counrty_dict:
                if i == predict:
                    dialect = counrty_dict[i]

        # Display the prediction as text
        st.write(f'The model predicted: {dialect}')

# Start the Streamlit app
if __name__ == '__main__':
    app()
