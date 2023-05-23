import joblib
import streamlit as st

# Load the scikit-learn model from disk using joblib
model = joblib.load('path/to/saved/model.joblib')

# Define the Streamlit app
def app():
    # Create a text input field for the user to enter a string
    input_string = st.text_input('Enter a string:')
    
    # Create a button to make the prediction
    if st.button('Make prediction'):
        # Make the prediction using the loaded model
        prediction = model.predict([input_string])[0]
        
        # Display the prediction as text
        st.write(f'The model predicted: {prediction}')

# Start the Streamlit app
if __name__ == '__main__':
    app()
