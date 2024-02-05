import React, { useState } from 'react';
import axios from 'axios';
import './App.css'

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [error, setError] = useState('');

  const onFileChange = event => {
    setFile(event.target.files[0]);
  };

  const onSubmit = async (event) => {
    event.preventDefault();

    if (!file) {
      setError('Please select a file to predict.');
      return;
    }
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error during prediction', error);
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error("Error Response Data:", error.response.data);
        console.error("Error Response Status:", error.response.status);
        console.error("Error Response Headers:", error.response.headers);
      } else if (error.request) {
        // The request was made but no response was received
        console.error("Error Request:", error.request);
      } else {
        // Something happened in setting up the request that triggered an Error
        console.error('Error Message:', error.message);
      }
    }
  };

  return (
    <div className="App">
      <div className="container">
        <h1>Image Prediction</h1>
        <form onSubmit={onSubmit} className="upload-form">
          <input type="file" onChange={onFileChange} />
          <button type="submit">Predict</button>
        </form>
        {prediction && <h2>Prediction: {prediction}</h2>}
        {error && <p className="error">{error}</p>}
      </div>
  </div>
  );
}

export default App;
