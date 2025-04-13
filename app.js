import React from 'react';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <h1>FinRobot</h1>
      <ul>
        <li><a href="#dashboard">Dashboard</a></li>
        <li><a href="#fraud-detection">Fraud Detection</a></li>
        <li><a href="#sentiment-analysis">Sentiment Analysis</a></li>
      </ul>
    </nav>
  );
};

export default Navbar;

import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';

const Dashboard = () => {
  const [data, setData] = useState({});

  useEffect(() => {
    // Simulating an API call for financial data
    const fetchData = async () => {
      const response = await axios.get('https://api.example.com/financial-insights');
      const financialData = response.data;

      setData({
        labels: financialData.dates,
        datasets: [
          {
            label: 'Portfolio Value',
            data: financialData.values,
            borderColor: 'rgba(75,192,192,1)',
            fill: false,
          },
        ],
      });
    };
    fetchData();
  }, []);

  return (
    <div id="dashboard">
      <h2>Dashboard</h2>
      {data.labels ? (
        <Line data={data} />
      ) : (
        <p>Loading data...</p>
      )}
    </div>
  );
};

export default Dashboard;



import React from 'react';

const FraudDetection = () => {
  return (
    <div id="fraud-detection">
      <h2>Fraud Detection</h2>
      <p>Our AI-powered fraud detection system monitors transactions in real-time to prevent fraudulent activities.</p>
    </div>
  );
};

export default FraudDetection;



import React, { useState } from 'react';

const SentimentAnalysis = () => {
  const [sentiment, setSentiment] = useState(null);

  const analyzeSentiment = () => {
    // Simulated sentiment analysis API call
    const simulatedSentiment = {
      sentiment: 'Positive',
      confidence: '85%',
    };
    setSentiment(simulatedSentiment);
  };

  return (
    <div id="sentiment-analysis">
      <h2>Sentiment Analysis</h2>
      <button onClick={analyzeSentiment}>Analyze Market Sentiment</button>
      {sentiment && (
        <div>
          <p>Sentiment: {sentiment.sentiment}</p>
          <p>Confidence: {sentiment.confidence}</p>
        </div>
      )}
    </div>
  );
};

export default SentimentAnalysis;



import React from 'react';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import FraudDetection from './components/FraudDetection';
import SentimentAnalysis from './components/SentimentAnalysis';
import './styles.css';

function App() {
  return (
    <div className="App">
      <Navbar />
      <div className="content">
        <Dashboard />
        <FraudDetection />
        <SentimentAnalysis />
      </div>
    </div>
  );
}

export default App;
