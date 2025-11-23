# Credit Card Fraud Detection

An interactive Streamlit web application that simulates credit card transactions, performs data analysis, trains multiple ML models, compares results, and predicts fraud in real time. This project demonstrates a complete end-to-end fraud detection pipeline — from data generation to prediction.

---

## Features

### Synthetic Data Generation
- Generates realistic fraud and normal credit card transactions
- 20+ features including amount, time, velocity, ratio, distance, and more
- Adjustable parameters:
  - Number of transactions
  - Fraud rate percentage
  - Train/test split ratio

### Interactive Data Exploration
- Fraud vs Normal distribution visualization
- Amount distribution analysis (box plots)
- Time-based transaction patterns
- Distance vs Amount scatter plots
- Correlation heatmap

Perfect for understanding hidden fraud patterns and data relationships.

### Machine Learning Models

The app trains and compares 4 powerful models:

| Model | Included |
|-------|----------|
| Logistic Regression | Yes |
| Random Forest | Yes |
| Gradient Boosting | Yes |
| Support Vector Machine | Yes |

Each model provides:
- Accuracy Score
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve Visualization

### Real-Time Fraud Prediction
Enter custom transaction values and instantly see:
- Fraud / Normal prediction
- Probability confidence percentage
- Side-by-side comparison across all models

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core programming language |
| Streamlit | Interactive web UI framework |
| NumPy & Pandas | Data generation and manipulation |
| Plotly | Interactive visualizations |
| Scikit-Learn | Machine learning models |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/RitikaBhati-55/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

2. Install required libraries:
```bash
pip install -r requirements.txt
```

---

## Run the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

Your app will automatically open at:
```
http://localhost:8501
```

---

## Project Structure

```
Credit-Card-Fraud-Detection
├── Credit_card_fraud.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── LICENSE               # MIT License (optional)
```

---

## How It Works

### Pipeline Flow

```
1. Generate Synthetic Dataset
         ⬇
2. Visualize & Explore Data
         ⬇
3. Train Multiple ML Models
         ⬇
4. Compare Model Performance
         ⬇
5. Predict New Transactions
         ⬇
6. Display Results + Confidence
```

### Key Steps:
1. Data Generation - Creates balanced synthetic transaction data
2. EDA - Visualizes patterns and correlations
3. Model Training - Trains 4 different classifiers
4. Evaluation - Compares metrics across models
5. Prediction - Real-time fraud detection on new data

---

## Use Cases

This project is perfect for:
- Machine learning beginners and students
- Portfolio and resume projects
- College assignments and capstone projects
- Fraud detection demonstrations
- Streamlit UI practice and learning

---

## Deployment Options

Deploy this app easily on popular platforms:

- Streamlit Community Cloud - Free and easiest
- Render - Free tier available
- Railway - Simple deployment
- Heroku - Classic PaaS option

### Quick Deploy to Streamlit Cloud:
1. Push your code to GitHub
2. Go to share.streamlit.io
3. Connect your repository
4. Click Deploy!

---

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

Ritika Bhati

- GitHub: @RitikaBhati-55
- Project Link: https://github.com/RitikaBhati-55/Credit-Card-Fraud-Detection

---

## Support

If this project helped you, please consider:

- Starring the repository
- Forking for your own use
- Sharing with others
- Providing feedback via issues

---

## Acknowledgments

- Inspired by real-world fraud detection systems
- Built with modern ML and web technologies
- Thanks to the open-source community

---

Made with love and Python
