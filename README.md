# ProfitPulse: Supermarket Sales Prediction

ProfitPulse is a machine learning-powered sales forecasting tool designed to predict future supermarket trends and business performance using historical retail data. Developed as an end-to-end data science web application, it features a robust backend built on Python and Flask with a modern, responsive frontend.

## 🚀 Features
- **Accurate Sales Predictions:** Uses supervised machine learning models to forecast branch-wise sales and performance.
- **Multiple ML Models:** Capable of running predictions via Support Vector Machine (SVM), Random Forest, Ridge, Lasso, and K-Nearest Neighbors (KNN).
- **Data Preprocessing Pipeline:** Features automated data encoding and standard scaling for raw input features.
- **Interactive Web Interface:** Beautiful, responsive UI built with pure CSS and HTML to accept user input seamlessly.

## 🛠️ Technology Stack
- **Backend Analytics:** Python 3, Pandas, NumPy, Scikit-Learn
- **Web Framework:** Flask
- **Frontend:** HTML5, CSS3, JavaScript
- **Version Control:** Git & GitHub

## 📖 Dataset Attributes
The predictive engine processes multiple attributes from store operations:
- **Demographics & Location:** Branch, City, Customer Type, Gender
- **Product Details:** Product Line, Unit Price, Quantity, Tax
- **Financial Metrics:** COGS (Cost of Goods Sold), Gross Income
- **Customer Feedback:** Rating

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Samay-P07/Supermarket-Sales-Prediction.git
   cd Supermarket-Sales-Prediction
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows use: venv\Scripts\activate
   # On Mac/Linux use: source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```
   The application will be hosted locally at `http://127.0.0.1:5000`.

## 👤 Developed By
**Samay Patel**  
[LinkedIn](https://www.linkedin.com/in/samaypatel7) • [GitHub](https://github.com/Samay-P07)
