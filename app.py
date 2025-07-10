from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = request.form

        gender = 1 if data["gender"] == "Male" else 0
        married = 1 if data["married"] == "Yes" else 0
        dependents = {"0": 0, "1": 1, "2": 2, "3+": 3}[data["dependents"]]
        education = 1 if data["education"] == "Graduate" else 0
        self_employed = 1 if data["self_employed"] == "Yes" else 0

        applicant_income = float(data["applicant_income"])
        coapplicant_income = float(data["coapplicant_income"])
        loan_amount = float(data["loan_amount"])
        loan_term = float(data["loan_term"])

        credit_history = {"Good": 1.0, "Normal": 0.5, "Bad": 0.0}[data["credit_history"]]
        property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[data["property_area"]]

        # Construct feature array
        raw_features = np.array([[gender, married, dependents, education, self_employed,
                                  applicant_income, coapplicant_income, loan_amount,
                                  loan_term, credit_history, property_area]])

        # Scale numerical features (ApplicantIncome, CoapplicantIncome, LoanAmount)
        scaled_input = raw_features.astype(float)
        scaled_input[:, [5, 6, 7]] = scaler.transform(scaled_input[:, [5, 6, 7]])

        # Predict loan eligibility
        prediction = model.predict(scaled_input)[0]
        result = "✅ Eligible for Loan" if prediction == 1 else "❌ Not Eligible for Loan"

        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
