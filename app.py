from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
le_edu = pickle.load(open("le_edu.pkl", "rb"))
le_house = pickle.load(open("le_house.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form.get('Age'))
        income = float(request.form.get('Income'))
        loan = float(request.form.get('Loan_Amount'))
        credit = float(request.form.get('Credit_Score'))
        emp = float(request.form.get('Employment_Years'))

        edu_input = request.form.get('Education_Level')
        house_input = request.form.get('Housing_Status')

        # Handle unknown categories safely
        if edu_input not in le_edu.classes_:
            return "Invalid Education Input"

        if house_input not in le_house.classes_:
            return "Invalid Housing Input"

        edu = le_edu.transform([edu_input])[0]
        house = le_house.transform([house_input])[0]

        data = np.array([[age, income, loan, credit, emp, edu, house]])

        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][prediction]

        if prediction == 0:
            result = f"✅ Loan Approved (Confidence: {round(prob*100,2)}%)"
        else:
            result = f"❌ Loan Rejected (Confidence: {round(prob*100,2)}%)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)