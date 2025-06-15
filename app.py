from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from models import train_model, predict_new_data
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
# Load environment variables from .env
load_dotenv()

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


def send_alert_email(
    sender, password, recipient_email, high_risk_jobs, single_alert_reason=None
):
    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient_email

        # Case 1: Multiple jobs alert
        if high_risk_jobs:
            msg["Subject"] = (
                f"🚨 High Risk Job Alerts - {datetime.now().strftime('%Y-%m-%d')}"
            )
            html = f"""<h2>High Risk Job Postings Detected</h2>
                    <p>{len(high_risk_jobs)} job postings were flagged as high risk:</p>
                    <table border="1" cellpadding="5" cellspacing="0">
                        <tr>
                            <th>Job ID</th>
                            <th>Title</th>
                            <th>Risk Score</th>
                        </tr>"""
            for job in high_risk_jobs:
                # Clean each field of non-ASCII characters
                job_id = str(job["job_id"]).encode("ascii", "ignore").decode("ascii")
                title = str(job["title"]).encode("ascii", "ignore").decode("ascii")
                score = f"{job['fraud_probability']*100:.1f}%"

                html += f"""<tr>
                            <td>{job_id}</td>
                            <td>{title}</td>
                            <td>{score}</td>
                        </tr>"""
            html += "</table><p>Please review these listings immediately.</p>"

        # Case 2: Single job alert with reason
        elif single_alert_reason:
            msg["Subject"] = "🚨 Fraudulent Job Alert"
            # Clean the reason text
            clean_reason = (
                str(single_alert_reason).encode("ascii", "ignore").decode("ascii")
            )
            html = f"""
            <html>
                <body>
                    <h2>🚨 Fraudulent Job Alert</h2>
                    <p>Dear user,</p>
                    <p>We've flagged the following job posting as <strong>potentially fraudulent</strong> based on our AI analysis.</p>
                    <p><strong>Reason:</strong> {clean_reason}</p>
                    <p>Please exercise caution and do not share personal information or money.</p>
                    <br>
                    <p>Stay safe,<br>Fraud Detection Team</p>
                </body>
            </html>
            """
        else:
            return False, "No data provided to send alert."

        # Attach the HTML content with UTF-8 encoding
        msg.attach(MIMEText(html, "html", "utf-8"))

        # Send the email
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully."
    except Exception as e:
        return False, f"Error sending email: {str(e)}"


# Load or train model
if not os.path.exists("model.joblib"):
    print("Training model...")
    model = train_model()
else:
    model = joblib.load("model.joblib")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file and allowed_file(file.filename):
        try:
            # Save file temporarily
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Read file
            if file.filename.endswith(".xlsx"):
                new_data = pd.read_excel(filepath, engine="openpyxl")
            else:
                new_data = pd.read_csv(filepath)

            # Verify required columns exist
            required_columns = ["title", "company_profile", "description"]
            missing_columns = [
                col for col in required_columns if col not in new_data.columns
            ]

            if missing_columns:
                return render_template(
                    "error.html",
                    error=f"Missing required columns: {', '.join(missing_columns)}",
                )

            # Predict
            results = predict_new_data(model, new_data)

            high_risk_jobs = results[results["fraud_probability"] > 0.7].to_dict(
                "records"
            )
            print(f"Found {len(high_risk_jobs)} high risk jobs")  # Debug

            if high_risk_jobs:
                print("Attempting to send email...")  # Debug
                success, message = send_alert_email(
                    EMAIL_SENDER,
                    EMAIL_PASSWORD,
                    "afzal.shaquib379@gmail.com",
                    high_risk_jobs,
                )
                print(f"Email result: {success}, {message}")  # Debug
                if not success:
                    print(f"Failed to send email: {message}")

            # Rest of your code...
            plot_urls = generate_visualizations(results)
            top_suspicious = results[["job_id", "title", "fraud_probability"]].head(10)
            os.remove(filepath)

            if "fraud_probability" in results.columns:
                high_risk_count = len(results[results["fraud_probability"] > 0.7])
                medium_risk_count = len(
                    results[
                        (results["fraud_probability"] > 0.3)
                        & (results["fraud_probability"] <= 0.7)
                    ]
                )
                low_risk_count = len(results[results["fraud_probability"] <= 0.3])
            else:
                high_risk_count = medium_risk_count = low_risk_count = 0

            return render_template(
                "results.html",
                tables=[results.to_html(classes="data", index=False)],
                top_suspicious=top_suspicious.to_dict("records"),
                plot_urls=plot_urls,
                high_risk_count=high_risk_count,
                medium_risk_count=medium_risk_count,
                low_risk_count=low_risk_count,
            )
        except Exception as e:
            print(f"Error in upload_file: {str(e)}")  # Debug
            return render_template("error.html", error=str(e))

    return redirect(url_for("index"))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ["xlsx", "csv"]


def generate_visualizations(results):
    plots = {}

    # Histogram of fraud probabilities
    plt.figure(figsize=(10, 6))
    sns.histplot(results["fraud_probability"], bins=20, kde=True)
    plt.title("Distribution of Fraud Probabilities")
    plt.xlabel("Probability of Being Fraudulent")
    plt.ylabel("Count")
    plots["histogram"] = get_plot_url(plt)
    plt.close()

    # Pie chart of real vs fake
    plt.figure(figsize=(6, 6))
    results["fraud_prediction"].value_counts().plot.pie(
        autopct="%1.1f%%",
        labels=["Genuine", "Fraudulent"],
        colors=["#4CAF50", "#F44336"],
    )
    plt.title("Percentage of Fraudulent vs Genuine Jobs")
    plots["pie"] = get_plot_url(plt)
    plt.close()

    # Feature importance (if available)
    if hasattr(model.named_steps["classifier"], "feature_importances_"):
        try:
            feature_importance = get_feature_importance(model)
            plt.figure(figsize=(12, 8))
            sns.barplot(x="importance", y="feature", data=feature_importance.head(10))
            plt.title("Top 10 Most Important Features")
            plots["features"] = get_plot_url(plt)
            plt.close()
        except Exception as e:
            print("Could not generate feature importance:", str(e))

    return plots


def get_plot_url(plt):
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def get_feature_importance(model):
    # Get feature names from transformers
    feature_names = []
    for name, trans, cols in model.named_steps["preprocessor"].transformers_:
        if hasattr(trans, "get_feature_names_out"):
            feature_names.extend(trans.get_feature_names_out(cols))
        elif name == "binary":
            feature_names.extend(cols)

    # Get importance scores
    importances = model.named_steps["classifier"].feature_importances_

    return pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
