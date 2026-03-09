import joblib
import pandas as pd
import pymysql
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = "dev-secret-change-in-prod"

# ── MySQL config — change these to match your MySQL Workbench setup ────────────
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "____",
    "database": "health_analyzer",
    "cursorclass": pymysql.cursors.DictCursor,
}

def get_db():
    return pymysql.connect(**DB_CONFIG)


# ── Model loading ──────────────────────────────────────────────────────────────
try:
    model          = joblib.load("productivity_model.pkl")
    _le_gender     = joblib.load("gender_encoder.pkl")
    _le_occupation = joblib.load("occupation_encoder.pkl")
    _le_device     = joblib.load("device_encoder.pkl")
    feature_order  = joblib.load("feature_order.pkl")
except FileNotFoundError as e:
    model = _le_gender = _le_occupation = _le_device = feature_order = None
    print(f"Missing file: {e}. Run main.py first.")

VALID_GENDERS     = ["Female", "Male"]
VALID_OCCUPATIONS = ["Business", "Doctor", "Engineer", "Freelancer", "Professional", "Student", "Teacher"]
VALID_DEVICES     = ["Android", "iOS"]


def safe_transform(encoder, value):
    if value not in encoder.classes_:
        return 0
    return int(encoder.transform([value])[0])


def get_band(score):
    clamped  = max(1.0, min(10.0, score))
    band_int = int(round(clamped))
    if band_int <= 3:
        label, color, emoji = "Low",      "#ef4444", "⚠️"
    elif band_int <= 6:
        label, color, emoji = "Moderate", "#f59e0b", "📊"
    else:
        label, color, emoji = "High",     "#22c55e", "🚀"
    return {"score": round(clamped, 1), "score_int": band_int,
            "label": label, "color": color, "emoji": emoji}


def login_required(f):
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in to continue.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


# ── Auth routes ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        try:
            db  = get_db()
            cur = db.cursor()
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            db.close()
        except Exception as e:
            flash(f"Database error: {e}", "error")
            return render_template("login.html")

        if user and check_password_hash(user["password"], password):
            session.update({
                "user_id":    user["id"],
                "username":   user["username"],
                "email":      user["email"],
                "first_name": user["first_name"],
                "last_name":  user["last_name"],
                "history":    [],
            })
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username   = request.form.get("username", "").strip()
        first_name = request.form.get("first_name", "").strip()
        last_name  = request.form.get("last_name", "").strip()
        email      = request.form.get("email", "").strip()
        password   = request.form.get("password", "")
        confirm    = request.form.get("confirm_password", "")

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("register.html")
        if not all([username, email, password]):
            flash("All fields are required.", "error")
            return render_template("register.html")

        hashed = generate_password_hash(password)

        try:
            db  = get_db()
            cur = db.cursor()
            cur.execute(
                "INSERT INTO users (username, first_name, last_name, email, password) VALUES (%s, %s, %s, %s, %s)",
                (username, first_name, last_name, email, hashed)
            )
            print("✅ User inserted, ID:", cur.lastrowid)
            db.commit()
            user_id = cur.lastrowid
            db.close()
        except pymysql.err.IntegrityError:
            flash("Email or username already exists.", "error")
            return render_template("register.html")
        except Exception as e:
            flash(f"Database error: {e}", "error")
            return render_template("register.html")

        session.update({
            "user_id":    user_id,
            "username":   username,
            "email":      email,
            "first_name": first_name,
            "last_name":  last_name,
            "history":    [],
        })
        flash("Account created! Welcome.", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("login"))


# ── App routes ─────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", active_page="dashboard",
                           predictions_count=len(session.get("history", [])))


@app.route("/form")
@login_required
def form():
    return render_template("form.html", active_page="form",
                           form_data=session.get("last_form_data", {}),
                           valid_genders=VALID_GENDERS,
                           valid_occupations=VALID_OCCUPATIONS,
                           valid_devices=VALID_DEVICES)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if not model:
        flash("Model not loaded. Run main.py first.", "error")
        return redirect(url_for("form"))
    try:
        age        = max(18,  min(60,  float(request.form.get("age", 28))))
        gender     = request.form.get("gender", "Male")
        occupation = request.form.get("occupation", "Professional")
        device     = request.form.get("device_type", "Android")
        daily_ph   = max(1.0, min(12.0, float(request.form.get("daily_phone_hours", 5.0))))
        social_mh  = max(0.5, min(8.0,  float(request.form.get("social_media_hours", 2.0))))
        sleep_h    = max(4.0, min(9.0,  float(request.form.get("sleep_hours", 7.0))))
        stress     = max(1.0, min(10.0, float(request.form.get("stress_level", 5.0))))
        app_cnt    = max(5,   min(60,   float(request.form.get("app_usage_count", 20.0))))
        caffeine   = max(0,   min(6,    float(request.form.get("caffeine_intake_cups", 2.0))))
        weekend_sh = max(2.0, min(14.0, float(request.form.get("weekend_screen_time_hours", 6.0))))

        data = {
            "Age": age, "Gender": safe_transform(_le_gender, gender),
            "Occupation": safe_transform(_le_occupation, occupation),
            "Device_Type": safe_transform(_le_device, device),
            "Daily_Phone_Hours": daily_ph, "Social_Media_Hours": social_mh,
            "Sleep_Hours": sleep_h, "Stress_Level": stress,
            "App_Usage_Count": app_cnt, "Caffeine_Intake_Cups": caffeine,
            "Weekend_Screen_Time_Hours": weekend_sh,
        }
        X      = pd.DataFrame([data])[feature_order]
        result = get_band(float(model.predict(X)[0]))

        history = session.get("history", [])
        history.append({
            "daily_phone_hours": daily_ph,  "sleep_hours": sleep_h,
            "stress_level":      stress,    "caffeine_intake_cups": int(caffeine),
            "occupation":        occupation, "score": result["score"],
            "score_int":         result["score_int"],
            "label":             result["label"], "color": result["color"],
        })
        session["history"]        = history
        session["last_result"]    = result
        session["last_form_data"] = data
        return redirect(url_for("result"))

    except Exception as e:
        flash(f"Prediction error: {e}", "error")
        return redirect(url_for("form"))


@app.route("/result")
@login_required
def result():
    return render_template("result.html", active_page="result",
                           result=session.get("last_result"))


@app.route("/history")
@login_required
def history():
    return render_template("history.html", active_page="history",
                           history=list(reversed(session.get("history", []))))


@app.route("/history/clear")
@login_required
def clear_history():
    session["history"] = []
    flash("History cleared.", "success")
    return redirect(url_for("history"))


@app.route("/profile")
@login_required
def profile():
    history   = session.get("history", [])
    avg_score = round(sum(h["score"] for h in history) / len(history), 1) if history else None
    last_band = history[-1]["label"] if history else None
    return render_template("profile.html", active_page="profile",
                           predictions_count=len(history),
                           avg_score=avg_score, last_band=last_band)


@app.route("/profile/update", methods=["POST"])
@login_required
def update_profile():
    first_name = request.form.get("first_name", session.get("first_name"))
    last_name  = request.form.get("last_name",  session.get("last_name"))
    email      = request.form.get("email",      session.get("email"))

    try:
        db  = get_db()
        cur = db.cursor()
        cur.execute(
            "UPDATE users SET first_name=%s, last_name=%s, email=%s WHERE id=%s",
            (first_name, last_name, email, session["user_id"])
        )
        db.commit()
        db.close()
        session.update({"first_name": first_name, "last_name": last_name, "email": email})
        flash("Profile updated successfully.", "success")
    except Exception as e:
        flash(f"Update failed: {e}", "error")

    return redirect(url_for("profile"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)