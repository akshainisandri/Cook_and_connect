from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import os, json
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
from collections import Counter


# ================= INGREDIENT NORMALIZATION =================
INGREDIENT_MAP = {
    "bell pepper": "capsicum",
    "green chilli": "chilli",
    "red chilli": "chilli",
    "scallion": "onion",
    "spring onion": "onion",
    "coriander leaves": "coriander",
    "cilantro": "coriander",
    "eggplant": "brinjal"
}

def normalize_ingredients(ingredients):
    normalized = []
    for ing in ingredients:
        ing = ing.lower().strip()
        normalized.append(INGREDIENT_MAP.get(ing, ing))
    return list(set(normalized))

# ================= APP SETUP =================
app = Flask(__name__)
CORS(app)

app.secret_key = "cook_connect_secret_key"
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False
app.permanent_session_lifetime = 86400

# ================= DATABASE =================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ================= UPLOADS =================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ================= MODELS =================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    profile_pic = db.Column(db.String(200), default="default.png")
    diet = db.Column(db.String(30), default="Veg 🌱")
    cuisines = db.Column(db.String(200), default="Indian 🇮🇳")
class Memory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    name = db.Column(db.String(100))
    calories = db.Column(db.Integer)
    ingredients = db.Column(db.Text)
    image = db.Column(db.String(200))
    note = db.Column(db.Text)
    cooked_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ✅ ADD THIS LINE
    likes = db.relationship('Like', backref='memory', cascade="all, delete")


class Friend(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    status = db.Column(db.String(20), default="pending")

class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    memory_id = db.Column(db.Integer, db.ForeignKey('memory.id'))
# ================= LOAD AI =================
print("🚀 Loading YOLOv8...")
yolo_model = YOLO("yolov8n.pt")
print("✅ YOLO Loaded")

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)

# ================= LOAD RECIPES =================
with open("recipes.json", "r", encoding="utf-8") as f:
    RECIPE_DB = json.load(f)

# ================= RECIPE MATCHING =================
def find_best_recipe(user_ingredients):
    best_recipe = None
    best_score = 0
    user_set = set(user_ingredients)

    for recipe in RECIPE_DB:
        recipe_set = set(normalize_ingredients(recipe.get("ingredients", [])))
        score = len(recipe_set & user_set)
        if score > best_score:
            best_score = score
            best_recipe = recipe

    return best_recipe

def find_all_possible_recipes(user_ingredients):
    user_set = set(normalize_ingredients(user_ingredients))
    matches = []

    difficulty_rank = {
        "Easy": 0,
        "Medium": 1,
        "Hard": 2
    }

    for recipe in RECIPE_DB:
        recipe_set = set(normalize_ingredients(recipe.get("ingredients", [])))
        common = recipe_set & user_set
        missing = recipe_set - user_set

        if len(recipe_set) == 0:
            continue

        match_ratio = len(common) / len(recipe_set)

        if match_ratio >= 0.6 and len(missing) <= 1:
            matches.append({
                "name": recipe.get("name"),
                "preview": recipe.get("preview", ""),
                "difficulty": recipe.get("difficulty"),
                "time": recipe.get("time"),
                "calories": recipe.get("calories"),
                "missing": list(missing),
                "missing_count": len(missing),
                "difficulty_rank": difficulty_rank.get(recipe.get("difficulty"), 3)
            })

    matches.sort(key=lambda x: (
        x["missing_count"],
        x["difficulty_rank"],
        x["time"] or 999
    ))

    return matches


# ================= ROUTES =================
@app.route('/')
def splash():
    return render_template("splash.html")

@app.route('/login-page')
def login_page():
    return render_template("login.html")

@app.route('/signup-page')
def signup_page():
    return render_template("signup.html")

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template("home.html")



@app.route('/account')
def account():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    return redirect(url_for('view_profile', user_id=session["user_id"]))


@app.route('/recipe')
def recipe_page():
    return render_template("recipe.html")

@app.route('/cooking')
def cooking():
    return render_template("cooking.html")

@app.route('/memories-page')
def memories_page():
    return render_template("memories.html")

@app.route('/analytics-page')
def analytics_page():
    return render_template("analytics.html")
@app.route('/friends-page')
def friends_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template("friends.html")


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ================= AUTH =================
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email already exists"}), 400

    user = User(
        name=data['name'],
        email=data['email'],
        password=generate_password_hash(data['password'])
    )

    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Signup successful"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()

    if user and check_password_hash(user.password, data['password']):
        session.permanent = True
        session['user_id'] = user.id
        session['user_name'] = user.name
        return jsonify({"message": "Login success"})

    return jsonify({"error": "Invalid credentials"}), 401

# ================= YOLO DETECTION =================
@app.route('/detect-ingredients', methods=['POST'])
def detect_ingredients():
    if 'image' not in request.files:
        return jsonify({"ingredients": []})

    file = request.files['image']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    results = yolo_model(path, conf=0.01)
    detected = []

    for r in results:
        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls)
                label = r.names.get(cls_id, "unknown")
                detected.append(label)

    return jsonify({
        "ingredients": normalize_ingredients(detected)
    })

# ================= DISH OPTIONS =================
@app.route('/get-dish-options', methods=['POST'])
def get_dish_options():
    data = request.json
    ingredients = normalize_ingredients(data.get("ingredients", []))

    dishes = find_all_possible_recipes(ingredients)

    if not dishes:
        dishes = [{
            "name": "Quick Kitchen Stir Fry",
            "difficulty": "Easy",
            "time": 15,
            "calories": 250,
            "missing": [],
            "ai_generated": True
        }]

    return jsonify({
        "count": len(dishes),
        "dishes": dishes
    })

# ================= GENERATE RECIPE =================
@app.route('/generate-recipe', methods=['POST'])
def generate_recipe():
    data = request.json
    ingredients = normalize_ingredients(data.get("ingredients", []))
    selected_dish = data.get("selected_dish")

    if selected_dish:
        recipe = next(
            (r for r in RECIPE_DB
             if r.get("name", "").strip().lower() == selected_dish.strip().lower()),
            None
        )
    else:
        recipe = find_best_recipe(ingredients)

    if not recipe:
        recipe = {
            "name": "Custom Dish",
            "ingredients": ingredients,
            "calories": 220,
            "difficulty": "Easy",
            "steps": [{"text": "Cook everything well", "time": 10}]
        }

    timed_steps = []
    for step in recipe.get("steps", []):
        if isinstance(step, dict):
            timed_steps.append({
                "description": step.get("text"),
                "time": step.get("time", 5)
            })

    return jsonify({
        "name": recipe.get("name"),
        "ingredients": recipe.get("ingredients"),
        "calories": recipe.get("calories"),
        "difficulty": recipe.get("difficulty"),
        "steps": timed_steps,
        "total_time": sum(s["time"] for s in timed_steps)
    })

# ================= SAVE MEMORY =================
# ================= SAVE MEMORY =================
@app.route("/save-memory", methods=["POST"])
def save_memory():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    name = request.form.get("name")
    calories = request.form.get("calories")
    ingredients = request.form.get("ingredients")
    note = request.form.get("note")

    if ingredients:
        ingredients = json.dumps(json.loads(ingredients))

    image = request.files.get("image")

    if not image or image.filename == "":
        return jsonify({"error": "Image is required"}), 400

    os.makedirs("static/uploads", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{secure_filename(image.filename)}"
    filepath = os.path.join("static/uploads", filename)
    image.save(filepath)

    new_memory = Memory(
        user_id=session["user_id"],
        name=name,
        calories=int(calories),
        ingredients=ingredients,
        image=f"/static/uploads/{filename}",
        note=note,
        cooked_at=datetime.now()
    )

    db.session.add(new_memory)
    db.session.commit()

    return jsonify({"message": "Saved"})

@app.route("/get-memories")
def get_memories():

    if "user_id" not in session:
        return jsonify([])

    memories = Memory.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Memory.cooked_at.desc()).all()

    result = []

    for m in memories:
        result.append({
            "id": m.id,
            "name": m.name,
            "calories": m.calories,
            "ingredients": json.loads(m.ingredients) if m.ingredients else [],
            "image": m.image,
            "note": m.note,
            "cooked_at": m.cooked_at.strftime("%d %b %Y, %I:%M %p") if m.cooked_at else "",
            "likes": len(m.likes)   # 🔥 IMPORTANT
        })

    return jsonify(result)



@app.route("/delete-memory", methods=["POST"])
def delete_memory():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    memory_id = request.json.get("id")

    memory = Memory.query.filter_by(
        id=memory_id,
        user_id=session["user_id"]
    ).first()

    if memory:
        db.session.delete(memory)
        db.session.commit()

    return jsonify({"message": "Deleted"})


@app.route("/analytics-data")
def analytics_data():
    if "user_id" not in session:
        return jsonify({
            "totalCalories": 0,
            "totalRecipes": 0,
            "topIngredient": "-",
            "weeklyCalories": [0]*7,
            "ingredientLabels": [],
            "ingredientCounts": []
        })

    memories = Memory.query.filter_by(user_id=session["user_id"]).all()

    total_calories = 0
    ingredient_counter = Counter()
    weekly_calories = [0]*7

    for mem in memories:
        cal = mem.calories or 0
        total_calories += cal

        if mem.cooked_at:
            weekday = mem.cooked_at.weekday()
            weekly_calories[weekday] += cal

        ingredients = json.loads(mem.ingredients)
        for ing in ingredients:
            ingredient_counter[ing] += 1

    top_ingredient = ingredient_counter.most_common(1)
    top_ingredient = top_ingredient[0][0] if top_ingredient else "-"

    return jsonify({
        "totalCalories": total_calories,
        "totalRecipes": len(memories),
        "topIngredient": top_ingredient,
        "weeklyCalories": weekly_calories,
        "ingredientLabels": list(ingredient_counter.keys()),
        "ingredientCounts": list(ingredient_counter.values())
    })


@app.route("/search-users")
def search_users():
    if "user_id" not in session:
        return jsonify([])

    query = request.args.get("q")
    if not query:
        return jsonify([])

    users = User.query.filter(User.name.ilike(f"%{query}%")).all()

    return jsonify([
        {"id": u.id, "name": u.name, "email": u.email}
        for u in users if u.id != session["user_id"]
    ])

@app.route("/send-request", methods=["POST"])
def send_request():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    sender = session["user_id"]
    receiver = request.json.get("receiver_id")

    if sender == receiver:
        return jsonify({"error": "Cannot send request to yourself"}), 400

    existing = Friend.query.filter(
        ((Friend.sender_id == sender) & (Friend.receiver_id == receiver)) |
        ((Friend.sender_id == receiver) & (Friend.receiver_id == sender))
    ).first()

    if existing:
        return jsonify({"message": "Request already exists"})

    friend = Friend(sender_id=sender, receiver_id=receiver)
    db.session.add(friend)
    db.session.commit()

    return jsonify({"message": "Request Sent"})

@app.route("/accept-request", methods=["POST"])
def accept_request():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    request_id = request.json.get("request_id")
    friend = Friend.query.get(request_id)

    if friend and friend.receiver_id == session["user_id"]:
        friend.status = "accepted"
        db.session.commit()

    return jsonify({"message": "Friend Added"})

@app.route("/friend-requests")
def friend_requests():
    if "user_id" not in session:
        return jsonify([])

    user_id = session["user_id"]

    requests = Friend.query.filter_by(
        receiver_id=user_id,
        status="pending"
    ).all()

    return jsonify([
        {
            "request_id": r.id,
            "name": User.query.get(r.sender_id).name
        }
        for r in requests
    ])



@app.route("/my-friends")
def my_friends():
    if "user_id" not in session:
        return jsonify([])

    user_id = session["user_id"]

    friends = Friend.query.filter(
        ((Friend.sender_id == user_id) |
         (Friend.receiver_id == user_id)) &
        (Friend.status == "accepted")
    ).all()

    friend_ids = [
        f.receiver_id if f.sender_id == user_id else f.sender_id
        for f in friends
    ]

    users = User.query.filter(User.id.in_(friend_ids)).all()

    return jsonify([
        {"id": u.id, "name": u.name}
        for u in users
    ])

@app.route("/profile/<int:user_id>")
def view_profile(user_id):

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    user = User.query.get_or_404(user_id)

    memories = Memory.query.filter_by(
        user_id=user_id
    ).order_by(Memory.cooked_at.desc()).all()

    friend_count = Friend.query.filter(
        ((Friend.sender_id == user_id) |
         (Friend.receiver_id == user_id)) &
        (Friend.status == "accepted")
    ).count()

    # If viewing own profile
    if user_id == session["user_id"]:
        return render_template(
            "account.html",
            user=user,
            memories=memories,
            viewing_self=True,
            friend_count=friend_count
        )

    # If viewing friend's profile
    return render_template(
        "friend_profile.html",
        friend=user,
        memories=memories,
        friend_count=friend_count
    )


@app.route("/toggle-like/<int:memory_id>", methods=["POST"])
def toggle_like(memory_id):

    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    existing = Like.query.filter_by(
        user_id=session["user_id"],
        memory_id=memory_id
    ).first()

    if existing:
        db.session.delete(existing)
    else:
        like = Like(
            user_id=session["user_id"],
            memory_id=memory_id
        )
        db.session.add(like)

    db.session.commit()
    return jsonify({"status": "ok"})
@app.route('/upload-profile-pic', methods=['POST'])
def upload_profile_pic():
    if "user_id" not in session:
        return "Unauthorized", 401

    file = request.files.get("image")
    if not file or file.filename == "":
        return "No file selected", 400

    filename = secure_filename(file.filename)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    user = User.query.get(session["user_id"])
    user.profile_pic = filename
    db.session.commit()

    return "Success", 200
# ================= RUN =================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(
        host="0.0.0.0",   # Allow access from other devices
        port=5000,
        debug=True
    )