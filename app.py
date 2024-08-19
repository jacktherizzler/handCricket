import random
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response, jsonify

model = load_model('hand_sign_model.h5')

app = Flask(__name__)

user_score = 0
computer_score = 0
user_batting = True
computer_batting = False
toss_won = None

def toss():
    global toss_won
    toss_result = random.choice(["user", "computer"])
    if toss_result == "user":
        toss_won = "user"
        print("You won the toss! Choose Batting or Bowling.")
    else:
        toss_won = "computer"
        print("Computer won the toss!")

    return toss_result

def predict_hand_sign(frame):
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    return np.argmax(prediction[0]) + 1

def play_game(user_choice):
    global user_score, computer_score, user_batting, computer_batting

    computer_choice = random.randint(1, 6)
    print(f"User's choice: {user_choice}, Computer's choice: {computer_choice}")

    if user_batting:
        if user_choice == computer_choice:
            print(f"You're out! Your final score: {user_score}")
            user_batting = False
            computer_batting = True
        else:
            user_score += user_choice

    elif computer_batting:
        if computer_choice == user_choice:
            print(f"Computer is out! Computer's final score: {computer_score}")
            computer_batting = False
        else:
            computer_score += computer_choice

def gen():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if user_batting or computer_batting:
            user_choice = predict_hand_sign(frame)
            play_game(user_choice)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toss', methods=['GET'])
def toss_decision():
    toss_winner = toss()
    return jsonify({"toss_winner": toss_winner})

if __name__ == "__main__":
    app.run(debug=True)
