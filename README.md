# handCricket
Recreated a game i used to play w my school friends.
## Logic
- **Toss Logic**: The game starts with a toss, determining whether the user or the computer will bat first.
- **Batting and Bowling**: During batting, the user (or computer) shows a hand sign, which is recognized by the CNN model. If the sign matches the bowler's choice, the batsman is out. Otherwise, the score is added.
- **Winning**: After both have batted, the one with the higher score wins.
## Running the Game
1) First, train the model with hand sign images using the data collection and training scripts provided.
2) Save the model as hand_sign_model.h5.
3) Run the Flask application using python app.py.
4) Navigate to http://127.0.0.1:5000/ in your web browser to play the game.



