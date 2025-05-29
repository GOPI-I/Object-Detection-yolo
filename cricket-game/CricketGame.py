import cv2
import time
import os
import random
import numpy as np
from HandTrackingModule import handDetector as htm

# Initialize camera
wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Game constants
MAX_WICKETS = 1
INNINGS = 2  # 1st innings (player bats), 2nd innings (computer bats)

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
ORANGE = (0, 165, 255)
BLACK = (0, 0, 0)

# Animation parameters
animation_speed = 0.1
ball_animation = {"active": False, "start_time": 0, "x": 0, "y": 0, "target_x": 0, "target_y": 0}
bat_animation = {"active": False, "start_time": 0, "angle": 0}
celebration_animation = {"active": False, "start_time": 0, "particles": []}


# Game state
class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.players = ['Player 1', 'Player 2']
        self.p_count = 2
        self.overs = 1
        self.player_score = 0
        self.computer_score = 0
        self.player_wickets = 0
        self.computer_wickets = 0
        self.current_innings = 1
        self.game_over = False
        self.last_choice_time = time.time()
        self.player_runs = -1
        self.computer_runs = -1
        self.out = False
        self.message = ""
        self.message_time = 0
        self.target_score = 0
        self.innings_complete = False
        self.toss_complete = False
        self.batting_first = None
        self.striker = 0
        self.non_striker = 1
        self.current_bowler = 1
        self.p_runs = [0] * self.p_count
        self.p_balls = [0] * self.p_count
        self.p_out = [0] * self.p_count
        self.p_wickets = [0] * self.p_count
        self.p_overs = [0.0] * self.p_count
        self.p_rpo = [0] * self.p_count
        self.o_runs = 0
        self.balls_bowled = 0
        self.balls_played = 0
        self.toss_choice = None
        self.toss_result = None
        self.stable_fingers = 0
        self.last_finger_time = 0
        self.confirmed_choice = False
        self.finger_history = []
        self.finger_history_max = 5
        self.timeout_penalty = False
        self.innings_started = False
        self.innings_transition = False
        self.showing_out = False
        self.out_start_time = 0
        self.last_action_time = 0
        self.pitch_length = 400
        self.bat_width = 30
        self.bat_length = 120


game = GameState()


# Load assets
def load_assets():
    assets = {}

    # Finger images (1-5 only)
    folderPath = "FingerImages"
    myList = sorted(os.listdir(folderPath))[:5]  # Only use 1-5 images
    overLayList = []
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        if image is not None:
            overLayList.append(image)
        else:
            print(f"Warning: Could not load image {imPath}")
    assets["finger_images"] = overLayList

    # Load other images
    assets["stadium"] = cv2.imread("assets/stadium.jpg")
    if assets["stadium"] is not None:
        assets["stadium"] = cv2.resize(assets["stadium"], (wCam, hCam))

    assets["pitch"] = cv2.imread("assets/pitch.png")
    if assets["pitch"] is not None:
        assets["pitch"] = cv2.resize(assets["pitch"], (300, game.pitch_length))

    assets["bat"] = cv2.imread("assets/bat.png")
    if assets["bat"] is not None:
        assets["bat"] = cv2.resize(assets["bat"], (game.bat_width, game.bat_length))

    assets["ball"] = cv2.imread("assets/ball.png")
    if assets["ball"] is not None:
        assets["ball"] = cv2.resize(assets["ball"], (40, 40))

    return assets


assets = load_assets()

# Initialize detector
detector = htm(detectionCon=0.75, trackCon=0.75)
tipIds = [4, 8, 12, 16, 20]


def get_computer_choice():
    if game.current_innings == 2 and random.random() < 0.2 and game.player_score > 0:
        return min(max(1, game.player_score), 5)  # Ensure between 1-5
    return random.randint(1, 5)  # Only 1-5


def improved_finger_count(lmList):
    if len(lmList) == 0:
        return 0

    fingers = []

    # Thumb detection (handedness aware)
    if lmList[17][1] < lmList[0][1]:  # Right hand
        fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0)
    else:  # Left hand
        fingers.append(1 if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1] else 0)

    # Other fingers
    for id in range(1, 5):
        fingers.append(1 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 0)

    return sum(fingers)


def create_scoreboard():
    scoreboard = np.zeros((180, wCam, 3), dtype=np.uint8)
    cv2.rectangle(scoreboard, (0, 0), (wCam, 180), (50, 50, 50), -1)
    cv2.rectangle(scoreboard, (5, 5), (wCam - 5, 175), (100, 100, 100), 2)
    return scoreboard


def display_scoreboard(img):
    scoreboard = create_scoreboard()

    innings_text = f"{'1st' if game.current_innings == 1 else '2nd'} Innings"
    target_text = f"Target: {game.target_score}" if game.current_innings == 2 else ""

    cv2.putText(scoreboard, innings_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
    cv2.putText(scoreboard, target_text, (wCam // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # Player score
    cv2.putText(scoreboard, "YOU", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    cv2.putText(scoreboard, f"{game.player_score}/{game.player_wickets}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                GREEN, 2)
    overs_played = f"Overs: {game.balls_played // 6}.{game.balls_played % 6}"
    cv2.putText(scoreboard, overs_played, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

    # Computer score
    cv2.putText(scoreboard, "COMPUTER", (wCam - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    cv2.putText(scoreboard, f"{game.computer_score}/{game.computer_wickets}", (wCam - 200, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
    overs_bowled = f"Overs: {game.balls_bowled // 6}.{game.balls_bowled % 6}"
    cv2.putText(scoreboard, overs_bowled, (wCam - 200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

    # Add decorative elements
    cv2.line(scoreboard, (wCam // 2, 10), (wCam // 2, 170), (150, 150, 150), 1)
    cv2.circle(scoreboard, (wCam // 2, 90), 3, YELLOW, -1)

    img[0:180, 0:wCam] = cv2.addWeighted(img[0:180, 0:wCam], 0.3, scoreboard, 0.7, 0)


def display_toss_screen(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (wCam, hCam), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    if not game.toss_choice:
        cv2.putText(img, "TOSS TIME!", (wCam // 2 - 100, hCam // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, YELLOW, 3)
        cv2.putText(img, "Show 1 finger for HEADS or 2 for TAILS", (wCam // 2 - 250, hCam // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

        # Add coin animation
        coin_size = 100
        coin_x = wCam // 2 - coin_size // 2
        coin_y = hCam // 2 + 100
        cv2.circle(img, (wCam // 2, coin_y + coin_size // 2), coin_size // 2, YELLOW, -1)
        cv2.putText(img, "?", (wCam // 2 - 15, coin_y + coin_size // 2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 3)
    elif not game.toss_result:
        cv2.putText(img, f"You chose: {'HEADS' if game.toss_choice == 1 else 'TAILS'}",
                    (wCam // 2 - 150, hCam // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
        cv2.putText(img, "Waiting for toss result...", (wCam // 2 - 150, hCam // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    WHITE, 2)

        # Spinning coin animation
        coin_size = 100
        coin_x = wCam // 2 - coin_size // 2
        coin_y = hCam // 2 + 100
        spin_angle = int(time.time() * 20) % 360
        if spin_angle < 180:
            cv2.ellipse(img, (wCam // 2, coin_y + coin_size // 2),
                        (coin_size // 2, coin_size // 4), 0, 0, 180, YELLOW, -1)
            cv2.ellipse(img, (wCam // 2, coin_y + coin_size // 2),
                        (coin_size // 2, coin_size // 4), 0, 180, 360, ORANGE, -1)
        else:
            cv2.ellipse(img, (wCam // 2, coin_y + coin_size // 2),
                        (coin_size // 2, coin_size // 4), 0, 0, 180, ORANGE, -1)
            cv2.ellipse(img, (wCam // 2, coin_y + coin_size // 2),
                        (coin_size // 2, coin_size // 4), 0, 180, 360, YELLOW, -1)
    else:
        cv2.putText(img, f"Toss Result: {'HEADS' if game.toss_result == 1 else 'TAILS'}",
                    (wCam // 2 - 150, hCam // 2 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
        if game.toss_choice == game.toss_result:
            cv2.putText(img, "You won the toss!", (wCam // 2 - 100, hCam // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
            cv2.putText(img, "Show 1 to BAT or 2 to BOWL", (wCam // 2 - 150, hCam // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, WHITE, 2)
        else:
            cv2.putText(img, "PC won the toss!", (wCam // 2 - 100, hCam // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
            cv2.putText(img, "PC is choosing...", (wCam // 2 - 100, hCam // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        WHITE, 2)

        # Show final coin
        coin_size = 100
        coin_x = wCam // 2 - coin_size // 2
        coin_y = hCam // 2 + 100
        if game.toss_result == 1:
            cv2.circle(img, (wCam // 2, coin_y + coin_size // 2), coin_size // 2, YELLOW, -1)
            cv2.putText(img, "H", (wCam // 2 - 15, coin_y + coin_size // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 3)
        else:
            cv2.circle(img, (wCam // 2, coin_y + coin_size // 2), coin_size // 2, ORANGE, -1)
            cv2.putText(img, "T", (wCam // 2 - 15, coin_y + coin_size // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 3)

    return img


def display_turn_info(img):
    if game.current_innings == 1:
        turn_text = "YOUR BATTING - SHOW RUNS (1-5)" if game.batting_first == "PLAYER" else "YOUR BOWLING - SHOW BALL (1-5)"
    else:
        turn_text = "YOUR BOWLING - SHOW BALL (1-5)" if game.batting_first == "PLAYER" else "YOUR BATTING - SHOW RUNS (1-5)"

    turn_color = GREEN if "BATTING" in turn_text else RED

    turn_bg = np.zeros((60, wCam, 3), dtype=np.uint8)
    cv2.rectangle(turn_bg, (0, 0), (wCam, 60), (50, 50, 50), -1)
    cv2.rectangle(turn_bg, (5, 5), (wCam - 5, 55), (100, 100, 100), 2)

    cv2.putText(turn_bg, turn_text, (wCam // 2 - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, turn_color, 2)

    if len(game.finger_history) > 0:
        avg_fingers = round(sum(game.finger_history) / len(game.finger_history))
        cv2.putText(turn_bg, f"Detected: {avg_fingers}", (wCam - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE, 2)

    img[180:240, 0:wCam] = cv2.addWeighted(img[180:240, 0:wCam], 0.3, turn_bg, 0.7, 0)


def start_ball_animation():
    ball_animation["active"] = True
    ball_animation["start_time"] = time.time()
    ball_animation["x"] = wCam // 2 - 20
    ball_animation["y"] = 300
    ball_animation["target_x"] = wCam // 2 - 20
    ball_animation["target_y"] = 300 + game.pitch_length

    # Start bat animation slightly later
    bat_animation["active"] = True
    bat_animation["start_time"] = time.time() + 0.3
    bat_animation["angle"] = 0


def update_animations():
    current_time = time.time()

    # Ball animation
    if ball_animation["active"]:
        progress = min(1.0, (current_time - ball_animation["start_time"]) / 0.5)
        if progress >= 1.0:
            ball_animation["active"] = False
        else:
            ball_animation["x"] = ball_animation["x"]
            ball_animation["y"] = ball_animation["y"] + int(
                progress * (ball_animation["target_y"] - ball_animation["y"]))

    # Bat animation
    if bat_animation["active"]:
        progress = min(1.0, (current_time - bat_animation["start_time"]) / 0.3)
        if progress >= 1.0:
            bat_animation["active"] = False
        else:
            bat_animation["angle"] = int(45 * (1 - progress))

    # Celebration animation
    if celebration_animation["active"]:
        if current_time - celebration_animation["start_time"] > 3.0:
            celebration_animation["active"] = False
        else:
            # Add new particles
            if random.random() < 0.3:
                celebration_animation["particles"].append({
                    "x": random.randint(100, wCam - 100),
                    "y": random.randint(100, hCam - 100),
                    "color": random.choice([RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE]),
                    "size": random.randint(5, 15),
                    "life": random.uniform(1.0, 2.0),
                    "created": current_time
                })

            # Update existing particles
            celebration_animation["particles"] = [
                p for p in celebration_animation["particles"]
                if current_time - p["created"] < p["life"]
            ]


def draw_animations(img):
    # Draw pitch
    if assets["pitch"] is not None:
        pitch_x = wCam // 2 - assets["pitch"].shape[1] // 2
        pitch_y = 300
        img[pitch_y:pitch_y + game.pitch_length, pitch_x:pitch_x + assets["pitch"].shape[1]] = assets["pitch"]

    # Draw ball
    if ball_animation["active"] and assets["ball"] is not None:
        ball_x = ball_animation["x"]
        ball_y = ball_animation["y"]
        img[ball_y:ball_y + 40, ball_x:ball_x + 40] = assets["ball"]

    # Draw bat
    if bat_animation["active"] and assets["bat"] is not None:
        bat_center_x = wCam // 2
        bat_center_y = 300 + game.pitch_length - 50

        # Rotate bat
        angle = bat_animation["angle"]
        bat_img = assets["bat"].copy()
        if angle != 0:
            M = cv2.getRotationMatrix2D((game.bat_width // 2, game.bat_length // 2), angle, 1)
            bat_img = cv2.warpAffine(bat_img, M, (game.bat_width, game.bat_length))

        bat_x = bat_center_x - game.bat_width // 2
        bat_y = bat_center_y - game.bat_length // 2
        if bat_y >= 0 and bat_y + game.bat_length < hCam and bat_x >= 0 and bat_x + game.bat_width < wCam:
            img[bat_y:bat_y + game.bat_length, bat_x:bat_x + game.bat_width] = cv2.addWeighted(
                img[bat_y:bat_y + game.bat_length, bat_x:bat_x + game.bat_width], 0.5,
                bat_img, 0.5, 0)

    # Draw celebration particles
    if celebration_animation["active"]:
        for particle in celebration_animation["particles"]:
            cv2.circle(img, (particle["x"], particle["y"]), particle["size"], particle["color"], -1)


def display_action(img):
    if game.player_runs > 0 and game.confirmed_choice and not game.showing_out:
        run_img = assets["finger_images"][game.player_runs - 1]
        h, w, c = run_img.shape

        # Create a fancy frame for the run display
        frame = np.zeros((h + 40, w + 40, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (w + 40, h + 40), GREEN, 2)
        cv2.rectangle(frame, (5, 5), (w + 35, h + 35), (0, 100, 0), -1)
        frame[20:20 + h, 20:20 + w] = run_img

        img[240:240 + h + 40, 50:50 + w + 40] = frame

        action_type = "Runs" if (game.current_innings == 1 and game.batting_first == "PLAYER") or (
                game.current_innings == 2 and game.batting_first == "COMPUTER") else "Ball"
        cv2.putText(img, f"You {action_type}: {game.player_runs}", (w + 100, 240 + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    GREEN, 2)

    if game.computer_runs > 0 and not game.showing_out:
        comp_run = min(max(1, game.computer_runs), 5)
        comp_img = cv2.resize(assets["finger_images"][comp_run - 1], (100, 150))

        # Create a fancy frame for the computer's choice
        frame = np.zeros((170, 140, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (140, 170), RED, 2)
        cv2.rectangle(frame, (5, 5), (135, 165), (0, 0, 100), -1)
        frame[10:160, 20:120] = comp_img

        img[240:240 + 170, wCam - 160:wCam - 160 + 140] = frame

        action_type = "Ball" if (game.current_innings == 1 and game.batting_first == "PLAYER") or (
                game.current_innings == 2 and game.batting_first == "COMPUTER") else "Runs"
        cv2.putText(img, f"PC {action_type}: {game.computer_runs}", (wCam - 400, 240 + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

    if game.message and time.time() - game.message_time < 2:
        text_size = cv2.getTextSize(game.message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (wCam - text_size[0]) // 2
        text_y = (hCam + text_size[1]) // 2

        # Create message background
        bg_width = text_size[0] + 40
        bg_height = text_size[1] + 20
        bg = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
        cv2.rectangle(bg, (0, 0), (bg_width, bg_height), PURPLE, -1)
        cv2.rectangle(bg, (0, 0), (bg_width, bg_height), WHITE, 2)

        # Blend message background
        img[text_y - text_size[1] - 10:text_y + 10, text_x - 20:text_x + text_size[0] + 20] = cv2.addWeighted(
            img[text_y - text_size[1] - 10:text_y + 10, text_x - 20:text_x + text_size[0] + 20], 0.5,
            bg, 0.5, 0)

        cv2.putText(img, game.message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 3)


def process_turn():
    if game.timeout_penalty:
        game.message = "TIME OUT! 0 runs"
        game.message_time = time.time()
        game.timeout_penalty = False

        if (game.current_innings == 1 and game.batting_first == "PLAYER") or (
                game.current_innings == 2 and game.batting_first == "COMPUTER"):
            game.player_runs = 0
            game.computer_runs = get_computer_choice()
        else:
            game.player_runs = 0
            game.computer_runs = get_computer_choice()

    # Start animations for this turn
    start_ball_animation()
    game.last_action_time = time.time()

    if game.player_runs == game.computer_runs:
        game.out = True
        game.message = "OUT!"
        game.message_time = time.time()
        game.showing_out = True
        game.out_start_time = time.time()

        if (game.current_innings == 1 and game.batting_first == "PLAYER") or (
                game.current_innings == 2 and game.batting_first == "COMPUTER"):
            game.player_wickets += 1
            game.p_out[game.striker] = 2

            if game.player_wickets == MAX_WICKETS:
                end_innings()
            else:
                game.striker = 1 if game.striker == 0 else 0
                game.p_out[game.striker] = 1
        else:
            game.computer_wickets += 1
            game.p_wickets[game.current_bowler] += 1

            if game.computer_wickets == MAX_WICKETS:
                if game.current_innings == 1:
                    end_innings()
                else:
                    end_game()
    else:
        if (game.current_innings == 1 and game.batting_first == "PLAYER") or (
                game.current_innings == 2 and game.batting_first == "COMPUTER"):
            game.player_score += game.player_runs
            game.p_runs[game.striker] += game.player_runs
            game.p_balls[game.striker] += 1
            game.balls_played += 1

            if game.current_innings == 2 and game.player_score >= game.target_score:
                celebration_animation["active"] = True
                celebration_animation["start_time"] = time.time()
                celebration_animation["particles"] = []
                end_game()
                return

            if game.player_runs == 1 or game.player_runs == 3:
                game.striker, game.non_striker = game.non_striker, game.striker
        else:
            game.computer_score += game.computer_runs
            game.p_rpo[game.current_bowler] += game.computer_runs
            game.p_overs[game.current_bowler] += 0.1
            game.o_runs += game.computer_runs
            game.balls_bowled += 1

        game.message_time = time.time()

    if (game.current_innings == 1 and game.batting_first == "PLAYER") or (
            game.current_innings == 2 and game.batting_first == "COMPUTER"):
        if game.balls_played >= game.overs * 6 or game.player_wickets == MAX_WICKETS:
            end_innings()
    else:
        if game.current_innings == 2 and game.computer_score >= game.target_score:
            end_game()
        elif game.balls_bowled >= game.overs * 6 or game.computer_wickets == MAX_WICKETS:
            if game.current_innings == 1:
                end_innings()
            else:
                end_game()


def end_innings():
    if game.current_innings == 1:
        game.target_score = game.computer_score + 1 if game.batting_first == "COMPUTER" else game.player_score + 1
        game.current_innings = 2
        game.innings_complete = True
        game.message = f"Innings Over! Target: {game.target_score}"
        game.message_time = time.time()
        game.innings_transition = True

        game.computer_runs = -1
        game.player_runs = -1
        game.out = False
        game.striker = 0
        game.non_striker = 1
        game.balls_played = 0
        game.balls_bowled = 0
        game.o_runs = 0
        game.p_out = [0] * game.p_count
        game.p_runs = [0] * game.p_count
        game.p_balls = [0] * game.p_count
        game.p_overs = [0.0] * game.p_count
        game.p_rpo = [0] * game.p_count
        game.p_wickets = [0] * game.p_count

        if game.batting_first == "COMPUTER":
            game.p_out[game.striker] = 1
            game.p_out[game.non_striker] = 1
    else:
        end_game()


def end_game():
    game.game_over = True
    if game.current_innings == 2:
        if game.batting_first == "COMPUTER" and game.player_score >= game.target_score:
            game.message = "YOU WIN!"
            celebration_animation["active"] = True
            celebration_animation["start_time"] = time.time()
            celebration_animation["particles"] = []
        elif game.batting_first == "PLAYER" and game.computer_score >= game.target_score:
            game.message = "YOU LOSE!"
        else:
            if game.player_score > game.computer_score:
                game.message = "YOU WIN!"
                celebration_animation["active"] = True
                celebration_animation["start_time"] = time.time()
                celebration_animation["particles"] = []
            elif game.player_score < game.computer_score:
                game.message = "YOU LOSE!"
            else:
                game.message = "DRAW!"
    else:
        if game.player_score > game.computer_score:
            game.message = "YOU WIN!"
            celebration_animation["active"] = True
            celebration_animation["start_time"] = time.time()
            celebration_animation["particles"] = []
        elif game.player_score < game.computer_score:
            game.message = "YOU LOSE!"
        else:
            game.message = "DRAW!"
    game.message_time = time.time()


def process_toss(fingers):
    if not game.toss_choice and fingers in [1, 2]:
        game.toss_choice = fingers
        game.toss_result = random.choice([1, 2])
        game.last_choice_time = time.time()
    elif game.toss_choice and not game.toss_complete:
        if game.toss_choice == game.toss_result:
            if fingers in [1, 2]:
                game.batting_first = "PLAYER" if fingers == 1 else "COMPUTER"
                game.toss_complete = True
                game.current_innings = 1
                game.last_choice_time = time.time()
                game.p_out[game.striker] = 1
                game.p_out[game.non_striker] = 1
        else:
            game.batting_first = random.choice(["PLAYER", "COMPUTER"])
            game.toss_complete = True
            game.current_innings = 1
            game.last_choice_time = time.time()
            game.p_out[game.striker] = 1
            game.p_out[game.non_striker] = 1


def detect_fingers(img, lmList):
    fingers = improved_finger_count(lmList)

    if fingers > 0:
        game.finger_history.append(fingers)
        if len(game.finger_history) > game.finger_history_max:
            game.finger_history.pop(0)

        avg_fingers = round(sum(game.finger_history) / len(game.finger_history))

        if avg_fingers != game.stable_fingers:
            game.stable_fingers = avg_fingers
            game.last_finger_time = time.time()
            game.confirmed_choice = False
        elif time.time() - game.last_finger_time > 1.0 and 1 <= avg_fingers <= 5:
            if not game.confirmed_choice:
                game.confirmed_choice = True
                return avg_fingers
    else:
        game.finger_history = []
        game.stable_fingers = 0
        game.confirmed_choice = False

    return None


def display_result(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (wCam, hCam), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    text_size = cv2.getTextSize(game.message, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0]
    text_x = (wCam - text_size[0]) // 2
    text_y = (hCam - text_size[1]) // 2

    cv2.putText(img, game.message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, YELLOW, 5)

    score_text = f"Final Score: {game.player_score}-{game.player_wickets} vs {game.computer_score}-{game.computer_wickets}"
    score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    score_x = (wCam - score_size[0]) // 2
    cv2.putText(img, score_text, (score_x, text_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

    restart_text = "Show 5 fingers to Restart or 0 to Quit"
    restart_size = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    restart_x = (wCam - restart_size[0]) // 2
    cv2.putText(img, restart_text, (restart_x, text_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    # Draw celebration particles if active
    if celebration_animation["active"]:
        for particle in celebration_animation["particles"]:
            cv2.circle(img, (particle["x"], particle["y"]), particle["size"], particle["color"], -1)

    return img


# Main game loop
pTime = 0
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    # Apply stadium background if available
    if assets["stadium"] is not None:
        img = cv2.addWeighted(img, 0.7, assets["stadium"], 0.3, 0)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    # Update animations
    update_animations()

    # Handle showing out state
    if game.showing_out:
        if time.time() - game.out_start_time >= 3:  # 3 second delay for OUT
            game.showing_out = False
            game.out = False
            game.player_runs = -1
            game.computer_runs = -1
        else:
            # Skip processing inputs while showing OUT message
            display_scoreboard(img)
            display_turn_info(img)
            display_action(img)
            draw_animations(img)

            # Display OUT message in center of screen
            text_size = cv2.getTextSize("OUT!", cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
            text_x = (wCam - text_size[0]) // 2
            text_y = (hCam + text_size[1]) // 2

            # Create OUT message background
            bg_width = text_size[0] + 40
            bg_height = text_size[1] + 20
            bg = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
            cv2.rectangle(bg, (0, 0), (bg_width, bg_height), RED, -1)
            cv2.rectangle(bg, (0, 0), (bg_width, bg_height), WHITE, 2)

            # Blend OUT message background
            img[text_y - text_size[1] - 10:text_y + 10, text_x - 20:text_x + text_size[0] + 20] = cv2.addWeighted(
                img[text_y - text_size[1] - 10:text_y + 10, text_x - 20:text_x + text_size[0] + 20], 0.5,
                bg, 0.5, 0)

            cv2.putText(img, "OUT!", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, WHITE, 5)

            cv2.imshow("Professional Hand Cricket", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

    if not game.toss_complete:
        img = display_toss_screen(img)
        fingers = detect_fingers(img, lmList)
        if fingers is not None:
            process_toss(fingers)
    elif game.innings_transition:
        display_scoreboard(img)

        # Create transition screen
        transition_bg = np.zeros((300, 600, 3), dtype=np.uint8)
        cv2.rectangle(transition_bg, (0, 0), (600, 300), (50, 50, 50), -1)
        cv2.rectangle(transition_bg, (5, 5), (595, 295), (100, 100, 100), 2)

        # Add transition text
        cv2.putText(transition_bg, f"Target: {game.target_score}", (600 // 2 - 100, 300 // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, YELLOW, 3)
        cv2.putText(transition_bg, "Show any finger to continue", (600 // 2 - 200, 300 // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        # Blend transition screen
        img[hCam // 2 - 150:hCam // 2 + 150, wCam // 2 - 300:wCam // 2 + 300] = cv2.addWeighted(
            img[hCam // 2 - 150:hCam // 2 + 150, wCam // 2 - 300:wCam // 2 + 300], 0.5,
            transition_bg, 0.5, 0)

        fingers = detect_fingers(img, lmList)
        if fingers is not None:
            game.innings_transition = False
            game.innings_complete = False
            game.message = ""
    elif not game.game_over:
        display_scoreboard(img)
        display_turn_info(img)
        draw_animations(img)

        fingers = detect_fingers(img, lmList)
        if fingers is not None and not game.out and not game.innings_complete:
            if ((game.current_innings == 1 and game.batting_first == "PLAYER") or
                (game.current_innings == 2 and game.batting_first == "COMPUTER")) and 1 <= fingers <= 5:
                game.player_runs = fingers
                game.computer_runs = get_computer_choice()
                process_turn()
                game.last_choice_time = time.time()
            elif ((game.current_innings == 1 and game.batting_first == "COMPUTER") or
                  (game.current_innings == 2 and game.batting_first == "PLAYER")) and 1 <= fingers <= 5:
                game.player_runs = fingers
                game.computer_runs = get_computer_choice()
                process_turn()
                game.last_choice_time = time.time()

        display_action(img)

        cv2.putText(img, f'FPS: {int(fps)}', (wCam - 150, 190), cv2.FONT_HERSHEY_PLAIN, 2, BLUE, 2)
    else:
        img = display_result(img)
        fingers = detect_fingers(img, lmList)
        if fingers == 5:
            game = GameState()
        elif fingers == 0:
            break

    cv2.imshow("Professional Hand Cricket", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()