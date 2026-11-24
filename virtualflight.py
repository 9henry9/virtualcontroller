"""
Webcam gesture controller for a jet / flight game.

Head steers (W/A/S/D), hand gestures fire weapons (arrow keys).
Start: show thumb + pinky only on BOTH hands. Control turns on 3 s later.

Quit (any of these):
  - F12 from ANYWHERE (works even while the game has focus)
  - q or Esc in the camera window
  - the X button on the camera window
  - Ctrl+C in the terminal
  - slam the mouse into a screen corner
"""
import ctypes
import math
import time

import cv2
import pyautogui
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector

# pyautogui waits 0.1 s after every call by default, which makes the loop laggy.
pyautogui.PAUSE = 0
# Moving the mouse to a screen corner raises an exception -> keys get released.
pyautogui.FAILSAFE = True

# ---- Tuning (raise numbers = LESS sensitive) ----------------------------------
UP_THRESHOLD = 45        # px head must rise above the neutral line to climb   (was 20)
DOWN_THRESHOLD = 55      # px head must drop below the neutral line to dive    (was 35)
TURN_MARGIN = 20         # px a mouth corner must cross past the nose line     (was -10)
PINCH_THRESHOLD = 25     # px thumb tip to index joint (real distance) = pinch (was 30, y only)
BOOST_THRESHOLD = 35     # px between both index fingers to trigger booster    (was 50)
HOLD_FRAMES = 4          # frames a gesture must be held before it counts      (was 1)
SMOOTHING = 0.35         # 0..1, lower = smoother/slower head tracking         (was none)

ENABLE_DELAY = 3.0       # s after the start gesture before control turns on
MISSILE_COOLDOWN = 2.0   # s
FLARE_COOLDOWN = 2.0     # s
BOOST_COOLDOWN = 2.0     # s

WINDOW_NAME = "Camera Feed"
VK_F12 = 0x7B
QUIT_KEYS = (ord('q'), ord('Q'), 27)  # q, Q, Esc

# ---- Key handling: every held key is tracked so it can always be released ----
held_keys = set()


def hold(key):
    """Press and keep a key down. Returns True only when newly pressed."""
    if key in held_keys:
        return False
    pyautogui.keyDown(key)
    held_keys.add(key)
    return True


def release(key):
    """Release a key if we are holding it. Returns True only when released."""
    if key not in held_keys:
        return False
    pyautogui.keyUp(key)
    held_keys.discard(key)
    return True


def release_all():
    for key in list(held_keys):
        release(key)


def f12_pressed():
    """Global F12 check - works no matter which window has focus (Windows)."""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(VK_F12) & 0x8000)
    except AttributeError:  # not on Windows
        return False


def window_closed():
    """True once the user clicks the X on the camera window."""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


class Cooldown:
    """Time-based cooldown."""

    def __init__(self, seconds, ready_msg):
        self.seconds = seconds
        self.ready_msg = ready_msg
        self.ready_at = 0.0
        self.waiting = False

    def ready(self, now):
        if self.waiting and now >= self.ready_at:
            self.waiting = False
            print(self.ready_msg)
        return now >= self.ready_at

    def trigger(self, now):
        self.ready_at = now + self.seconds
        self.waiting = True


class Steady:
    """A condition only counts after it has been true for HOLD_FRAMES frames in a row."""

    def __init__(self, frames=HOLD_FRAMES):
        self.frames = frames
        self.count = 0

    def update(self, condition):
        self.count = self.count + 1 if condition else 0
        return self.count >= self.frames


class Smoother:
    """Exponential smoothing of a point to remove webcam jitter."""

    def __init__(self, alpha=SMOOTHING):
        self.alpha = alpha
        self.value = None

    def update(self, x, y):
        if self.value is None:
            self.value = (x, y)
        else:
            px, py = self.value
            self.value = (px + self.alpha * (x - px), py + self.alpha * (y - py))
        return int(self.value[0]), int(self.value[1])

    def reset(self):
        self.value = None


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return
    cap.set(3, 640)
    cap.set(4, 480)

    last_frame_time = time.time()
    face_detector = FaceMeshDetector(maxFaces=1)
    hand_detector = HandDetector(maxHands=2, detectionCon=0.75)

    missile_cd = Cooldown(MISSILE_COOLDOWN, "Missile reloaded")
    flare_cd = Cooldown(FLARE_COOLDOWN, "Flares reloaded")
    boost_cd = Cooldown(BOOST_COOLDOWN, "Boosters ready")

    ctrl_s, left_s, right_s = Smoother(), Smoother(), Smoother()

    steady_up, steady_down = Steady(), Steady()
    steady_right, steady_left = Steady(), Steady()
    steady_missile, steady_flare, steady_gun = Steady(), Steady(), Steady()
    steady_boost, steady_start = Steady(), Steady()

    neutral_y = None      # head height saved when the start gesture is made
    enable_at = None      # time when control switches on
    gesture_on = False
    window_shown = False

    print("Running. Press F12 (anywhere), or q / Esc / X on the camera window to quit.")

    try:
        while True:
            if f12_pressed():
                print("F12 pressed - quitting.")
                break

            success, img = cap.read()
            if not success or img is None:
                print("Camera frame not received - releasing keys.")
                release_all()
                if (cv2.waitKey(30) & 0xFF) in QUIT_KEYS:
                    break
                continue

            now = time.time()
            img = cv2.flip(img, 1)
            img, faces = face_detector.findFaceMesh(img, draw=False)

            face_ok = bool(faces)
            hands_ok = False

            if face_ok:
                face = faces[0]
                ctrl_x, ctrl_y = ctrl_s.update(face[168][0], face[168][1])
                # Mouth corners, pushed OUTWARD by TURN_MARGIN so you must turn further
                left_x, left_y = left_s.update(face[57][0] - TURN_MARGIN, face[57][1])
                right_x, right_y = right_s.update(face[287][0] + TURN_MARGIN, face[287][1])

                cv2.circle(img, (ctrl_x, ctrl_y), 5, (255, 255, 0), 2)
                cv2.circle(img, (left_x, left_y), 5, (255, 255, 0), 2)
                cv2.circle(img, (right_x, right_y), 5, (255, 255, 0), 2)
                cv2.line(img, (ctrl_x, 0), (ctrl_x, 1000), (255, 255, 255), 2)
                cv2.line(img, (0, ctrl_y), (1000, ctrl_y), (255, 255, 255), 2)

                hands, img = hand_detector.findHands(img, draw=True, flipType=False)
                if len(hands) == 2:
                    if hands[0]['type'] == "Left":
                        hand_l, hand_r = hands[0], hands[1]
                    else:
                        hand_l, hand_r = hands[1], hands[0]

                    lm_l, lm_r = hand_l['lmList'], hand_r['lmList']
                    r_idx_x, r_idx_y = lm_r[6][0], lm_r[6][1]
                    l_idx_x, l_idx_y = lm_l[6][0], lm_l[6][1]
                    r_thmb_x, r_thmb_y = lm_r[4][0], lm_r[4][1]
                    l_thmb_x, l_thmb_y = lm_l[4][0], lm_l[4][1]
                    hands_ok = True

                    for pt in ((r_idx_x, r_idx_y), (l_idx_x, l_idx_y),
                               (r_thmb_x, r_thmb_y), (l_thmb_x, l_thmb_y)):
                        cv2.circle(img, pt, 5, (0, 255, 255), 2)

                    # Start gesture: thumb + pinky only, on both hands
                    if enable_at is None:
                        l_up = hand_detector.fingersUp(hand_l)
                        r_up = hand_detector.fingersUp(hand_r)
                        start_pose = (l_up[0] == 1 and l_up[4] == 1 and sum(l_up) == 2 and
                                      r_up[0] == 1 and r_up[4] == 1 and sum(r_up) == 2)
                        if steady_start.update(start_pose):
                            neutral_y = ctrl_y
                            enable_at = now + ENABLE_DELAY
                            print("Gesture Enabled - control starts in 3 s")
            else:
                ctrl_s.reset()
                left_s.reset()
                right_s.reset()

            if not gesture_on and enable_at is not None and now >= enable_at:
                gesture_on = True
                print("Gesture Control On")

            if gesture_on:
                cv2.line(img, (0, neutral_y - UP_THRESHOLD), (1000, neutral_y - UP_THRESHOLD),
                         (0, 200, 0), 1)
                cv2.line(img, (0, neutral_y), (1000, neutral_y), (255, 0, 255), 2)
                cv2.line(img, (0, neutral_y + DOWN_THRESHOLD), (1000, neutral_y + DOWN_THRESHOLD),
                         (0, 0, 200), 1)

                if not face_ok:
                    # Face lost: don't keep flying on stale positions
                    release_all()
                else:
                    # ---- Up / down ----
                    offset = neutral_y - ctrl_y          # positive = head raised
                    if steady_up.update(offset > UP_THRESHOLD):
                        if hold('w'):
                            print("Going Up")
                    else:
                        release('w')

                    if steady_down.update(-offset > DOWN_THRESHOLD):
                        if hold('s'):
                            print("Going Down")
                    else:
                        release('s')

                    # ---- Left / right ----
                    turn_right = steady_right.update(right_x < ctrl_x and not left_x > ctrl_x)
                    turn_left = steady_left.update(left_x > ctrl_x and not right_x < ctrl_x)
                    if turn_right:
                        release('a')
                        if hold('d'):
                            print("Going Right")
                    elif turn_left:
                        release('d')
                        if hold('a'):
                            print("Going Left")
                    else:
                        release('a')
                        release('d')

                    # ---- Weapons (need both hands visible) ----
                    if hands_ok:
                        r_pinch = math.dist((r_thmb_x, r_thmb_y), (r_idx_x, r_idx_y)) < PINCH_THRESHOLD
                        l_pinch = math.dist((l_thmb_x, l_thmb_y), (l_idx_x, l_idx_y)) < PINCH_THRESHOLD

                        missile = steady_missile.update(r_pinch and l_pinch)
                        flare = steady_flare.update(l_pinch and not r_pinch)
                        gun = steady_gun.update(r_pinch and not l_pinch)

                        if missile:
                            if missile_cd.ready(now):
                                pyautogui.press('up')
                                missile_cd.trigger(now)
                                print("Missile Launched")
                        elif flare:
                            if flare_cd.ready(now):
                                pyautogui.press('down')
                                flare_cd.trigger(now)
                                print("Flares Deployed")
                        elif gun:
                            if hold('left'):
                                print("Gun Firing..........")

                        if not r_pinch and release('left'):
                            print("Firing stopped")

                        boost = steady_boost.update(abs(r_idx_x - l_idx_x) < BOOST_THRESHOLD)
                        if boost and boost_cd.ready(now):
                            pyautogui.press('right')
                            boost_cd.trigger(now)
                            print("Booster Engaged")
                    else:
                        # Hands lost: stop the gun instead of firing forever
                        steady_missile.update(False)
                        steady_flare.update(False)
                        steady_gun.update(False)
                        steady_boost.update(False)
                        if release('left'):
                            print("Firing stopped")

                # keep reload messages coming even when not firing
                missile_cd.ready(now)
                flare_cd.ready(now)
                boost_cd.ready(now)

            # FPS counter (cvzone 2.x no longer ships cvzone.FPS)
            frame_time = time.time()
            fps = 1.0 / max(frame_time - last_frame_time, 1e-6)
            last_frame_time = frame_time
            cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 2)
            cv2.putText(img, "F12 / q / Esc = quit", (20, 470), cv2.FONT_HERSHEY_PLAIN,
                        1.5, (0, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, img)
            window_shown = True

            if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS:
                break
            if window_shown and window_closed():
                print("Camera window closed - quitting.")
                break
    except KeyboardInterrupt:
        print("Ctrl+C - quitting.")
    finally:
        release_all()
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped. All keys released.")


if __name__ == "__main__":
    main()
