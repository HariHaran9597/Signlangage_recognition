from flask import Flask, render_template, Response, jsonify
import time
import cv2
import mediapipe as mp
import math
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from googletrans import Translator
from gtts import gTTS

# Global variable to store predicted text
predicted_text = "Waiting for prediction..."

# Class hands
class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info


dic = ['drink',
       'food',
       'full',
       'have',
       'hello',
       'i',
       'i love you',
       'police',
       'prefer',
       'shirt',
       'telephone',
       'water',
       'wrong',
       'yes',
       'you']

offset = 10
imgSize = 350
frame_rate = 5
counter = 0
result = 0
detector = HandDetector(maxHands=1)
model = load_model("VGG16_Augmented1.h5")


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.predicted_text = "Waiting for prediction..."  # Initialize predicted_text attribute

    def __del__(self):
        self.video.release()

    def get_frame_original(self):
        ret, img = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def get_frame(self, time_elapsed):
        ret, img = self.video.read()
        imgOutput = img.copy()

        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            imgWhiteCopy = imgWhite.copy()
            imgWhite = img_to_array(imgWhite)
            imgWhite = imgWhite.reshape((1, imgWhite.shape[0], imgWhite.shape[1], imgWhite.shape[2]))
            imgWhite = preprocess_input(imgWhite)

            prev = time.time()
            result = model.predict(imgWhite)
            word = dic[result.argmax()]
            result = result * 100000000
            maxVal = (result[0].max() / sum(result[0])) * 100

            if maxVal > 99.9999999:
                self.predicted_text = dic[result.argmax()]  # Update predicted_text attribute
                cv2.putText(imgOutput, dic[result.argmax()], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            else:
                self.predicted_text = ""  # Clear predicted_text if prediction is not confident
                cv2.putText(imgOutput, "", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        ret, jpeg = cv2.imencode('.jpg', imgOutput)
        return jpeg.tobytes(), self.predicted_text


app = Flask(__name__)
prev = 0
detector = HandDetector(maxHands=1)
model = load_model("VGG16_Augmented1.h5")

@app.route('/')
def index():
    return render_template('index.html')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to translate text to Tamil
def translate_text(text, target="ta"):
    translator = Translator()
    translation = translator.translate(text, dest=target)
    return translation.text

# Modify the gen function to include translation
def gen(camera):
    global prev
    global predicted_text, translated_text  # Declare these as global
    while True:
        time_elapsed = time.time() - prev
        frame, predicted_text = camera.get_frame(time_elapsed)
        translated_text = translate_text(predicted_text)
        yield (b'--frame\r\n'
               b'Content-Type:image/jpeg\r\n\r\n' + frame
               + b'\r\n\r\n')
        # Update the global variables with the latest predicted and translated text
        predicted_text = predicted_text
        translated_text = translated_text

@app.route('/get_text', methods=['GET'])
def get_text():
    return jsonify({
        'predicted_text': predicted_text,
        'translated_text': translate_text(predicted_text)})

if __name__== '__main__':
    app.run(debug=True)