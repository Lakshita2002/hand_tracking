import cv2
import mediapipe as mp
import time

# video object
cap = cv2.VideoCapture(0)

# create an object from the class hands
mpHands = mp.solutions.hands
# static_image_mode = False (kept as default)
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# to run our webcam
while True:   
    # gives us our frame
    success, image = cap.read()
    
    # send the RGB image of the hands obj
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # processes the image and gives us the result
    results = hands.process(imageRGB)
    # to check if we have multiple hands (to extract them one by one)
    # print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for hand_ld in results.multi_hand_landmarks:
            # to draw the 21 points
            # draws the hand landmark points
            # mpDraw.draw_landmarks(image, hand_ld)
            for id, ldm in enumerate(hand_ld.landmark):
                # print(id, ldm)
                # width, height, channels
                h, w, c = image.shape
                cx, cy = int(ldm.x*w), int(ldm.y*h)
                print(id, cx, cy)

                # detecting the 16th landmark (tip of index finger)
                if id==16:
                    cv2.circle(image, (cx, cy), 10, (217, 221, 107), cv2.FILLED)
            mpDraw.draw_landmarks(image, hand_ld, mpHands.HAND_CONNECTIONS)
            
    cTime = time.time()
    # frames per second
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (10, 78), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (141, 40, 40), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(1)
    