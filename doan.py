import cv2
import mediapipe as mp
import numpy as np
import math
from sklearn.linear_model import LogisticRegression

anh1_path = r'C:\Users\Admin\Desktop\cs231\DoanCV\test.png'
anh2_path = r'C:\Users\Admin\Desktop\cs231\DoanCV\train.png'
anhnen_path = r'C:\Users\Admin\Desktop\cs231\DoanCV\dmeDuc3.png'
anh1 = cv2.imread(anh1_path, cv2.IMREAD_UNCHANGED)
anh2 = cv2.imread(anh2_path, cv2.IMREAD_UNCHANGED)
anhnen = cv2.imread(anhnen_path, cv2.IMREAD_UNCHANGED)
anh1 = cv2.resize(anh1, (640, 480))
anh2 = cv2.resize(anh2, (640, 480))
anhnen = cv2.resize(anhnen, (640,480))
x = np.concatenate((anh1.reshape(-1, 3), anh2.reshape(-1, 3)), axis=0)
y = np.concatenate((np.zeros(anh1.shape[0] * anh1.shape[1]), np.ones(anh2.shape[0] * anh2.shape[1])), axis=0)

class_weights = {0: 1, 1: 2} 
model_dt = LogisticRegression(class_weight=class_weights)
model_dt.fit(x, y)

mpose = mp.solutions.pose
pose = mpose.Pose()
mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1)
mpFace = mp.solutions.face_detection
face_detection = mpFace.FaceDetection()

video=cv2.VideoCapture(0)

blue_path=r'C:\Users\Admin\Desktop\cs231\DoanCV\blue.png'
blue=cv2.imread(blue_path, -1)
mask_path=r'C:\Users\Admin\Desktop\cs231\DoanCV\ironman.png'
mask=cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

def bg(frame):
    frame = cv2.resize(frame, (640, 480))
    x_test = frame.reshape(-1, 3)
    y_pred = model_dt.predict(x_test)
    
    output_image = np.where(y_pred.reshape(frame.shape[0], frame.shape[1], 1) == 0, anhnen, frame)
    
    return output_image


def position_data(lmlist):
    global wrist, index_mcp, index_tip, midle_mcp, pinky_tip
    wrist = (lmlist[0][0], lmlist[0][1])
    index_mcp = (lmlist[5][0], lmlist[5][1])
    index_tip = (lmlist[8][0], lmlist[8][1])
    midle_mcp = (lmlist[9][0], lmlist[9][1])
    pinky_tip = (lmlist[20][0], lmlist[20][1])

def calculateDistance(p1,p2):
    x1,y1, x2,y2 = p1[0],p2[1],p2[0],p1[1]
    length = ((x2-x1)**2 + (y2-y1)**2)**(1.0/2)
    return length

def overlay_image(image, overlay_image, face_location, rotation_angle):
    # Copy the original image to avoid modifying the original image
    image_copy = image.copy()

    # Get face bounding box coordinates
    x, y, width, height = face_location
    # Calculate the position and size of the overlay image
    x_start = max(0, x - int(width * 0.01))  # Start x-coordinate of overlay image
    y_start = max(0, y - int(height * 0.25))  # Start y-coordinate of overlay image
    x_end = min(image_copy.shape[1], x + width + int(width * 0.01))  # End x-coordinate of overlay image
    y_end = min(image_copy.shape[0], y + height + int(height * 0.25))  # End y-coordinate of overlay image

    overlay_width = x_end - x_start
    overlay_height = y_end - y_start

    # Resize the overlay image to fit the calculated size
    overlay_image = cv2.resize(overlay_image, (overlay_width, overlay_height))

    # Rotate the overlay image
    M = cv2.getRotationMatrix2D((overlay_image.shape[1] // 2, overlay_image.shape[0] // 2), rotation_angle, 1)
    overlay_image = cv2.warpAffine(overlay_image, M, (overlay_image.shape[1], overlay_image.shape[0]))

    # Extract the alpha channel from the overlay image
    overlay_alpha = overlay_image[:, :, 3] / 255.0

    # Convert the overlay and background images to float
    overlay_float = overlay_image[:, :, :3].astype(float)
    background_float = image_copy[y_start:y_end, x_start:x_end].astype(float)

    # Blend the images using alpha blending
    blended = (overlay_alpha[:, :, np.newaxis] * overlay_float) + ((1 - overlay_alpha[:, :, np.newaxis]) * background_float)

    # Convert the blended image back to uint8   
    blended = blended.astype(np.uint8)

    # Replace the corresponding region in the background image with the blended image
    image_copy[y_start:y_end, x_start:x_end] = blended

    return image_copy

def eye_check(results):
    landmark_5 = None
    landmark_2 = None

    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id == 5:
                landmark_5 = (lm.x, lm.y, lm.z, lm.visibility)
            elif id == 2:
                landmark_2 = (lm.x, lm.y, lm.z, lm.visibility)
    return landmark_5, landmark_2

def touch(result, hand):
    if result.pose_landmarks and hand:
        pose_landmarks = result.pose_landmarks.landmark
        landmark_9 = pose_landmarks[11]
        landmark_10 = pose_landmarks[12]
        cx = (landmark_9.x + landmark_10.x) / 2
        cy = (landmark_9.y + landmark_10.y) / 2
        touch_threshold = 0.1  # Adjust this threshold as needed

        if abs(hand[0] - cx) < touch_threshold and abs(hand[1] - cy) < touch_threshold+0.5:
            return 1

    return 0

def hand_check(results):
    landmark_hand = None

    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id == 15 or id==16:
                landmark_hand = (lm.x, lm.y, lm.z, lm.visibility)
                
    return landmark_hand


def overlay_transparent(background_img, img_to_overlay_t,x,y,overlay_size=None):
    bg_img = background_img.copy()

    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(),overlay_size)

    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))

    mask = cv2.medianBlur(a,5)

    h,w, _ =overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]

    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))

    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

    return bg_img

lm_list=[]
key=0
previous_key=0

while True:
    ret,frame=video.read()
    frame = cv2.resize(frame, (640, 480))
    img=cv2.flip(frame,1)
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result=hands.process(imgRGB)
    pose_results=pose.process(imgRGB)
    # print(result.multi_hand_landmarks)
    lmlist=[]
    face_results = face_detection.process(imgRGB)
    faces = face_results.detections
    if pose_results.pose_landmarks and faces:
        for face in faces:
            bbox = face.location_data.relative_bounding_box
            face_location = (
                int(bbox.xmin * frame.shape[1]),
                int(bbox.ymin * frame.shape[0]) -20,
                int(bbox.width * frame.shape[1]),
                int(bbox.height * frame.shape[0])
            )
            x,y,width, height=face_location
            #cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
            landmark_5, landmark_2 = eye_check(pose_results)
            left_hand=hand_check(pose_results)
            key=touch(pose_results, left_hand)
            if key == 1 and (landmark_5 is None or landmark_2 is None):
                key = 0
            if previous_key == 1 and key == 0:
                key = previous_key
    
            previous_key = key
            
            if landmark_5 is not None and landmark_2 is not None:
                lm_list.append((landmark_5, landmark_2))
                # Calculate distance between pose_landmark ID 5 and pose_landmark ID 1
                if len(lm_list) >= 2 and previous_key==1:
                    
                    img=bg(img)
                    prev_landmark_5, prev_landmark_2 = lm_list[-2]
                    current_landmark_5, current_landmark_2 = lm_list[-1]
                    distance_5_1 = calculateDistance(prev_landmark_5, current_landmark_2)

                    # Calculate rotation angle based on the distance
                    rotation_angle = math.degrees(math.atan2(current_landmark_2[1] - current_landmark_5[1],
                                                                current_landmark_2[0] - current_landmark_5[0]))
                    
                    img = overlay_image(img, mask, face_location, -rotation_angle)
                    
    if result.multi_hand_landmarks and key==1:
        for handslms in result.multi_hand_landmarks:
            for id, lm in enumerate(handslms.landmark):
                h,w,c = img.shape
                coorx, coory = int(lm.x*w), int(lm.y*h)
                lmlist.append([coorx,coory])
                # cv2.circle(img, (coorx, coory), 6, (0,255,0), -1)
                # mpDraw.draw_landmarks(img, handslms, mpHands.HAND_CONNECTIONS)
        position_data(lmlist)
        palm=calculateDistance(wrist, index_mcp)
        distance = calculateDistance(index_tip, pinky_tip)
        ratio = distance/palm
        #print(ratio)
        if ratio>1.2:
            # print("hand close")
            centerX=midle_mcp[0]
            centerY=midle_mcp[1]
            shield_size=1.7
            diameter=round(palm*shield_size)
            x1=round(centerX-(diameter/2))
            y1=round(centerY-(diameter/2))
            h,w,c = img.shape
            if x1<0:
                x1=0
            elif x1>w:
                x1=w
            if y1<0:
                y1=0
            elif y1>h:
                y1=h
            if x1+diameter >w:
                diameter = w -x1
            if y1+diameter > h:
                diameter = h-y1
            shield_size = diameter,diameter
            if (diameter != 0):
                img = overlay_transparent(img, blue, x1, y1, shield_size)

    # print(img.shape)
    cv2.imshow("Frame",img)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()