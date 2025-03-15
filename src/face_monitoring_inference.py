import cv2 as cv
import cv2, time
import numpy as np
import pandas as pd
import yaml, pymongo
import mediapipe as mp
import faiss, glob, os
from deepface import DeepFace
from datetime import datetime, timedelta

with open('secrets.yaml') as f:
    secrets = yaml.load(f, Loader=yaml.FullLoader)

os.environ["MONGO_DB_URI"] = secrets['MONGO_DB_URI']

try:
    client = pymongo.MongoClient(os.environ["MONGO_DB_URI"])
    db = client['Elearning']
    ffeatures_collection = db['ffeatures']
    print("Connected to MongoDB")
    
except Exception as e:
    print(e)

face_mesh = mp.solutions.face_mesh.FaceMesh(
                                            min_detection_confidence=0.5, 
                                            min_tracking_confidence=0.5
                                            )

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(
                                    color=(128,0,128),
                                    circle_radius=1,
                                    thickness=2
                                    )
p_face_mesh = mp.solutions.face_mesh

models = [
        "VGG-Face", 
        "Facenet", 
        "Facenet512", 
        "OpenFace", 
        "DeepFace", 
        "DeepID", 
        "ArcFace", 
        "Dlib", 
        "SFace",
        "GhostFaceNet",
        ]

def head_pose_inference(
                        image,
                        image_flag = False
                        ):
    start = time.time()

    if image_flag:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False

    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    img_h , img_w, img_c = image.shape
    face_2d = []
    face_3d = []

    texts = []
    face_centroids = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                    if idx ==1:
                        nose_2d = (lm.x * img_w,lm.y * img_h)
                        nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                    x,y = int(lm.x * img_w),int(lm.y * img_h)

                    face_2d.append([x,y])
                    face_3d.append(([x,y,lm.z]))

            face_2d = np.array(face_2d,dtype=np.float64)
            face_3d = np.array(face_3d,dtype=np.float64)

            face_centroid = np.mean(face_2d,axis=0)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length,0,img_h/2],
                                  [0,focal_length,img_w/2],
                                  [0,0,1]])
            distortion_matrix = np.zeros((4,1),dtype=np.float64)
            success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)

            rmat,jac = cv2.Rodrigues(rotation_vec)
            angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y < -10:
                text="Looking Left"
            elif y > 10:
                text="Looking Right"
            elif x < -10:
                text="Looking Down"
            elif x > 10:
                text="Looking Up"
            else:
                text="Forward"
            texts.append(text)
            face_centroids.append(face_centroid)
            nose_3d_projection,jacobian = cv2.projectPoints(nose_3d,rotation_vec,translation_vec,cam_matrix,distortion_matrix)

            p1 = (int(nose_2d[0]),int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] -x *10))

            cv2.line(image,p1,p2,(255,0,0),3)

            cv2.putText(image,text,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),3)
            cv2.putText(image,"x: " + str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(image,"y: "+ str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(image,"z: "+ str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        end = time.time()
        totalTime = end-start

        fps = 1/totalTime
        print("FPS: ",fps)

        cv2.putText(image,f'FPS: {int(fps)}',(20,450),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)

        mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                connection_drawing_spec=drawing_spec,
                                landmark_drawing_spec=drawing_spec
                                )
    return image, texts, face_centroids

def extract_face_information_for_db(img_path):
    face_objs = DeepFace.represent(
                                img_path = img_path,
                                model_name = models[2],
                                enforce_detection = False
                                )
    img_path = img_path.replace("\\", "/")
    user_name = img_path.split("/")[-2]

    if len(face_objs) != 1:
        if len(face_objs) == 0:
            Warning(f"No faces detected in the image : {img_path}")
        else:
            Warning(f"Multiple faces detected in the image : {img_path}")
        return None, None, None, None

    else:
        facial_area = face_objs[0]['facial_area']
        embeddings = face_objs[0]['embedding']
        face_confidence = face_objs[0]['face_confidence']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

    return embeddings, face_confidence, (x, y, w, h), user_name

def build_face_embedding_index(
                                d = 512,
                                face_index_path = 'models/face_index',
                                face_image_dir = 'data/facedb/*/*.jpg',
                                face_details_path = 'models/face_details.npz',
                                ):
    if (not os.path.exists(face_index_path)) or (not os.path.exists(face_details_path)):
        faiss_index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)

        embeddings = []
        user_names = []
        facial_areas = []
        face_confidences = []
        
        for idx, img_path in enumerate(glob.glob(face_image_dir)):
            emb, face_confidence, facial_area, user_name = extract_face_information_for_db(img_path)
            if emb is not None:
                embeddings.append(emb)
                user_names.append(user_name)
                facial_areas.append(facial_area)
                face_confidences.append(face_confidence)

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(glob.glob(face_image_dir))} images")

        embeddings = np.asarray(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, face_index_path)

        np.savez(
                face_details_path, 
                user_names=user_names, 
                facial_areas=facial_areas, 
                face_confidences=face_confidences
                )
        
    else:
        faiss_index = faiss.read_index(face_index_path)
        face_details = np.load(face_details_path)
        user_names = face_details['user_names']
        facial_areas = face_details['facial_areas']
        face_confidences = face_details['face_confidences']

    return faiss_index, user_names, facial_areas, face_confidences

def extract_face_information_for_inference(img_path):
    face_objs = DeepFace.represent(
                                img_path = img_path,
                                model_name = models[2],
                                enforce_detection = False
                                )
    img_path = img_path.replace("\\", "/")

    embeddings = []
    facial_areas = []
    face_confidences = []

    if len(face_objs) == 0:
        Warning(f"No faces detected in the image : {img_path}")
    else:
        for i in range(len(face_objs)):
            embs = face_objs[i]['embedding']
            facial_area = face_objs[i]['facial_area']
            face_confidence = face_objs[i]['face_confidence']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            embeddings.append(embs)
            facial_areas.append((x, y, w, h))   
            face_confidences.append(face_confidence)

    return embeddings, face_confidences, facial_areas

def search_face_in_db(
                    img_path, 
                    face_index_path = 'models/face_index',
                    face_details_path = 'models/face_details.npz',
                    ):
    index, user_names, _, _ = build_face_embedding_index(
                                                        face_index_path = face_index_path,
                                                        face_details_path = face_details_path,
                                                        )
    embeddings, face_confidences, facial_areas = extract_face_information_for_inference(img_path)

    retrieved_user_names = []
    retrieved_facial_areas = []
    retrieved_face_confidences = []

    if embeddings is not None:
        for idx, emb in enumerate(embeddings):
            if face_confidences[idx] >= 0.8:
                emb = np.array(emb).reshape(1, -1).astype('float32')
                faiss.normalize_L2(emb)
                D, I = index.search(emb, 5)
                I = np.array(I).squeeze()
                D = np.array(D).squeeze()
                user_name_list = [user_names[i] for i in I]
                user_name = max(set(user_name_list), key = user_name_list.count)
                avg_confidence = np.mean([d for i, d in zip(I, D) if user_names[i] == user_name])
                retrieved_face_confidences.append(np.round(avg_confidence, 3))
                retrieved_facial_areas.append(facial_areas[idx])
                retrieved_user_names.append(user_name)

    return retrieved_user_names, retrieved_facial_areas, retrieved_face_confidences

def eculedian_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def face_image_inference(
                        username,
                        face_image_path
                        ):
    img = cv2.imread(face_image_path)
    img_cp = img.copy()

    img_cp, texts, face_centroids = head_pose_inference(img_cp, image_flag = True)

    retrieved_user_names, retrieved_facial_areas, retrieved_face_confidences = search_face_in_db(face_image_path)
    for i in range(len(retrieved_user_names)):
        x, y, w, h = retrieved_facial_areas[i]
        face_centhroid_bbox = (x + w//2, y + h//2)
        face_area = w * h
        if (face_area >= 20000):
            timestamp = datetime.now()
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if (retrieved_user_names[i] == username) and (retrieved_face_confidences[i] >= 0.5):
                if (len(face_centroids) > 1) and (len(face_centroids) > 1):
                    distances = [eculedian_distance(face_centhroid_bbox[0], face_centhroid_bbox[1], x, y) for x, y in face_centroids]
                    head_pose_text = texts[np.argmin(distances)]
                else:
                    head_pose_text = texts[0]
                    
                ffeatures_collection.insert_one({
                                                "exp_username": username,
                                                "det_username": retrieved_user_names[i],
                                                "head_pose": head_pose_text,
                                                "face_confidence": float(retrieved_face_confidences[i]),
                                                "timestamp": timestamp
                                                })
                det_username = retrieved_user_names[i]

                cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 255, 0), 2)
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(img_cp, f'User: {retrieved_user_names[i]}', (x-30, y-40), font, 1, (0, 255, 0), 2)
            else:
                cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 0, 255), 2)
                ffeatures_collection.insert_one({
                                                "exp_username": username,
                                                "det_username": "N/A",
                                                "head_pose": "Unknown",
                                                "face_confidence": "N/A",
                                                "timestamp": timestamp
                                                })
                det_username = "N/A"
                
    return head_pose_text, det_username
    # cv.imshow('Face Monitoring Inference', img_cp)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

def video_face_inference(
                        username,
                        is_vis = False
                        ):
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img_cp = img.copy()
        img_cp_ = cv2.flip(img_cp, 1)
        cv.imwrite("data/temp_dir/temp.jpg", img_cp_)

        img_cp, texts, face_centroids = head_pose_inference(img_cp)

        retrieved_user_names, retrieved_facial_areas, retrieved_face_confidences = search_face_in_db("data/temp_dir/temp.jpg")
        for i in range(len(retrieved_user_names)):
            x, y, w, h = retrieved_facial_areas[i]
            face_centhroid_bbox = (x + w//2, y + h//2)
            face_area = w * h
            if (face_area >= 20000):
                timestamp = datetime.now()
                timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if (retrieved_user_names[i] == username) and (retrieved_face_confidences[i] >= 0.5):
                    if (len(face_centroids) > 1) and (len(face_centroids) > 1):
                        distances = [eculedian_distance(face_centhroid_bbox[0], face_centhroid_bbox[1], x, y) for x, y in face_centroids]
                        head_pose_text = texts[np.argmin(distances)]
                    elif len(face_centroids) == 1:
                        head_pose_text = texts[0]

                    else:
                        head_pose_text = "UnKnown"

                    ffeatures_collection.insert_one({
                                                    "exp_username": username,
                                                    "det_username": retrieved_user_names[i],
                                                    "head_pose": head_pose_text,
                                                    "face_confidence": float(retrieved_face_confidences[i]),
                                                    "timestamp": timestamp
                                                    })
                
                    cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(img_cp, f'User: {retrieved_user_names[i]}', (x-30, y-40), font, 1, (0, 255, 0), 2)
                else:
                    cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    ffeatures_collection.insert_one({
                                                    "exp_username": username,
                                                    "det_username": "N/A",
                                                    "head_pose": "Unknown",
                                                    "face_confidence": "N/A",
                                                    "timestamp": timestamp
                                                    })
                
        if is_vis:
            cv.imshow('Face Monitoring Inference', img_cp)
            if cv.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv.destroyAllWindows()

def face_analysis(
                username,
                x_min = 1440
                ):
    current_time = datetime.now()
    current_time_minus_x = current_time - timedelta(minutes=x_min)

    current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    current_time_minus_x = current_time_minus_x.strftime("%Y-%m-%d %H:%M:%S")

    data_user = ffeatures_collection.find({
                                        "exp_username": username,
                                        "timestamp": {
                                                    "$gte": current_time_minus_x,
                                                    "$lt": current_time
                                                    }
                                        })
    data_user = pd.DataFrame(list(data_user))
    
    if data_user.empty:
        return None, None
    
    else:
        data_user = data_user.drop(columns = ["_id", "exp_username", "face_confidence", "timestamp"])
        n_forward = len(data_user[data_user["head_pose"] == "Forward"])
        n_detected = len(data_user[data_user["det_username"] == username])
        n_total = len(data_user)

        detected_percentage = (n_detected/n_total)*100
        forward_percentage = (n_forward/n_total)*100

        detected_percentage = round(detected_percentage, 2)
        forward_percentage = round(forward_percentage, 2)


        detected_percentage = f"{detected_percentage} %"
        forward_percentage = f"{forward_percentage} %"

        return {
                "detected_percentage": detected_percentage,
                "forward_percentage": forward_percentage
                }
    
# # face_image_inference("Isuru Alagiyawanna", 'data/facedb/Isuru Alagiyawanna/IMG-20240804-WA0009.jpg')
# face_image_inference("Akshay Kumar", 'data/test_images/qqq.jpg')
# video_face_inference("Isuru Alagiyawanna", is_vis=True)