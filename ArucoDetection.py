import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys
from picamera2 import Picamera2

Width = 1920
Height = 1080
Screen_Center = np.array([Width/2 , Height/2])
cameraMatrix = np.load('cameraMatrix.npy')
dist = np.load('dist.npy')


"""
    picam_Init(Width, Height)
    Initialize the picam object
    
    Parameters:
        Width, Height: for screen resolution
        
    Returns:
        Picam object
"""
def picam_Init(Width, Height):
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (Width, Height)})
    picam2.configure(preview_config)
   # picam2.video_configuration.controls.FrameRate = 30 
    picam2.set_controls({"FrameRate": 24})
    picam2.start()
    return picam2




"""
    obtain_Vector(corners, ids)
    Calculate vectors from the center of the screen to the center of detected ArUco markers.
    
    Parameters:
        ids (numpy.ndarray): The ids of the detected ArUco markers.
        corners (list): The corners of the detected ArUco markers.
        
    Returns:
        List of Tuples: a list containing the vectors as tuples
"""

def obtain_Vector(corners, ids):
    vector_array = []
    if ids is not None:
        for corner in corners:
            marker_center = np.mean(corner[0], axis=0)
            vector = marker_center - Screen_Center
            vector_array.append((vector[0], vector[1]))
    return vector_array


"""
    obtain_Distance(corners)
    Calculate distance from the aruco marker to the camera

    Parameters:
        
        corners (list): The corners of the detected ArUco markers.

    Returns:
        List of Tuples: a list containing the {x,y,z} vectors
"""


def obtain_Distance(corners):
    distance_vectors = []
    for corner in corners:
        rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers((corner, 120, cameraMatrix, dist))
        distance_vectors.append(tvecs)
    return distance_vectors




ap = argparse.ArgumentParser()
ap.add_argument("-i", "--camera", required=True, help="Set to True if using webcam")
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())

if args["camera"].lower() == "true":
    time.sleep(2.0)
else:
    if args["video"] is None:
        print("[Error] Video file location is not provided")
        sys.exit(1)
    # For video file, use a different approach or handle as needed
    print("Loading video files is not supported with Picamera2")
    sys.exit(1)

if ARUCO_DICT.get(args["type"], None) is None:
    print(f"ArUCo tag type '{args['type']}' is not supported")
    sys.exit(0)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#output_video_filename = 'detection.avi'
#frame_width, frame_height = 1920,1080
#video_writer = cv2.VideoWriter(output_video_filename, fourcc, 60.0,  (frame_width, frame_height))
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
parameters = cv2.aruco.DetectorParameters()
parameters.useAruco3Detection = True
parameters.minMarkerPerimeterRate = 0.01
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
picam2 = picam_Init(Width, Height)


while True:
    frame = picam2.capture_array()

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    h, w, _ = frame.shape
    

    frame = cv2.resize(frame, (Width, Height), interpolation=cv2.INTER_CUBIC)
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.25, cameraMatrix, dist)
    print(tvecs)
    #vector_array = obtain_Vector(corners, ids)


    detected_markers = aruco_display(corners, ids, rejected, frame)
   # shown_vectors = detected_markers(detected_markers, Screen_Center,  vector_array, (255, 0, 0), 5, 0, 1)
    cv2.imshow("Image", detected_markers)
#    video_writer.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
