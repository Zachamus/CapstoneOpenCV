import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys
from picamera2 import Picamera2
import gpiod

Width = 640
Height = 480
Screen_Center = np.array([Width/2 , Height/2])
cameraMatrix = np.load('cameraMatrix.npy')
dist = np.load('dist.npy')
chip=gpiod.Chip('gpiochip4')
line = chip.get_line(17)
line.request(consumer='foobar', type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])

LastState = "low"

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash






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
    

    #frame = cv2.resize(frame, (Width, Height), interpolation=cv2.INTER_CUBIC)
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    rvecs, tvecs, trash = my_estimatePoseSingleMarkers(corners, 1, cameraMatrix, dist)
    for rvec, tvec in zip(rvecs, tvecs):
        cv2.drawFrameAxes(frame, cameraMatrix, dist, rvec, tvec, length=0.3)
    
    
    if (len(tvecs) > 0 and (tvecs[0][0] < 3) and (tvecs[0][1] < 3) and (tvecs[0][2] < 10)):
        if (LastState == "low"):
            line.set_value(1)
            LastState = "high"
            print("Turning on GPIO")
    else:
        if(LastState == "high"):
            line.set_value(0)
            print("Turning off GPIO")
            LastState = "low"


    
    #vector_array = obtain_Vector(corners, ids)


    #detected_markers = aruco_display(corners, ids, rejected, frame)
   # shown_vectors = detected_markers(detected_markers, Screen_Center,  vector_array, (255, 0, 0), 5, 0, 1)
    cv2.imshow("Image", frame)
#    video_writer.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
line.release()
