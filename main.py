import cv2
from mosse import *
from pioneer_sdk import *

pioneer_mini = None


def video_read_mini():
    global pioneer_mini

    if pioneer_mini is None:
        pioneer_mini = Pioneer

    camera_frame = pioneer_mini.get_raw_video_frame()
    if camera_frame is None:
        return

    camera_frame = cv2.imdecode(np.frombuffer(camera_frame, dtype=np.uint8), cv2.IMREAD_COLOR)

    return camera_frame

video = None

def video_read_file():
    global video

    if video is None:
        video = cv2.VideoCapture("save.avi")

    _, image = video.read()

    return image


video_read = video_read_file


frame = video_read()
assert frame is not None
roi = cv2.selectROI(frame)
img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
tracker = MOSSE(img_gray, roi, learning_rate=0.125, train_num=8, sigma=100)
tracker.pre_training()  # get initial correlation filter

# Tracking Loop
while(True):

    frame = video_read()

    if frame is None:
        continue

    frame_h, frame_w ,_=frame.shape
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    tracker.update_frame(frame_gray)
    print(frame_gray)
    print(type(frame_gray))
    new_roi,psr = tracker.get_new_roi()     # apply H and get new bounding box on object

    x,y,w,h = new_roi

    # limiting the coordinates in case the tracked roi is going outside the frame
    x = max(0, x)
    y = max(0, y)

    if x + w >= frame_w:
        x = frame_w-w
    if y + h >= frame_h:
        y = frame_h-h

    new_roi = (x, y, w, h)
    tracker.update_roi(new_roi)
    print(x, y)

    if psr > 8:
        tracker.update()

    cv2.rectangle(frame, (x, y),(x + w, y + h), (0,255,0), 2)
    cv2.imshow('frame',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27 :
        break

cv2.destroyAllWindows()
