import cv2

def show_frame(frame):
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked: ({x}, {y})")

    cv2.namedWindow("test",cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("test", mouse)
    cv2.resizeWindow("test",1920,1080)

    while True:
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# --- Load image correctly ---
image_path = r"D:\project\bachelor\modules\trach\frame1.png"
frame = cv2.imread(image_path)

if frame is None:
    raise ValueError(f"Cannot load image: {image_path}")

show_frame(frame)
