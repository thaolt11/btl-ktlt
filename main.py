import cv2
import numpy as np
from collections import deque

class BackgroundExtraction:
    def __init__(self, width, height, scale, maxlen=10):
        self.maxlen = maxlen
        self.scale = scale
        self.width = width // scale
        self.height = height // scale
        self.buffer = deque(maxlen=maxlen)
        self.background = None

    def calculate_background(self):
        self.background = np.zeros((self.height, self.width), dtype='float32')
        for item in self.buffer:
            self.background += item
        self.background /= len(self.buffer)

    def update_background(self, old_frame, new_frame):
        self.background -= old_frame / self.maxlen
        self.background += new_frame / self.maxlen

    def update_frame(self, frame):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(frame)
            self.calculate_background()
        else:
            old_frame = self.buffer.popleft()
            self.buffer.append(frame)
            self.update_background(old_frame, frame)

    def get_background(self):
        return self.background.astype('uint8')

    def apply(self, frame):
        down_scale = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(down_scale, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        self.update_frame(gray)
        abs_diff = cv2.absdiff(self.get_background(), gray)
        _, ad_mask = cv2.threshold(abs_diff, 15, 255, cv2.THRESH_BINARY)
        return cv2.resize(ad_mask, (self.width * self.scale, self.height * self.scale))

class PlayGame:
    def __init__(self, width, height, size=50):
        self.width = width
        self.height = height
        self.size = size
        self.logo = cv2.imread("logo.png")
        if self.logo is None:
            raise FileNotFoundError("Error: The image file logo.png was not found.")
        self.logo = cv2.resize(self.logo, (self.size, self.size))
        gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
        self.mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        self.x = np.random.randint(0, self.width - self.size)
        self.y = 0
        self.speed = 15
        self.score = 0

    def update_frame(self, frame):
        roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
        roi[np.where(self.mask)] = 0
        roi += self.logo

    def update_position(self, fg_mask):
        self.y += self.speed
        if self.y + self.size >= self.height:
            self.score -= 1
            self.y = 0
            self.speed = np.random.randint(10, 15)
            self.x = np.random.randint(0, self.width - self.size)
        roi = fg_mask[self.y:self.y + self.size, self.x:self.x + self.size]
        check = np.any(roi[np.where(self.mask)])
        if check:
            self.score += 1
            self.y = 0
            self.speed = np.random.randint(10, 20)
            self.x = np.random.randint(0, self.width - self.size)
        return check

width = 640
height = 480
scale = 2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

bg_buffer = BackgroundExtraction(width, height, scale, maxlen=5)
game = PlayGame(width, height)

while True:
    # Đọc fram từ webcam
    _, frame = cap.read() #_ Biến tạm thời để chứa trạng thái đọc( không sử dụng) và frame là khung hiện tại từ cam
    frame = cv2.resize(frame, (width, height)) # cố định size
    frame = cv2.flip(frame, 1) #lật khung hình để tạo cảm giác "gương" phù hợp với người chơi
    # xử lý frame đọc từ webcam để tạo mặt nạ chuyển động
    fg_mask = bg_buffer.apply(frame)
    #Xác định va chạm
    collision = game.update_position(fg_mask) # fg_mask ktra va chạm, collision là true or false
    game.update_frame(frame)
    #Nếu va chạm
    if collision:
        frame[::2] = 255 #Tạo hiệu ứng nhấp nháy trên khung hình bằng cách đặt giá trị pixel trên các hàng chẵn (frame[::2]) thành màu trắng (255).
    #Hiển thị điểm số
    text = f"Diem So: {game.score}"
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 3)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break
