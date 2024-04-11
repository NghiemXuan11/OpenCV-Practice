import cv2
import os

# Định nghĩa hàm để chạy video
def play_video(video_path, logo_path):
    # Kiểm tra xem file logo có tồn tại không
    if not os.path.isfile(logo_path):
        print("File logo không tồn tại. Hãy kiểm tra lại đường dẫn.")
        return

    # Tạo một đối tượng VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Đọc logo từ file
    logo = cv2.imread(logo_path)
    # Chỉnh kích thước logo nếu cần
    logo = cv2.resize(logo, (500, 500))

    # Tạo cửa sổ video với thuộc tính mặc định
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    # Đặt thuộc tính của cửa sổ video thành toàn màn hình
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Định nghĩa font
    font = cv2.FONT_HERSHEY_SIMPLEX 

    while(cap.isOpened()):
        # Chụp từng khung hình
        ret, frame = cap.read()

        # Nếu khung hình được đọc chính xác thì ret là True
        if not ret:
            print("Không nhận được khung hình (kết thúc luồng?). Thoát ...")
            break

        # Thêm text vào khung hình
        # cv2.putText(frame,  
        #             'Nguyen Truong Xuan Nghiem',
        #             (50, 50),  
        #             font, 2,  
        #             (0, 255, 255),  
        #             2,  
        #             cv2.LINE_4)

        # Thêm logo vào khung hình
        frame[100:600, 0:500] = cv2.addWeighted(frame[100:600, 0:500], 0.7, logo, 0.3, 0)  # Điều chỉnh vị trí của logo để nó hiển thị bên dưới text

        # Hiển thị khung hình kết quả
        cv2.imshow('Video', frame)

        # Nhấn 'q' trên bàn phím để thoát
        if cv2.waitKey(1) == ord('q'):
            break

    # Khi hoàn tất, giải phóng đối tượng VideoCapture
    cap.release()

    # Đóng tất cả các khung hình
    cv2.destroyAllWindows()

# Gọi hàm để chạy video
play_video("D:\\CV\\Lab1\\bigvideo.mp4",'D:\CV\Lab1\meoden.jpg')
