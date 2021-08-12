import cv2

########## GIAO DIỆN LỰA CHỌN ĐẦU VÀO CỦA CHƯƠNG TRÌNH ##########

# Menu giao diện
print("Chào mừng đến với chương trình nhận diện khuân mặt")
print("Vui lòng lựa chọn kiểu đầu vào: ")
print("1. Chạy với video đầu vào.")
print("2. Chạy với camera trên laptop.")
print("3. Chạy với camera IP.")
print("4. Thoát chương trình.")

# Lấy lựa chọn của người dùng.
option = int(input("Nhập vào lựa chọn của bạn: "))

# Chạy với video.
if option == 1:
    path = input("Nhập vào đường dẫn của video: ")
    cap = cv2.VideoCapture(path)
# Chạy với camera local trên laptop.
elif option == 2:
    cap = cv2.VideoCapture(0)
# Chạy với IP Camera.
elif option == 3:
    ip = input("Nhập vào ip của camera ip: ")
    cap = cv2.VideoCapture("http://"+ ip + "/video")
# Thoát chương trình.
elif option == 4:
    print("Tạm biệt!")
    exit()
else:
    print("Lựa chọn của bạn không khả dụng!")
    exit()

########## BẮT ĐẦU CHƯƠNG TRÌNH NHẬN DIỆN KHUÂN MẶT ##########

# Khởi tạo mô hình nhận diện khuân mặt Haar.
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Set cấu hình video để lưu lại
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640,  480))

# Chạy liên tục cho đến khi hết dừng video/ tắt webcame hoặc ấn ESC
while True:

    # Nếu camera khả dụng -> Đọc frame ảnh từ video và tiến hành xử lý ảnh.
    try:
        # Đọc frame ảnh
        _, img = cap.read()

        # Chuyển ảnh RGB (ảnh màu) -> GRAY (ảnh xám).
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load mô hình haar, đẩy frame ảnh vào mô hình và tiến hành tìm kiếm khuân mặt có trong ảnh.
        # Output là tọa độ x_min, y_min, weight, height của các khuân mặt tìm được trong ảnh.
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minSize=(200, 200))

        # Nếu nhận diện được có khuân mặt trong frame ảnh sẽ tiến hành vẽ box tại vị trí tìm được
        if len(faces) > 0:
            # Sử dụng vòng lặp for để lấy toàn bộ vị trị của các khuân mặt được model nhận diện ra.
            for (x, y, w, h) in faces:
                # Sử dụng hàm rectangle của opencv để vẽ box bao quanh khuân mặt dựa vào x, y, w, h
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
                # Sử dụng hàm putText để thêm nội dung vào vị trí box vừa được phát hiện
                cv2.putText(img, 'Face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        #Lưu frame vào video
        frame = cv2.resize(img, (640, 480))
        out.write(frame)

        # Hiển thị kết quả cuối cùng.
        cv2.imshow("Face Detection", img)

        # Dừng xử lý khi ấn ESC
        key = cv2.waitKey(30) & 0xFF
        if key==27:
            print("Kết thúc chương trình. Tạm biệt!")
            break

    # Nếu camera không khả dụng -> Kết thúc chương trình
    except:
        print("LỖI: Camera hiện chưa được bật hoặc đã kết thúc!")
        break
cap.release()
