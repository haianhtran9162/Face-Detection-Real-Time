import cv2

#Import model nhận diện khuân mặt.
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

#Đọc ảnh.
img = cv2.imread('Data_Test/test_img.jpg')

#Chuyển ảnh RGB -> GRAY.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Load ảnh vào model và nhận được output là tọa độ x_min, y_min, weight, height của các khuân mặt trong ảnh.
faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)

#Sử dụng vòng lặp for để lấy toàn bộ vị trị của các khuân mặt được model nhận diện ra.
for (x, y, w, h) in faces:
    #Sử dụng hàm rectangle của opencv để vẽ box bao quanh khuân mặt dựa vào x, y, w, h
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    # Sử dụng hàm putText để thêm nội dung vào vị trí box vừa được phát hiện
    cv2.putText(img, 'Face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#Hiển thị kết quả cuối cùng.
cv2.imshow("Face Detection", img)
cv2.waitKey()