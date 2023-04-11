import cv2

stream = cv2.VideoCapture(0)

while True:
    _, img = stream.read()
    face_cascade = cv2.CascadeClassifier('Neuro/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stream.release()
        cv2.destroyAllWindows()  # Уничтожает все открытые окна
        break
