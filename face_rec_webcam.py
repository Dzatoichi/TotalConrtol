import cv2
import os


def face_rec_webcam():
    # получаем путь к этому скрипту
    path = os.path.dirname(os.path.abspath(__file__))
    # создаём новый распознаватель лиц
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # добавляем в него модель, которую мы обучили на прошлых этапах
    recognizer.read(path + r'/trainer/trainer.yml')
    # указываем, что мы будем искать лица по примитивам Хаара
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # получаем доступ к камере
    cam = cv2.VideoCapture(0)
    # настраиваем шрифт для вывода подписей
    font = cv2.FONT_HERSHEY_SIMPLEX

    # запускаем цикл
    while True:
        # получаем видеопоток
        ret, im = cam.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Нажатие 'q' - выход
            break
        # переводим его в ч/б
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # определяем лица на видео
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        # перебираем все найденные лица
        for (x, y, w, h) in faces:
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Нажатие 'q' - выход
                break
            # получаем id пользователя
            nbr_predicted, coord = recognizer.predict(gray[y:y + h, x:x + w])
            # рисуем прямоугольник вокруг л≈ица
            cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
            # если мы знаем id пользователя
            # добавляем текст к рамке
            cv2.putText(im, str(nbr_predicted), (x, y + h), font, 1.1, (0, 255, 0))
            # выводим окно с изображением с камеры
            cv2.imshow('TotalControl', im)
            # делаем паузу
            cv2.waitKey(2)


if __name__ == "__main__":
    face_rec_webcam()
