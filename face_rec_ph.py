import cv2
import os


def face_rec_ph(ph_path):
    # получаем путь к этому скрипту
    path = os.path.dirname(os.path.abspath(__file__))
    # создаём новый распознаватель лиц
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # добавляем в него модель, которую мы обучили на прошлых этапах
    recognizer.read(path + r'/trainer/trainer.yml')
    # указываем, что мы будем искать лица по примитивам Хаара
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # фото
    ph = cv2.imread(ph_path)
    # настраиваем шрифт для вывода подписей
    font = cv2.FONT_HERSHEY_SIMPLEX
    # переводим его в ч/б
    gray = cv2.cvtColor(ph, cv2.COLOR_BGR2GRAY)
    # определяем лица на фото
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15, minSize=(50, 50),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    count = 0
    # перебираем все найденные лица
    for (x, y, w, h) in faces:
        count += 1
        # получаем id пользователя
        nbr_predicted, coord = recognizer.predict(gray[y:y + h, x:x + w])
        # рисуем прямоугольник вокруг л≈ица
        cv2.rectangle(ph, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
        # если мы знаем id пользователя
        # добавляем текст к рамке
        cv2.putText(ph, str(nbr_predicted), (x, y + h), font, 1.1, (0, 255, 0))
        # выводим окно с изображением с камеры

    resized = cv2.resize(ph, (1000, 1000), interpolation=cv2.INTER_AREA)
    cv2.imshow('Face recognition', resized)
    # делаем паузу
    cv2.waitKey(0)
    return count


if __name__ == "__main__":
    face_rec_ph("imgs/test4.jpg")
