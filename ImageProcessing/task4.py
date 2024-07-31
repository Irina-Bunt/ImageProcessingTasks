import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkinter as tk

img = cv2.imread('./maxresdefault.jpg')
img2 = cv2.imread('./lena.png')
plt.hist(img2.ravel(), 256, [0, 256])
plt.xlabel('Яркость')
plt.ylabel('Частота')
plt.title('Гистограмма 1')
plt.show()

img3 = cv2.imread('./lena.png', 0)
# Применение эквализации гистограммы
equalized_image = cv2.equalizeHist(img3)
cv2.imshow('Equalized Image', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.hist(img3.ravel(), 256, [0, 256])
plt.xlabel('Яркость')
plt.ylabel('Частота')
plt.title('Гистограмма 2')
plt.show()


def update_value(event):
    value_label.config(text=f"Значение: {slider_a.get()}")
    value_label.config(text=f"Значение: {slider_b.get()}")
    value_label.config(text=f"Значение: {slider_g.get()}")

    img_new = cv2.convertScaleAbs(img, alpha=slider_a.get(), beta=slider_b.get())

    gamma = slider_g.get()
    gamma_corrected = np.power(img_new / 255.0, gamma) * 255.0
    gamma_corrected = np.uint8(gamma_corrected)

    cv2.imshow("win1", gamma_corrected)

root = tk.Tk()
root.title("Ползунок")

    # Создаем ползунок
slider_a = tk.Scale(root, from_=0, to=10, orient="horizontal", command=update_value)
slider_b = tk.Scale(root, from_=0, to=50, orient="horizontal", command=update_value)
slider_a.pack()
slider_b.pack()

slider_g = tk.Scale(root, from_=0, to=3, orient="horizontal", command=update_value)
slider_g.pack()

    # Метка для отображения значения ползунка
value_label = tk.Label(root, text="Значение: 0")
value_label.pack()

root.mainloop()

#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



