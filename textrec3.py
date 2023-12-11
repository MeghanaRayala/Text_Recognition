from tkinter import *
import tkinter.filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
import pyttsx3
import pytesseract as p
import easyocr
import imutils
import numpy as np
p.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
def resize(pic):
    chng = cv2.resize(pic, (900, 600))
    return chng
def max_bound_area(items):
    big = np.array([])
    max_area = 0
    for i in items:
        area = cv2.contourArea(i)
        if area > 5000:
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
            if (area > max_area) and len(approx) == 4:
                big = approx
                max_area = area
    return big, max_area
def sort_contour_points(points):
    points = points.reshape((4, 2))
    b = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    dif = np.diff(points, axis=1)
    b[0] = points[np.argmin(add)]
    b[1] = points[np.argmin(dif)]
    b[2] = points[np.argmax(dif)]
    b[3] = points[np.argmax(add)]
    return b
def filter_and_edgedetect(pic):
    filter = cv2.bilateralFilter(pic, 11, 21, 21)
    edges = cv2.Canny(filter, 30, 200)
    return edges
def contours_and_mask(pic, img, gray):
    points = cv2.findContours(pic.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(points)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    loc = []
    for item in contours:
        area = cv2.approxPolyDP(item, 10, True)
        if len(area) == 4:
            loc.append(area)
    mask = np.zeros(pic.shape, np.uint8)
    for i in loc:
        chng = cv2.drawContours(mask, [i], 0, 255, -1)
        chng = cv2.bitwise_and(img, img, mask=mask)
    (x, y) = np.where(mask == 255)
    for i in range(len(loc)):
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        crop = img[x1:x2 + 1, y1:y2 + 1]
    return crop
def padding(pic):
    colour = [255, 255, 255]
    pad = cv2.copyMakeBorder(pic, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=colour)
    chng = cv2.resize(pad, (900, 600))
    return chng
'''def extract_text_from_image(pic):
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    extracted_text = p.image_to_string(gray, config='--psm 6 -l eng')  # Specify language and font
    return extracted_text'''
def scene_detect(path):
    image = cv2.imread(path)
    z = padding(image)
    a = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
    b = filter_and_edgedetect(a)
    c = contours_and_mask(b, z, a)
    read = easyocr.Reader(['en'], gpu=False)
    colour = [255, 255, 255]
    c = cv2.copyMakeBorder(c, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=colour)
    res = read.readtext(c)
    for i in res:
        print(i[1])
        cord_1 = tuple(i[0][0])
        cord_2 = tuple(i[0][2])
        k = (int(cord_1[0]), int(cord_1[1]))
        t = (int(cord_2[0]), int(cord_2[1]))
        text = i[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        c = cv2.rectangle(c, k, t, (0, 0, 0), 1)
        c = cv2.putText(c, text, k, font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        player = pyttsx3.init()
        player.say(text)
    player.runAndWait()
    cv2.imshow("output", c)
    cv2.waitKey(0)
def deskew(pic):
    newImage = pic.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return rotate(pic, -1.0 * angle)
def rotate(pic, angle: float):
    newImage = pic.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage
def preview_image():
    filename = askopenfilename(filetypes=[("All files", "*.*")])
    img = Image.open(filename)
    img = img.resize((550, 480), resample=Image.BILINEAR)
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image = img
    scene_detect(filename)
root = Tk()
frame = Frame(root)
frame.pack(side=BOTTOM, padx=15, pady=15)
lbl = Label(root)
lbl.pack()
btn1 = Button(frame, text="Browse Image", command=preview_image)
btn1.pack(side=tkinter.LEFT, padx=10)
i = IntVar()
r = Radiobutton(root, text="Scene detection", value=1, variable=i)
r.pack()
btn2 = Button(frame, text="Exit", command=root.destroy)
btn2.pack(side=tkinter.LEFT)
root.title("Text Extractor")
root.geometry("300x300")
root.mainloop()
