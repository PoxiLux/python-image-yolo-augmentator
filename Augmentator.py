import os
import cv2
import glob 
import time
import random
import shutil
import numpy as np
from alive_progress import alive_bar

os.chdir(os.path.dirname(__file__))

if not os.path.exists('images'):
    os.mkdir('images')

os.chdir("images")

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))
filesToIgnore = []

# Check if images exist
if (len(files) == 0):
    print("No iamges found in /images")
    os._exit(0)

def flip(img, axis = 0):
    return cv2.flip(img, axis)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([
      ((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)])
   return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def cutout(image, iterations=5, scale = 100):
    (w, h, _) = image.shape

    for i in range(iterations):
        topX = random.randint(0, w-scale)
        topY = random.randint(0, h-scale)
        bottomX = topX + scale
        bottomY = topY + scale

        cv2.rectangle(image, (topX, topY), (bottomX, bottomY), (0, 0, 0), -1)

    return image

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (float("{:.6f}".format(x)), float("{:.6f}".format(y)), float("{:.6f}".format(w)), float("{:.6f}".format(h)))

def unconvert(class_id, width, height, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))
randomfiles = random.choices(files, k=round(len(files)*.15));

print("Mixing up...")
with alive_bar( int((len(randomfiles))) ) as compute:
    for file1 in randomfiles:
        (name1, ext1) = file1.split('.')

        for file2 in randomfiles:
            xFiles = glob.glob("*.png")
            xFiles.extend(glob.glob("*.jpg"))
            (name2, ext2) = file2.split('.')

            if file1 == file2:
                continue

            img1 = cv2.imread(file1)
            img2 = cv2.imread(file2)
            
            mixup = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
            count = len(xFiles) + 1
            filesToIgnore.append(count)
            cv2.imwrite(str(count) + '.' + ext1, mixup)

            fileLines = []

            with open(name1 + '.txt') as file:
                for line in file:
                    fileLines.append(line)
            
            with open(name2 + '.txt') as file:
                for line in file:
                    fileLines.append(line)

            with open(str(count) + '.txt', 'w') as f:
                for line in fileLines:
                    f.write(line)

        compute()

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))

print("Augmenting...")
with alive_bar(len(files)) as compute:
    for file in files:
        xFiles = glob.glob("*.png")
        xFiles.extend(glob.glob("*.jpg"))
        (name, ext) = file.split('.')

        if int(name) in filesToIgnore:
            compute()
            continue

        newLines = []
        img = cv2.imread(file)

        # Flipped
        flipped = flip(img, 1)
        count = len(xFiles) + 1
        cv2.imwrite(str(count) + '.' + ext, flipped)

        if os.path.isfile(name + '.txt'):
            shutil.copy(name + '.txt', str(count) + '.txt')

            with open(name + '.txt') as file:
                for line in file:
                    (id, x, y, w, h) = line.strip().split(' ')

                    (uid, ux, uw, uy, uh) = unconvert(id, img.shape[0], img.shape[1], float(x), float(y), float(w), float(h))

                    oldUX = ux
                    ux = img.shape[0] - uw
                    uw = img.shape[0] - oldUX

                    (cx, cy, cw, ch) = convert((img.shape[0], img.shape[1]), (ux, uw, uy, uh))

                    newLines.append(id + " " + str(cx) + " " + str(cy) + " " + str(cw) + " " + str(ch) + "\n")
            
            with open(str(count) + '.txt', 'w') as f:
                for line in newLines:
                    f.write(line)

        # Brightness UP
        brigher = increase_brightness(img, 30)
        count = len(xFiles) + 2
        cv2.imwrite(str(count) + '.' + ext, brigher)
        
        if os.path.isfile(name + '.txt'):
            shutil.copy(name + '.txt', str(count) + '.txt')

        # Darkness UP
        darker = adjust_gamma(img, 0.5)
        count = len(xFiles) + 3
        cv2.imwrite(str(count) + '.' + ext, darker)

        if os.path.isfile(name + '.txt'):
            shutil.copy(name + '.txt', str(count) + '.txt')

        compute()

print("Generating noise")
with alive_bar(len(files)) as compute:
    for file in files:
        xFiles = glob.glob("*.png")
        xFiles.extend(glob.glob("*.jpg"))
        (name, ext) = file.split('.')

        if int(name) in filesToIgnore:
            compute()
            continue

        img = cv2.imread(file)

        prosessed = sp_noise(img, 0.05)
        count = len(xFiles) + 1
        filesToIgnore.append(count)
        cv2.imwrite(str(count) + '.' + ext, prosessed)

        if os.path.isfile(name + '.txt'):
            shutil.copy(name + '.txt', str(count) + '.txt')

        compute()

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))

print("Generating cutouts")
with alive_bar(len(files)) as compute:
    for file in files:
        xFiles = glob.glob("*.png")
        xFiles.extend(glob.glob("*.jpg"))
        (name, ext) = file.split('.')

        if int(name) in filesToIgnore:
            compute()
            continue
        
        img = cv2.imread(file)

        prosessed = cutout(img, random.randint(6, 18), 50)
        count = len(xFiles) + 1
        cv2.imwrite(str(count) + '.' + ext, prosessed)

        if os.path.isfile(name + '.txt'):
            shutil.copy(name + '.txt', str(count) + '.txt')

        compute()


