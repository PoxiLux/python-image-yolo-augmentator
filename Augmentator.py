import os
import cv2
import glob 
import time
import random
import shutil
import numpy as np
import datetime
from alive_progress import alive_bar

os.chdir(os.path.dirname(__file__))

if not os.path.exists('images'):
    os.mkdir('images')

os.chdir("images")

# Start of functions! --------------------

def splitFiles(output, originalFiles, validRatio = 0.15, testRatio = 0.08):
    files = glob.glob("*.png")
    files.extend(glob.glob("*.jpg"))

    if os.path.isdir(output):
        shutil.rmtree(output)

    os.makedirs(output)
    os.makedirs(output +'/train/')
    os.makedirs(output + '/valid/')
    os.makedirs(output + '/test/')

    files = glob.glob("*.png")
    files.extend(glob.glob("*.jpg"))

    print("Splitting " + str(len(originalFiles)) + " images. Valid:" + str(validRatio) + " Test:" + str(testRatio));

    random.shuffle(originalFiles)
    _, validFiles, testFiles = np.split(originalFiles, [int(len(originalFiles) * (1 - (validRatio + testRatio))), int(len(originalFiles) * (1 - testRatio))])

    with alive_bar(len(validFiles)) as compute:
        for file in validFiles:
            (name, ext) = file.split('.')
            shutil.copy(file, output + '/valid/' + file)

            if os.path.isfile(name + '.txt'):
                shutil.copy(name + '.txt', output + '/valid/' + name + '.txt')

            compute()

    with alive_bar(len(testFiles)) as compute:
        for file in testFiles:
            (name, ext) = file.split('.')
            shutil.copy(file, output + '/test/' + file)

            if os.path.isfile(name + '.txt'):
                shutil.copy(name + '.txt', output + '/test/' + name + '.txt')

            compute()

    with alive_bar(len(files)) as compute:
        for file in files:
            (name, ext) = file.split('.')
            shutil.copy(file, output + '/train/' + file)

            if os.path.isfile(name + '.txt'):
                shutil.copy(name + '.txt', output + '/train/' + name + '.txt')

            compute()

    if os.path.isfile('classes.txt'):
        shutil.copy('classes.txt', output + '/train/' + '_darknet.labels')
        shutil.copy('classes.txt', output + '/valid/' + '_darknet.labels')
        shutil.copy('classes.txt', output + '/test/' + '_darknet.labels')

    print("Zipping dataset, please wait!");
    shutil.make_archive("output", 'zip', output)

    print("Cleaning up...")
    shutil.rmtree(output)

    files = glob.glob("*.png")
    files.extend(glob.glob("*.jpg"))
    files.extend(glob.glob('*.txt'))

    with alive_bar(len(files)) as compute:
        for file in files:
            os.remove(file)

            compute()

    print("Training dataset zipped! See: ./images/output.zip");


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

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

# End of functions! ----------------------

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))
filesToIgnore = []

# Check if images exist
if (len(files) == 0):
    print("No iamges found in /images")
    os._exit(0)

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))
index = 1

# Rename filenames and delete old files
print("Fixing file sequence...")
with alive_bar(len(files)) as compute:
    for file in files:
        (name, ext) = file.split('.')

        shutil.copy(file, "x_" + str(index) + '.' + ext)
        os.remove(file)
        
        if os.path.isfile(name + '.txt'):
            shutil.copy(name + '.txt', "x_" + str(index) + '.txt')
            os.remove(name + '.txt')

        index += 1
        compute()

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))

# Rename files and remove any missing labels
with alive_bar(len(files)) as compute:
    for file in files:
        (name, ext) = file.split('.')

        if ("x_" in name):
            newName = name.replace('x_', '');
            
            if os.path.isfile(name + '.txt'):
                os.rename(name + '.txt', newName + '.txt')
                os.rename(file, newName + '.' + ext)
            else:
                os.remove(file)
        else:
            os.remove()

        compute()
        
files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))
orgFiles = files.copy()

print("Resizing images...")
with alive_bar(int((len(files)))) as compute:
    for file in files:
        
        try:
            img = cv2.imread(file)

            if (img.shape[0] != 416 or img.shape[1] != 416):
                resized = image_resize(img, 416, 416)
                cv2.imwrite(file, resized)
        except:
            print("Failed to resize an image!")
            os._exit(0)

        compute()

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))
randomfiles = random.choices(files, k=round(len(files)*.02)); # only 2% of images
count = len(files)

print("Mixing up...")
with alive_bar(int((len(randomfiles)))) as compute:
    for file1 in randomfiles:
        (name1, ext1) = file1.split('.')

        xFiles = glob.glob("*.png")
        xFiles.extend(glob.glob("*.jpg"))

        for file2 in randomfiles:
            (name2, ext2) = file2.split('.')

            if file1 == file2:
                continue

            try:
                img1 = cv2.imread(file1)
                img2 = cv2.imread(file2)

                mixup = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

                count += 1
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
            except:
                print("Failed to mixup: " + file)

        compute()

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))
count = len(files)

print("Augmenting...")
with alive_bar(len(files)) as compute:
    for file in files:
        (name, ext) = file.split('.')

        if int(name) in filesToIgnore:
            compute()
            continue

        xFiles = glob.glob("*.png")
        xFiles.extend(glob.glob("*.jpg"))
        newLines = []
        img = cv2.imread(file)

        # Flipped
        flipped = flip(img, 1)
        count += 1
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
        count += 1
        cv2.imwrite(str(count) + '.' + ext, brigher)
        
        if os.path.isfile(name + '.txt'):
            shutil.copy(name + '.txt', str(count) + '.txt')

        # Darkness UP
        darker = adjust_gamma(img, 0.5)
        count += 1
        cv2.imwrite(str(count) + '.' + ext, darker)

        if os.path.isfile(name + '.txt'):
            shutil.copy(name + '.txt', str(count) + '.txt')

        compute()

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))
randomfiles = random.choices(files, k=round(len(files)*.05)); # only 5% of images
count = len(files)

print("Generating noise")
with alive_bar(len(randomfiles)) as compute:
    for file in randomfiles:
        (name, ext) = file.split('.')

        if int(name) in filesToIgnore:
            compute()
            continue

        xFiles = glob.glob("*.png")
        xFiles.extend(glob.glob("*.jpg"))
        img = cv2.imread(file)
        
        try:
            prosessed = sp_noise(img, 0.05)
            count += 1
            filesToIgnore.append(count)
            cv2.imwrite(str(count) + '.' + ext, prosessed)

            if os.path.isfile(name + '.txt'):
                shutil.copy(name + '.txt', str(count) + '.txt')
        except:
            print("Failed to add noise to " + file)

        compute()

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))
randomfiles = random.choices(files, k=round(len(files)*.05)); # only 5% of images
count = len(files)

print("Generating cutouts")
with alive_bar(len(randomfiles)) as compute:
    for file in randomfiles:
        (name, ext) = file.split('.')

        if int(name) in filesToIgnore:
            compute()
            continue
        
        xFiles = glob.glob("*.png")
        xFiles.extend(glob.glob("*.jpg"))

        try:
            img = cv2.imread(file)

            prosessed = cutout(img, random.randint(6, 18), 50)
            count += 1
            filesToIgnore.append(count)
            cv2.imwrite(str(count) + '.' + ext, prosessed)

            if os.path.isfile(name + '.txt'):
                shutil.copy(name + '.txt', str(count) + '.txt')
        except:
            print("Failed to add cutouts to " + file)

        compute()


splitFiles("../output", orgFiles)
