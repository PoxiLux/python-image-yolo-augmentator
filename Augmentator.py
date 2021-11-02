import os
import cv2
import glob 
import time
import random
import shutil
import numpy as np
import datetime
from alive_progress import alive_bar

# --- START OF ADJUSTABLE VARIABLES

mixupPercent = .002 
augmentPercent = .20
noisePercent = .05
cutoutPercent = .05

mixups = True
augment = True
noise = True
cutout = True

generateSplitOutput = True
validPercent = 0.08
testPercent = 0.05

# --- END OF ADJUSTABLE VARIABLES

os.chdir(os.path.dirname(__file__))

if not os.path.exists('images'):
    os.mkdir('images')

os.chdir("images")

# Start of functions! --------------------

def splitFiles(output, originalFiles, validRatio, testRatio):
    files = glob.glob("*.jpg")

    if os.path.isdir(output):
        shutil.rmtree(output)

    os.makedirs(output)
    os.makedirs(output +'/train/')
    os.makedirs(output + '/valid/')
    os.makedirs(output + '/test/')

    files = glob.glob("*.jpg")

    random.shuffle(originalFiles)
    trainFiles, validFiles, testFiles = np.split(originalFiles, [int(len(originalFiles) * (1 - (validRatio + testRatio))), int(len(originalFiles) * (1 - testRatio))])

    print("Splitting " + str(len(files)) + " images. Valid:" + str(validRatio) + " Test:" + str(testRatio));

    with alive_bar(len(validFiles)) as compute:
        for file in validFiles:
            (name, ext) = getNameAndExt(file)
            shutil.copy(file, output + '/valid/' + file)

            if os.path.isfile(name + '.txt'):
                shutil.copy(name + '.txt', output + '/valid/' + name + '.txt')

            compute()

    with alive_bar(len(testFiles)) as compute:
        for file in testFiles:
            (name, ext) = getNameAndExt(file)
            shutil.copy(file, output + '/test/' + file)

            if os.path.isfile(name + '.txt'):
                shutil.copy(name + '.txt', output + '/test/' + name + '.txt')

            compute()

    with alive_bar(len(files)) as compute:
        for file in files:
            (name, ext) = getNameAndExt(file)
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

    files = glob.glob("*.jpg")
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

def getNameAndExt(filePath):
    (name, ext)  = os.path.splitext(filePath)
    return (name, ext.replace('.', ''))

# End of functions! ----------------------

files = glob.glob("*.png")
files.extend(glob.glob("*.jpg"))

# Check if images exist
if (len(files) == 0):
    print("No iamges found in /images")
    os._exit(0)

index = 1

# Rename filenames and delete old files
print("Fixing file sequence...")
with alive_bar(len(files)) as compute:
    for file in files:
        (name, ext) = getNameAndExt(file)

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
        (name, ext) = getNameAndExt(file)

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

print("Resizing and converting images...")
with alive_bar(int((len(files)))) as compute:
    for file in files:
        try:
            img = cv2.imread(file)
            (name, ext) = getNameAndExt(file)

            if (img.shape[0] != 416 or img.shape[1] != 416 or ext != "jpg"):
                resized = image_resize(img, 416, 416)       
                cv2.imwrite(name + ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                if (ext != "jpg"):
                    os.remove(file)
        except:
            print("Failed to resize an image!")
            os._exit(0)

        compute()

classifierId = 0
classifier = "target"

print("Setting classifers: " + str(len(files)))
files = glob.glob("*.txt")

with alive_bar(int((len(files)))) as compute:
    for file in files:
        newLines = []

        with open(file) as f:
            for line in f:
                try:
                    if line != '':
                        (_, x, y, w, h) = line.strip().split(' ')
                        newLines.append(str(classifierId) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
                except:
                    newLines.append(line)
                    continue;

        with open(file, 'w') as f:
            for line in newLines:
                f.write(line)

        compute()

with open("classes.txt", 'w') as f:
    f.write(classifier)

print("Valid images to process: " + str(len(files)))

files = glob.glob("*.jpg")
count = len(files)
randomfiles = random.choices(files, k=round(len(files)*mixupPercent)); # only 0.2% of images

if mixups:
    print("Mixing up...")
    with alive_bar(int((len(randomfiles)))) as compute:
        for file1 in randomfiles:
            (name1, ext1) = file1.split('.')

            for file2 in randomfiles:
                (name2, ext2) = file2.split('.')

                if file1 == file2:
                    continue

                try:
                    img1 = cv2.imread(file1)
                    img2 = cv2.imread(file2)

                    mixup = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

                    count += 1
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

randomfiles = random.choices(files, k=round(len(files)*augmentPercent)); # only 20% of images

if augment:
    print("Augmenting...")
    with alive_bar(len(randomfiles)) as compute:
        for file in randomfiles:
            (name, ext) = getNameAndExt(file)

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
                        try:
                            (id, x, y, w, h) = line.strip().split(' ')
                            (uid, ux, uw, uy, uh) = unconvert(id, img.shape[0], img.shape[1], float(x), float(y), float(w), float(h))

                            oldUX = ux
                            ux = img.shape[0] - uw
                            uw = img.shape[0] - oldUX

                            (cx, cy, cw, ch) = convert((img.shape[0], img.shape[1]), (ux, uw, uy, uh))

                            newLines.append(id + " " + str(cx) + " " + str(cy) + " " + str(cw) + " " + str(ch) + "\n")
                        except:
                            newLines.append(line) 
                            continue;
                
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

randomfiles = random.choices(files, k=round(len(files)*noisePercent)); # only 5% of images

if noise:
    print("Generating noise")
    with alive_bar(len(randomfiles)) as compute:
        for file in randomfiles:
            (name, ext) = getNameAndExt(file)

            img = cv2.imread(file)
            
            try:
                prosessed = sp_noise(img, 0.05)
                count += 1
                cv2.imwrite(str(count) + '.' + ext, prosessed)

                if os.path.isfile(name + '.txt'):
                    shutil.copy(name + '.txt', str(count) + '.txt')
            except:
                print("Failed to add noise to " + file)

            compute()

randomfiles = random.choices(files, k=round(len(files)*cutoutPercent)); # only 5% of images

if cutout:
    print("Generating little cutouts")
    with alive_bar(len(randomfiles)) as compute:
        for file in randomfiles:
            (name, ext) = getNameAndExt(file)
            
            try:
                img = cv2.imread(file)

                prosessed = cutout(img, random.randint(8, 20), 16)
                count += 1
                cv2.imwrite(str(count) + '.' + ext, prosessed)

                if os.path.isfile(name + '.txt'):
                    shutil.copy(name + '.txt', str(count) + '.txt')
            except:
                print("Failed to add cutouts to " + file)

            compute()

if generateSplitOutput:
    splitFiles("../output", files, validPercent, testPercent)
