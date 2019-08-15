import numpy as np
import cv2
from PIL import Image
from sklearn import svm
from sklearn.externals import joblib

def load_fnt_dataset():
    """
    Loads the Chars74K's "Fnt" dataset. Returns a list
    of flat arrays (lists) representing greyscale images, and
    their associated labels.
    Can be downloaded from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
    Direct download link: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz
    """
    images = np.zeros((36576, 16384), np.bool_)
    labels = np.zeros((36576, 1), np.uint8)
    indice = 0
    for i in range(1, 1017):
        for x in range(1, 37):
            folder_name = "Sample{0:03d}".format(x)
            file_name_prefix = "img{0:03d}-".format(x)

            file_name = "{0}{1:05d}.png".format(file_name_prefix, i)
            img_path = "Fnt\{}\{}".format(folder_name, file_name)

            this_img = cv2.imread(img_path, 0)#Image.open(img_path)
            this_data_img = this_img
            this_target_label = x-1

            images[indice, :] = this_img.reshape(1, 16384)
            labels[indice, 0] = this_target_label
            indice += 1
    print(indice)
    return images, labels
#128x128
print("Cargando datos")
x, y = load_fnt_dataset()
x2 = x
mask = x2 == 255
x2[mask] = 1
#np.savetxt("prueba.csv", x2[1, :].reshape(128, 128), fmt="%i")
print("Entrenando")
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(x2, y)
print("Probando")
#print("Prediction: ", clf.predict([x2[9]]), "Result: ", y[9])
print(x.shape, y.shape)
joblib.dump(clf, 'filename.pkl')