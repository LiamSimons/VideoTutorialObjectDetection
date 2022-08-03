from Detector import *
# Here you can find the models:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
classFile = "objects"
imagePath = "present.6.jpg"
treshold = 0.2

if __name__ == '__main__':
    detector = Detector()
    detector.readClasses(classFile)
    detector.downloadModel(modelURL)
    detector.loadModel()
    detector.predictImage(imagePath, treshold)
