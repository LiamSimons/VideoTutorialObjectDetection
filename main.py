from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz"
classFile = "objects"
imagePath = "test/1.jpg"

if __name__ == '__main__':
    detector = Detector()
    detector.readClasses(classFile)
    detector.downloadModel(modelURL)
    detector.loadModel()
    detector.predictImage(imagePath)
