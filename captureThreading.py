from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import cv2
import numpy as np
import align.detect_face
import tensorflow as tf
import os
import time
import pickle
import facenet
from PIL import Image


class RecordVideo(QObject):
    # Signal * Slot
    imageData = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    imageList = pyqtSignal(list)
    # Model
    modeldir = './model/20180402-114759.pb'
    classifier_filename = './class/classifier.pkl'
    npy = './align'
    train_img = "./imageTrain"

    # Store Image Current
    imgStore = []

    # Variable
    finishFlag = False

    def __init__(self, parent=None):
        super(RecordVideo, self).__init__(parent)

        self.finished.connect(self.finishedFunc)

    def startRecord(self):
        print('Creating networks and loading parameters')
        print(tf.test.is_gpu_available())
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=1)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, self.npy)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 183
            input_image_size = 160

            HumanNames = os.listdir(self.train_img)
            HumanNames.sort()

            print('Loading Modal')
            facenet.load_model(self.modeldir)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser(
                self.classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            video_capture = cv2.VideoCapture(0)
            c = 0

            print('Start Recognition')
            prevTime = 0
            while True:
                # Real Video
                ret, frame = video_capture.read()
                QApplication.processEvents()

                # Resize
                frame = cv2.resize(frame, (460, 350), fx=0.5, fy=0.5)

                # Return Boxpoint and Keypoint
                bounding_boxes, points = align.detect_face.detect_face(
                    frame, minsize, pnet, rnet, onet, threshold, factor)
                # fps = video_capture.get(cv2.CAP_PROP_FPS)

                # print("FPS : {}" .format(fps))
                curTime = time.time()+1    # calc fps
                timeF = frame_interval

                if (c % timeF == 0):
                    find_results = []

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, points = align.detect_face.detect_face(
                        frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    # print('Detected_FaceNum: %d' % nrof_faces)

                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                        for i in range(nrof_faces):

                            # Check Face not over
                            if i > len(cropped):
                                break
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('Face is very close!')
                                continue

                            cropped.append(
                                frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            # Update PTL IMAGE
                            scaled.append(np.array(Image.fromarray(cropped[i]).resize(
                                (image_size, image_size), Image.BILINEAR)))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(
                                scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                            feed_dict = {
                                images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(
                                embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            # print(predictions)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(
                                len(best_class_indices)), best_class_indices]
                            # print("predictions")
                            # print(best_class_indices, ' with accuracy ',
                            #       best_class_probabilities)

                            # Probalility over 80 percent
                            if best_class_probabilities > 0.8:
                                # boxing face
                                cv2.rectangle(
                                    frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)

                                for p in points.T:
                                    for index in range(5):
                                        cv2.circle(
                                            frame, (p[index], p[index+5]), 1, (0, 0, 255), 2)

                                # plot result idx under box
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                # print('Result Indices: ',
                                #       best_class_indices[0])
                                print(HumanNames)
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = "{}, {:.2f}%".format(HumanNames[best_class_indices[0]
                                                                                       ], best_class_probabilities[0]*100)
                                        cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 255), thickness=1, lineType=2)

                                    # Store Name
                                    if not HumanNames[best_class_indices[0]] in self.imgStore:
                                        self.imgStore.append(
                                            HumanNames[best_class_indices[0]])

                                        # Emit List Name
                                        self.imageList.emit(self.imgStore)
                            else:
                                # boxing face
                                cv2.rectangle(
                                    frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                name = "Unknown, {:.2f}%".format(
                                    best_class_probabilities[0]*100)
                                cv2.putText(frame, name, (bb[i][0], bb[i][3] + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 255), thickness=1, lineType=2)
                    else:
                        print('Alignment Failure')
                if self.finishFlag:
                    break

                # Emit Image to pyQt5
                self.imageData.emit(frame)

            # Clean up
            cv2.destroyAllWindows()
            video_capture.release()

    def finishedFunc(self):
        self.finishFlag = True
