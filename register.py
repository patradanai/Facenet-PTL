import cv2
import align.detect_face
import tensorflow as tf

npy = './align'
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
frame_interval = 3
batch_size = 1000
image_size = 183
input_image_size = 160

# init Cv2

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        cap = cv2.VideoCapture(1)
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, npy)
        while True:
            ret, frame = cap.read()

            bounding_boxes, points = align.detect_face.detect_face(
                frame, minsize, pnet, rnet, onet, threshold, factor)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print("FPS: ", fps)
            if ret:
                for i in range(len(bounding_boxes)):
                    cv2.rectangle(frame, (int(bounding_boxes[i][0]), int(bounding_boxes[i][1])), (
                        int(bounding_boxes[i][2]), int(bounding_boxes[i][3])), (0, 255, 0), 2)

                for p in points.T:
                    for x in range(5):
                        cv2.circle(frame, (p[x], p[x+5]), 1, (0, 255, 0), 2)

                cv2.putText(frame, "FPS : {}".format(fps), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), False)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
