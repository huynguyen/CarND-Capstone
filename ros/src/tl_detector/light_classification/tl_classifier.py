from styx_msgs.msg import TrafficLight
import cv2
import tensorflow as tf
from keras import backend as K
from keras.models import load_model 
import os
import rospy
import numpy as np

class TLClassifier(object):
    def __init__(self, config):
        #TODO load classifier
        self.model = None
        self.graph = None
        self.model_path = './models/sim_model.h5'

        if not(os.path.exists(self.model_path)):
            rospy.logerror("Failed to find model at path: {}".format(self.model_path))
        else:
            self.model = load_model(self.model_path)
            self.model._make_predict_function()
            self.graph = K.tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        try:
            with self.graph.as_default():
                if self.graph == None:
                    return TrafficLight.UNKNOWN
                    
                img = np.reshape(image, (1, 600, 800, 3))
                score_list = self.model.predict(img)

                if score_list is None or len(score_list) == 0:
                    return TrafficLight.UNKNOWN

                light_type = np.argmax(score_list)
                if (light_type == 0):
                    return TrafficLight.RED
                elif(light_type == 1):
                    return TrafficLight.GREEN
                else:
                    return TrafficLight.UNKNOWN

        except Exception as e:
            rospy.logerr("Something went horribly wrong with the classifier.")
            rospy.logerr(e)
            return TrafficLight.UNKNOWN
