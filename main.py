EdgeDetection
Owner
me
Modified Mar 29, 2022
Created Mar 29, 2022 

#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import matplotlib.pyplot as plt


class CannyFilter(object):

    def __init__(self):
        self.image_sub = rospy.Subscriber("/camera_fr/image_raw", Image, self.camera_callback)
        self.bridge_object = CvBridge()

    def region_of_interest(self, image):
        h = image.shape[0]
        w = image.shape[1]
        polygans = np.array([[(0, h-10), (w, h-10), (w, 125), (0, 125)]])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygans, 255)
        masked_img = cv2.bitwise_and(image, mask)
        return masked_img

    def raw_line(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
        return line_image

    def camera_callback(self, data):
        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        img = cv2.resize(cv_image, (450, 350))
        #cv2.imshow('Original', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # erosion
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel_a = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(blur, kernel_a, iterations=1)
        edges1 = cv2.Canny(erosion, 50, 100)
        cropped1 = self.region_of_interest(edges1)
        #cv2.imshow('Edges1', cropped1)
        cv2.waitKey(1)

        # closing
        blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=0, sigmaY=0)
        kernel_d = np.ones((7, 7), np.uint8)
        closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel_d)
        edges2 = cv2.Canny(closing, 50, 100)
        cropped2 = self.region_of_interest(edges2)
        cv2.imshow('Edges2', cropped2)
        lines = cv2.HoughLinesP(cropped2, 1.5, np.pi / 180, 10, np.array([]), minLineLength=10, maxLineGap=5)
        show_lines_img = self.raw_line(cropped2, lines)
        #cv2.imshow('lines', show_lines_img)

        img_combo = cv2.addWeighted(gray, 0.8, show_lines_img, 1, 1)
        cv2.imshow('img_combo', img_combo)
        cv2.waitKey(1)

        # blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 100)
        cropped = self.region_of_interest(edges)
        #cv2.imshow('Edges', cropped)
        cv2.waitKey(1)
        # plt.imshow(edges)
        # plt.show()


def main():
    canny_filter_object = CannyFilter()
    rospy.init_node('canny_filter_node', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
