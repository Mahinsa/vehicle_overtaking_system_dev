import numpy as np
import cv2
import math
import logging

class laneLines():
    
    def lane_line_process(self, image):

        def grayscale(img):
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        def gaussian_blur(img, kernel_size=5):
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        def canny(img, low_threshold = 50, high_threshold = 150):
            return cv2.Canny(img, low_threshold, high_threshold)

        def region_selection(img):
            mask = np.zeros_like(img)   
            #Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
            if len(img.shape) > 2:
                channel_count = img.shape[2]
                ignore_mask_color = (255,) * channel_count
            else:
                ignore_mask_color = 255
            #We could have used fixed numbers as the vertices of the polygon,
            #but they will not be applicable to images with different dimesnions.
            rows, cols = img.shape[:2]
            bottom_left  = [cols * 0.1, rows * 0.95]
            top_left     = [cols * 0.4, rows * 0.6]
            bottom_right = [cols * 0.9, rows * 0.95]
            top_right    = [cols * 0.6, rows * 0.6]
            vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
            cv2.fillPoly(mask, vertices, ignore_mask_color)
            masked_image = cv2.bitwise_and(img, mask)
            return masked_image
        
        def slope(x1, y1, x2, y2):
            if (x2 - x1) == 0:
                return 0
            else:
                return (y2 - y1) / (x2 - x1)
            
        
        def separate_lines(lines):
            right = []
            left = []
            for x1,y1,x2,y2 in lines[:, 0]:
                m = slope(x1,y1,x2,y2)
                if m >= 0:
                    right.append([x1,y1,x2,y2,m])
                else:
                    left.append([x1,y1,x2,y2,m])
            return right, left

        def hough_lines_basic(img, rho, theta, threshold, min_line_len, max_line_gap):
#             distanceChange = 0
            lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
            line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

            rightLines, leftLines = separate_lines(lines)
#             if (leftLines[0][1] - leftLines[0][3]) > 200
#             print(leftLines[0][3],leftLines[0][1])
            print(leftLines[0][1]-leftLines[0][3])
    #     distanceBetweenLines_values.append(right_lines[1])
    #     for distanceBetweenLines,distanceBetweenLinesPrevious in zip(distanceBetweenLines_values[1:],distanceBetweenLines_values):
    #         distanceChange = abs(distanceBetweenLines-distanceBetweenLinesPrevious)
    #     print(distanceChange)

            draw_lines(line_img, lines)
            return line_img

        def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
            return cv2.addWeighted(initial_img, α, img, β, λ)

        def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
            if lines is None:
                exit()
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        #Reduce image to single grayscale dimension
        gray_image = grayscale(image)
        #get image dimensions
        dims = gray_image.shape

        #Set kernel size and apply gaussian smoothing (aka low pass filter)
        kernel_size = 5
        low_pass_image = gaussian_blur(gray_image, kernel_size)

        #Define the parameters and apply canny edge detection algorithm
        low_threshold = 50
        high_threshold = 150
        edge_image = canny(low_pass_image, low_threshold, high_threshold)

        #Define the vertices and apply a mask to the edge detected image (4sided polygon)
    #     vertices = np.array([[(0,dims[0]),(450, 320),(520, 320),(dims[1], dims[0])]], dtype=np.int32)
        mask_image = region_selection(edge_image)

        #Define paramters and apply hough
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 10    # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 25 #minimum number of pixels making up a line
        max_line_gap = 30    # maximum gap in pixels between connectable line segments

        image_lines = hough_lines_basic(mask_image, rho, theta, threshold, min_line_length, max_line_gap)

        final_result = weighted_img(image_lines, image)

        return final_result