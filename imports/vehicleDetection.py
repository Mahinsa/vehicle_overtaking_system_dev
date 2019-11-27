import cv2
import numpy as np
import math

class BoundingBox():
    
    def __init__(self):    
        #initialization of arguments and lists
        self.cenX_values = []
#         self.cenY_values = []
        self.distanceFromCamera_values = []
    
    def getBoundingBox(self, image, engine):
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        
        def region_selection_left(img):
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
            bottom_left  = [cols * 0.2, rows * 0.95]
            top_left     = [cols * 0.4, rows * 0.2]
            bottom_right = [cols * 0.8, rows * 0.95]
            top_right    = [cols * 0.6, rows * 0.2]
            vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
            cv2.fillPoly(mask, vertices, ignore_mask_color)
            masked_image = cv2.bitwise_and(img, mask)
            return masked_image
        
        def region_selection_right(img):
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
            bottom_left  = [cols * 0.6, rows * 0.95]
            top_left     = [cols * 0.7, rows * 0.2]
            bottom_right = [cols * 1.0, rows * 0.95]
            top_right    = [cols * 0.9, rows * 0.2]
            vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
            cv2.fillPoly(mask, vertices, ignore_mask_color)
            masked_image = cv2.bitwise_and(img, mask)
            return masked_image

        # function to get the output layer names 
        # in the architecture
        def get_output_layers(net):
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            return output_layers
        
        def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, engine):
            car_width = 1.905
            camera_focal_length = 350.00
            
            label = str(classes[class_id])
            cX = abs((x+x_plus_w)/2)
            cY = abs((y+y_plus_h)/2)
            self.cenX_values.append(cX)
#             self.cenY_values.append(cY)
            dstX = 0
            dstY = 0
            distanceChange = 0
            speed = 0
            # determine on-going and previous values in the cenX list
            for cenX,cenPreviousX in zip(self.cenX_values[1:], self.cenX_values):
                dstX = abs(cenX-cenPreviousX)
#             for cenY,cenPreviousY in zip(self.cenY_values[1:], self.cenY_values):
#                 dstY = abs(cenY-cenPreviousY)
#             displacementOfCentroids = math.sqrt(dstX**2+dstY**2)
#             print(displacementOfCentroids)
            distanceFromCamera = (car_width*camera_focal_length)/(x_plus_w-x)
#            self.distanceFromCamera_values.append(distanceFromCamera)
#            for dis,disPrevious in zip(self.distanceFromCamera_values[1:], self.distanceFromCamera_values):
#                distanceChange = abs(dis-disPrevious)
#            for distanceFromCamera,distanceFromCameraPrevious in zip(self.distanceFromCamera_values[1:], self.distanceFromCamera_values):
#                distanceChange = abs(distanceFromCamera-distanceFromCameraPrevious)
#                speed = (distanceChange/Time)*3.6
#         print(distance)
#         speed = distance/0.11
#         text = 'speed:{:.2f}'.format(speed)
            color = COLORS[class_id]
            if label == 'car':
                cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
                cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(img, "{0:.2f}m".format(distanceFromCamera), (x-10,y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
#                cv2.putText(img, "{0:.2f}m".format(distanceChange), (x-10,y-75), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                if dstX>=300 and dstX<=400 and distanceChange<15:
                    cv2.putText(img, "do not overtake the vehicle", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                    engine.say("do not overtake the vehicle")
                    engine.runAndWait()
#             cv2.putText(img, "{0:.2f}pixels".format(displacementOfCentroids), (x-10,y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
        # read class names from text file
        classes = None
        with open('yolov3.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # generate different colors for different classes 
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        # read pre-trained model and config file
        net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
        
        #create masked image only with restricted area from original image
        left_masked_image = region_selection_left(image)
        right_masked_image = region_selection_right(image)
#         cv2.imshow("frame", left_masked_image)
#         cv2.imshow("frame1", right_masked_image)

        # create input blob 
        blob_left = cv2.dnn.blobFromImage(left_masked_image, scale, (416,416), (0,0,0), True, crop=False)
        blob_right = cv2.dnn.blobFromImage(right_masked_image, scale, (416,416), (0,0,0), True, crop=False)

        # set input blob for the network
        net.setInput(blob_left)
            
        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), engine)

        # set input blob for the network
        net.setInput(blob_right)
            
        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), engine)
            return image