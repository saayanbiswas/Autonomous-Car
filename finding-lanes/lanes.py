import cv2
import numpy as np
import matplotlib.pyplot as plt
def canny(image):
    gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5), 0)
    canny=cv2.Canny(blur, 50, 150)
    return(canny)

def region_of_interest(image):
    height=image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height),(550,250)]
    ]) # region of interest estimated from math lib plots
    mask=np.zeros_like(image) #creates a 2d matrix (pixel image) of 0s (black image of same resolution as 'image')
    cv2.fillPoly(mask,polygons,255) #fill the mask with triangle with white color

    masked_image= cv2.bitwise_and(image,mask)
    return(masked_image)

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        # for line in lines:
        #     x1,y1,x2,y2=line.reshape(4)
        #     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 10)
        #     #blue(255,0,0) width=10px 
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 10)
    return(line_image)

def make_coordinate(image, line_parameters):
    slope, intercept=line_parameters
    y1=image.shape[0]#represents the height (bottom most part of image since top is 0)
    y2=int(y1*(3/5)) #assuming that y1 starts at 704 and goes to 420 i.e 3/5ths the way
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2),1)
        #poly fit 2 points of polynomial of degree 1 i.e linear, returns m and b
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average=np.average(left_fit, axis=0)
    right_fit_average=np.average(right_fit, axis=0)
    left_line=make_coordinate(image, left_fit_average)
    right_line=make_coordinate(image, right_fit_average)
    return np.array([left_line,right_line])
    


# lane_image=np.copy(image)
# canny_image=canny(lane_image)
# cropped_image=region_of_interest(canny_image)
# lines=cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=5)   
# ''' 2 pixels binsize with 1 degree precision, threshold of 100 votes on each bin
#     5th argument is placeholder array, 6th argument is threshold length of line in pixels,
#     maxLinegap in pixels between segmented lines to be assumed as an single line    
# '''
# averaged_lines=average_slope_intercept(lane_image, lines)

# line_image=display_lines(lane_image,averaged_lines)
# combo_image=cv2.addWeighted(lane_image,0.8, line_image,1,1)
# # blend images (add pixels of both with multiplier 0.8 and 1 to each) and 1 is gamma
# # plt.imshow(canny)
# # plt.show()
# cv2.imshow("result",combo_image)
# cv2.waitKey(0)

cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame=cap.read()
    lane_image=np.copy(frame)
    canny_image=canny(lane_image)
    cropped_image=region_of_interest(canny_image)
    lines=cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=5)   
    ''' 2 pixels binsize with 1 degree precision, threshold of 100 votes on each bin
        5th argument is placeholder array, 6th argument is threshold length of line in pixels,
        maxLinegap in pixels between segmented lines to be assumed as an single line    
    '''
    averaged_lines=average_slope_intercept(lane_image, lines)

    line_image=display_lines(lane_image,averaged_lines)
    combo_image=cv2.addWeighted(lane_image,0.8, line_image,1,1)
    # blend images (add pixels of both with multiplier 0.8 and 1 to each) and 1 is gamma
    # plt.imshow(canny)
    # plt.show()
    cv2.imshow("result",combo_image)
    if (cv2.waitKey(1) & 0xFF== ord('q')):
        break

cap.release()
cv2.destroyAllWindows()