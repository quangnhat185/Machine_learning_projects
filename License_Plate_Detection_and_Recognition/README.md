### The methodology to detect and recongize License Plate contains 3 main stages:
  1. Detect and crop the region of License Plate from rear care image using WPOD model
  
  <p align = 'center'>
  <img src = './plate_crop.png', width=480>
  </p>
  
  2. Using common OpenCV techniques such as blur, threshold, findContour to crop all Plate's digits
  
  <p align = 'center'>
  <img src = './grab_digit_contour.png', width=480>
  </p>
  
  3. Using a Nvidia model which was trained to recognize plate digits. The accuracy of this training model as well as the result can be seen as below:
  <p align='center'>
  <img src = './Accuracy.png', width=320>
  <img src = './Loss.png', width = 320>
  <p>
  <p align='center'>
  <img src = './digit_crops.png', width = 600>
  </p>
  <br>
  <p align='center'>
  <img src = './final_result.jpg', width = 600>
  </p>

## The End!
