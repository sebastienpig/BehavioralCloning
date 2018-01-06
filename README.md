# BehavioralCloning

<h3>Strategies for Collecting Data</h3>

Using the simulator on the first track I trained the car over 3 laps:
-stay in the middle of the road
-decrease speed when approaching a curve
-recover to the center when veering off the side

data are saved in a separate folder 'data' under the form of a csv and IMG folder containing screen shots of the scene

<h3> visualizing the acquired data </h3>
Each sampling has a <b>center</b>, <b>left</b> and <b>right</b> image with the same reference:<br>
<center><left><right>_2017_12_31_17_54_56_108.jpg: <br><br>

<table style="width:100%">
  <tr>
    <th>CENTER</th>
    <th>LEFT</th> 
    <th>RIGHT</th>
  </tr>
  <tr>
    <td><img src="assets/center_2017_12_31_17_54_56_108.jpg"></td>
    <td><img src="assets/left_2017_12_31_17_54_56_108.jpg"> </td> 
    <td><img src="assets/right_2017_12_31_17_54_56_108.jpg"> </td>
  </tr>
 
</table>

<h3> Augmented the number of images </h3>

Two ways have been tried out:

<li> adding new images by flipping each image horizontally using the following code:
<pre>

"""
#Augment the number of images by getting a flip of each image
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(-1*measurement)
"""
</pre>

<li> adding the image from the left and right cameras and associating angles to them </li>
<pre>
    image = cv2.imread(current_path)
    images.append(image)
    images.append(cv2.imread(line[1]))
    images.append(cv2.imread(line[2]))
    
</pre>

associating angles:
 <pre>
  # extracting the steering wheel as labels
    #print ("steering angle", line[3])
    measurement = float(line[3])
    
    #print ("measurement", measurement)
    measurements.append(measurement)
    measurements.append(measurement+0.275)
    measurements.append(measurement-0.275)
    
  </pre>  
    
<h3> Fine Tuning the Model </h3>
   
   <li>I chose epochs = 7.</li>
   <li>the angles: </li> <br> the angles created frtom the left and right camera the value that allowed the car to stay on the road was +/- 0.275
   With 0.27 the car was slightly driving over the side after the bridge
   
<h3> Error Loss </h3>

<img src="loss_graph_010518.png"> The error loss shows that after 5 epochs the error is increasing, I could have reduced the number of epochs to 5 instead of 7



