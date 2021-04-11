---------------------------------------------------
 -- Office Hydration Monitoring (OHM) Dataset -- 
 	Version 1.0 				  
 	Date 04/2021				  
---------------------------------------------------



This dataset is a public collection of labelled data for classifying office workers' hydration patterns. 

It contains 1000 instances performed by 10 subjects and includes 25 variations of different interactions that could be made with with liquid containers. 
Each of the 25 variations was repeated 4 times for each one of the volunteers.

Those interactions (listed in Table 1) are grouped into three main classes: 

(1) drinking from a bottle
(2) drinking from a cup 
(3) other kinds of interactions)


This dataset was created with the idea of having a non-controlled activity dataset that resemble real-world scenarios. 
Therefore, the interaction to be recorded was intentionally described very vaguely to the volunteer and no detailed instructions were given to guide their movements. 
Moreover, each of them had their own containers (no instructions were given apart from the bottle and mug categories).




The dataset provides labelled recorded executions of different variations of the three main classes: 

Table 1

Container	Label	Posture		Description				        Number
-----------------------------------------------------------------------------------------------
			Sit		Grab glass, drink, leave glass			1
					Grab glass, sip, leave glass			2
			-----------------------------------------------------------------------
					Grab glass, drink, leave glass			3
		Drink	Stand-up	Grab glass, drink, walk				4
					Grab glass, sip, leave glass			5
			-----------------------------------------------------------------------
			Walk		Drink						6
		-------------------------------------------------------------------------------
					Grab and leave the glass			7
"Glass/Mug"				Raise and lower 				8
			Sit		Move the glass (from one point to another)	9
					Grab the glass and stand up			10
		Other			Inspect the glass				11
					Shake the glass					12
			-----------------------------------------------------------------------
			Stand-up	Grab the glass and walk				13
					Sit and leave the glass				14
			-----------------------------------------------------------------------
			Walk		Walk						15
					Walk and leave the glass			16
-----------------------------------------------------------------------------------------------
			Sit		Grab bottle, drink, leave bottle		17
					Grab bottle, open, drink, leave			18
			-----------------------------------------------------------------------
					Grab bottle, drink, leave bottle		19
		Drink	Stand-up	Grab bottle, open, drink, leave bottle		20
Bottle					Garb bottle, open, drink, walk			21
			-----------------------------------------------------------------------
			Walk		Open bottle and drink				22
		-------------------------------------------------------------------------------
		Other			Grab the bottle and close it			23
			Sit		Pour the bottle of water into sth		24
					Inspect the botlle				25
-----------------------------------------------------------------------------------------------
		


1. Files tree
-----------------

ROOT
- Drink_bottle (240 elements)
- Drink_glass (240 elements)
- Other (520 elements)
- README.TXT



2. Files and included data
--------------------------

Each txt file contains one recorded trial and includes the acceleration (m/s^2), rotation speed (rad/s), and rotation angles for X, Y and Z.

Each file contains the following columns separated by comma:

Acc_x, Acc_y, Acc_z, Gyro_x, Gyro_y, Gyro_z, Pitch, Roll, Yaw


The name of the file has the following format: 

[Label]_[Subject]_[Container]_[Posture]_[Number]_[Date]_[Time].txt

- [Label]: 	name of the activity performed: Drink or Other
- [Subject]:	the volunteer performing the recorded motion in the format XN
		where X (F -> female, M -> male)
		and N is the number is associated to the volunteer
- [Container]:	Type of container: Glass/Mug or Bottle
- [Posture]:    Initial state when recording the movement
		Sitting (sit),  Standing (standup), Walking (Walk)
- [Number]: 	The number (from 1 to 25) corresponding to the subcathegory 
		of the motion according to Table 1.
- [Date]: 	Date of the recording
		Format : Day-Month-Year
- [Time]: 	Timestamp of the starting moment of the recording
		Format : Hour-Minute-Second




3. Hardware Setup and Sensor location 
--------------------------------------------

Data was captured with a MPU6886 6-axis IMU sensor, with 3-axis gravity accelerometer and 3-axis gyroscope. 
Data transmission to the PC was via Bluetooth.

The placement of the sensor, with respect to the glass, mug or bottle, was not fixed. Only the component of the signal perpendicular to the plane (y) pointed the same direction in every case. 
(i.e., volunteers could rotate the water container with the sensor attached, and the initial orientation was not fixed)
Thus, this induces a high variance on the recorded data, as the reference system for the accelerometer and gyroscope signals can vary. 


Type:		   Tri-axial accelerometer and gyroscope
Measurement range: Acc: [±8g] 	
		   Gyro: [±2000 DPS]
Output data rate:  50 Hz
Location:	   attached arround the the container
		   - x axis: pointing in any direction parallel to the plante of the glass/mug/bottle
		   - y axis: perpendicular to the plane of the glass/mug/bottle section
		   - z axis: pointing in any direction parallel to the plante of the glass/mug/bottle

A post processing stage was carried out to filter the signal and remove the stacionary state of the recording (i.e., when the container is on the table)





4 Volunteers
--------------

The dataset is composed of the recordings of a total of 10 volunteers

Basic information about them is included in Table 2.

Table 2:
| Gender|       Age        | 
| F | M | Min | Max | Avg  | 
|-------|------------------|
| 4 | 6 | 25  | 37  | 29   | 


