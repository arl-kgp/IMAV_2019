# IMAV_2019

The repository is the complete compilation of all the resources for automated and manual warehouse management using DJI Tello micro-drone.

Now, let's get started on explaining the constituents of the repository.

###First, lets start with the sub-repo "imav-warehouse-management"

It is the solution if the person wants to use the manual mode, i.e. wants to know about only a small/specified region of the inventory then can perform it manually. It is possible to connect a joystick to the laptop simply and fly the drone, meanwhile, do the necessary work too.

The main work going to be explained here is detecting the QR-codes, alphanumeric codes and the algorithm for associating them.

###Second comes to the sub-repo "NEW_FINAL_NO_TRACK"

The algorithm for detecting the QR-codes and alphanumeric code is similar to as the previous sub-repo. But what is different is, this is an automated approach towards solving the problem statement.

The localisation in the dynamic GPS denied Environment is the most significant thing in the approach. Here, we localised our drone concerning the arena. The number of alphanumeric codes between the path was used to locate the drone i.e. we create a mesh of the arena in the vertical direction.


Considering the mesh something like this, we created a 2D matrix to mark the position of the drone by counting the number of alphanumeric codes passed. Further, it was ensured that the field of view of the drone was limited, such that no two alphanumeric codes were visible at a single time.
We align our drone in front of the alphanumeric with alphanumeric code at the bottom of the image. Then the QR codes in that block were assigned to the alphanumeric code. Further, the distance from the shelf was calculated to ensure that only one block was visible as the constraint being that the width of the shelf is known.

Further, we adopted the searching strategy of searching from left to right and then going down, searching from right to left. Through this manner, the least time for searching was seen. 

There are various subcodes in the folder, but all of them are linked via the "main_final.py" file. 

Here comes a brief on the sub-files:
1) "warehouse" folder consists of a complete warehouse solution when we are aligned to the top left shelf. 
2) "tello" folder consists of some subsidiary files which might not be needed right now.
3) "start_to_next" folder consists of the codebase to reach to the starting point from where warehouse code could start from the takeoff point.
4) "tello_height.py" is the function to take the drone to a particular height with respect to the takeoff point.
5) "orient_yaw.py" is the function to orient the drone at a particular yaw angle.
6) "final_csv_final.py" contains some final alterations to be done to the CSV file. (might not be of use to you)
7) "Joystick_Controller" folder consists of the code to manually take over the using Joystick incase of an emergency.


###Finally comes to the sub-repo "NEW_FINAL"

This sub-repo closely resembles the previous folder but with a minimal change.
Previously, when we aligned to a particular rectangle, we used to give the drone a specific command to go a fixed amount of distance. However, it often worked, but sometimes it wasn't such a right solution. 
Thus, we modified to track the detected rectangle and report we had moved ahead of it only when we lost its tracking. Such that we were accurately able to count all the boxes.






This work was done as a part of the indoor competition at IMAV'19.
We are proud to announce that we have won the competition with this codebase.

