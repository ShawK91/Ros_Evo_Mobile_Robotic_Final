#!/usr/bin/env python
import rospy, numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseActionGoal
#from geometry_msgs.msg import PoseStamped 
from nav_msgs.msg import Odometry
import random
import sys

tag = "\n".join(sys.argv)
tag = str(tag[-1])
print tag
filename_x = 'x_' + tag + '.csv' 
filename_y = 'y_' + tag + '.csv' 
print filename_x, filename_y

class Data_gatherer:
	def __init__(self):
		self.train_x = np.zeros((1,32))
		self.train_y = np.zeros((1,3))
		self.is_sync = False




	
	
	def update_command(self, cmd):
	    self.train_y = np.concatenate((self.train_y, np.array([[cmd.linear.x, cmd.linear.y, cmd.angular.z]])))
	    self.is_sync = True
	

	def update_scan(self, scan):
		if self.is_sync:
			scan_readings = scan.ranges
			scan_readings = np.array(scan_readings)
			scan_readings = scan_readings[::20]
			scan_readings = np.reshape(scan_readings, (1, len(scan_readings)))
			self.train_x = np.concatenate((self.train_x, scan_readings))
			self.is_sync = False
	   




	  


	  

if __name__ == "__main__":
	handle = Data_gatherer()
	rospy.init_node('3_deep_data_collector', anonymous=True) #Node
	scan_sub = rospy.Subscriber('/scan', LaserScan, handle.update_scan)
	cmd_sub = rospy.Subscriber('/navigation_velocity_smoother/raw_cmd_vel', Twist, handle.update_command)

	odom_pub = rospy.Publisher('/base_pose_ground_truth', Odometry, queue_size=1)
	goal_pub = rospy.Publisher('/move_base/goal',MoveBaseActionGoal, queue_size=1)

	
	while not rospy.is_shutdown():
		rospy.spin()
		
	print handle.train_x.shape, handle.train_y.shape
	np.savetxt(filename_x, handle.train_x, delimiter = ',')  
	np.savetxt(filename_y, handle.train_y, delimiter = ',')   	

    





	#def reset_odom(self, odom_sub):
		#print 'RESET'
	 	#odom = Odometry()
		#odom.pose.pose.position.x = 5.0
		#odom.pose.pose.position.y = 4.0
		#odom.pose.pose.orientation.w = 1.0
		#odom_sub.publish(odom)

