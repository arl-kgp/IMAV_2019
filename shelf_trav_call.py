from shelf_trav import FrontEnd as shelf_traversal
from djitellopy import Tello
import cv2

tello = Tello()
tello.connect()
tello.streamoff()
tello.streamon()
trav1 = shelf_traversal(tello)
frame_read = trav1.tello.get_frame_read()

should_stop = False

while not should_stop:

	frame = frame_read.frame
	print(trav1.tello.get_bat())

	if frame_read.stopped:
		frame_read.stop()
		break

	cv2.imshow("original",frame)

	key = cv2.waitKey(1) & 0xFF;
	if (key == ord("m")):
		trav1.run(frame)
		if trav1.num_text_frames == 1:
			should_stop = True
			print("Finished")
			trav1.tello.land()
	else:
		trav1.manualRcControl(key)

	trav1.sendRcControl()


tello.streamoff()
tello.end()