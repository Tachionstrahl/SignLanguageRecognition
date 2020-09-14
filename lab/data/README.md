# Description
Folders inside this folder contain variants of CSV files of extracted landmarks from videos.
Here the description of each folder:

## ./absolute
Contains variants of extracted landmarks with absolute coordinates.
This means, the x, y and z which MediaPipe's Hand Tracking and Face Tracking output, are serialized.

### ./2D
x and y coordinates of landmarks.

### ./2D_reduced
We reduced number of landmarks from 21 to 12.

### ./2D_unknown
Contains CSV files from unseen examples. Only one video per class/word. Extracted from the SignDict.org videos.

### ./3D
x, y and z coordinates of landmarks.

### ./3D_pose
x, y and z coordinates of MediaPipe's PoseTracking.

## ./relative
Contains variants of extracted landmarks with relative coordinates.
This means, the delta between x,y,z of a frame and its next are serialized.
-1 means the movement is backwards along the axis.
0 means no movement.
1 means the forwards along the axis.

### ./2D
x and y deltas of landmarks.


