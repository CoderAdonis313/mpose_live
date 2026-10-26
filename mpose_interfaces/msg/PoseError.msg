# Time at which the estimated pose was evaluated.
# frame_id identifies the arena marker frame.
std_msgs/Header header

# Marker whose pose was evaluated.
string bot_label

# Euclidean translation error in metres.
float64 translation_error_m

# Geodesic rotation error in degrees.
float64 rotation_error_deg

# Quaternion rotation error 
float64 rotation_error_quat

# Combined translation/quaternion loss.
float64 pose_loss

# beta balance factor
float64 beta