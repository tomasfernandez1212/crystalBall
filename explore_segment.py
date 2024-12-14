import tensorflow as tf
import matplotlib.pyplot as plt
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils

def visualize_frame_images(frame):
    for camera_image in frame.images:
        img = tf.image.decode_jpeg(camera_image.image).numpy()
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.title(f"Camera: {dataset_pb2.CameraName.Name.Name(camera_image.name)}")
        plt.axis('off')
        
        # Display camera pose (position)
        camera_pose = camera_image.pose
        position = camera_pose.position
        plt.text(10, 30, f"Position: ({position.x:.2f}, {position.y:.2f}, {position.z:.2f})", 
                 color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
        
        plt.show()

# Load a TFRecord file
dataset = tf.data.TFRecordDataset('data/waymo_open_dataset_v_1_4_3/individual_files/training/segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord')

# Get the first frame
for data in dataset:
    frame = dataset_pb2.Frame()
    frame.ParseFromString(data.numpy())
    visualize_frame_images(frame)
    break  # Only process the first frame
