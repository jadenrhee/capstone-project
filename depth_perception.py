import cv2
import pyzed.sl as sl
import time

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error {err}, exiting program.")
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.enable_fill_mode = True  # Enable fill mode

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_configuration.resolution
    image_size.width = image_size.width // 2
    image_size.height = image_size.height // 2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()

    key = ''
    while key != 113:  # # for key code of 'q'
       # Grab an image
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            # Retrieve the depth map
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

            # Get the depth value at the center of the image
            center_x = image_size.width // 2
            center_y = image_size.height // 2
            err, point_cloud_value = point_cloud.get_value(center_x, center_y)
            if err == sl.ERROR_CODE.SUCCESS:
                depth_value = point_cloud_value[2]  # Z value is the depth
                print(f"Depth at center ({center_x}, {center_y}): {depth_value:.2f} meters")

            # Display the images
            # Draw a dot at the center of the image
            image_with_dot = image_zed.get_data().copy()
            cv2.circle(image_with_dot, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", image_with_dot)
            #cv2.imshow("Depth", depth_image_zed.get_data())

        key = cv2.waitKey(10) # Wait 10 ms

    # Close the camera
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
