from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from launch import LaunchDescription

_PACKAGE_SHARE = Path(get_package_share_directory("pothole_rover"))
_DEFAULT_WEIGHTS = next(
    (
        str(_PACKAGE_SHARE / name)
        for name in ("best.pt", "new_model.pt")
        if (_PACKAGE_SHARE / name).is_file()
    ),
    str(_PACKAGE_SHARE / "new_model.pt"),
)


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("image_type", default_value="image"),
            DeclareLaunchArgument(
                "detections_topic", default_value="/pothole/detections"
            ),
            DeclareLaunchArgument(
                "annotated_topic", default_value="/pothole/annotated"
            ),
            DeclareLaunchArgument("target_fps", default_value="5.0"),
            DeclareLaunchArgument("confidence", default_value="0.25"),
            DeclareLaunchArgument("calibration_file", default_value=""),
            DeclareLaunchArgument("weights", default_value=_DEFAULT_WEIGHTS),
            DeclareLaunchArgument("device", default_value=""),
            DeclareLaunchArgument("dashboard_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("dashboard_port", default_value="8080"),
            Node(
                package="pothole_rover",
                executable="pothole_dashboard",
                name="pothole_dashboard",
                output="screen",
                parameters=[
                    {
                        "image_topic": LaunchConfiguration("image_topic"),
                        "image_type": LaunchConfiguration("image_type"),
                        "detections_topic": LaunchConfiguration("detections_topic"),
                        "annotated_topic": LaunchConfiguration("annotated_topic"),
                        "target_fps": ParameterValue(
                            LaunchConfiguration("target_fps"), value_type=float
                        ),
                        "confidence": ParameterValue(
                            LaunchConfiguration("confidence"), value_type=float
                        ),
                        "calibration_file": LaunchConfiguration("calibration_file"),
                        "weights": LaunchConfiguration("weights"),
                        "device": LaunchConfiguration("device"),
                        "dashboard_host": LaunchConfiguration("dashboard_host"),
                        "dashboard_port": ParameterValue(
                            LaunchConfiguration("dashboard_port"), value_type=int
                        ),
                    }
                ],
            ),
        ]
    )
