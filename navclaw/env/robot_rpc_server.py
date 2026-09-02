from __future__ import annotations

import argparse
import math
import time
from threading import Event
from threading import Lock
from threading import Thread
from typing import Any

import cv2
import numpy as np
from cv_bridge import CvBridge
from navclaw.env.depth_filter import filter_depth_edges
from flask import Flask
from flask import jsonify
from flask import request
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TwistStamped
from navclaw.env.rpc_protocol import encode_array
from navclaw.env.rpc_protocol import encode_depth_meters
from navclaw.env.rpc_protocol import encode_rgb_image
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import JointState
from tf2_ros import Buffer
from tf2_ros import TransformListener


TURN_ANGLE_DEGREES = 90.0
TF_LOOKUP_TIMEOUT_SEC = 1.0
TF_READY_TIMEOUT_SEC = 20.0
POLL_STEP_SEC = 0.1
TURN_YAW_TOLERANCE_DEGREES = 2.0
TURN_STABLE_TIME_SEC = 0.15
TURN_MAX_ANGULAR_SPEED_RAD_PER_SEC = 0.3
TURN_MIN_ANGULAR_SPEED_RAD_PER_SEC = 0.0
TURN_CONTROLLER_KP = 0.5
TURN_TIMEOUT_SEC = 8.0
NAV_SERVER_WAIT_TIMEOUT_SEC = 10.0
NAV_GOAL_RESPONSE_TIMEOUT_SEC = 10.0
NAV_CANCEL_TIMEOUT_SEC = 2.0
OBS_SYNC_QUEUE_SIZE = 30
OBS_SYNC_SLOP_SEC = 0.03
ODOM_TOPIC = "/state_estimation"
ODOMETRY_POSE_SEMANTIC_FRAME = "imu_link"
IMU_TO_BASE_TRANSLATION = np.array([-0.247, -0.28529, -0.21488], dtype=np.float32)
IMU_TO_BASE_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class NavigationStoppedException(Exception):
    pass


def _yaw_degrees_from_rotation(rotation: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))))


def _transform_to_matrix(transform) -> np.ndarray:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    quat = np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64)
    rot_matrix = R.from_quat(quat).as_matrix()

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rot_matrix.astype(np.float32)
    matrix[:3, 3] = np.array([translation.x, translation.y, translation.z], dtype=np.float32)
    return matrix


def _dict_to_quaternion(q_dict: dict[str, float]) -> Quaternion:
    quaternion = Quaternion()
    quaternion.x = float(q_dict["x"])
    quaternion.y = float(q_dict["y"])
    quaternion.z = float(q_dict["z"])
    quaternion.w = float(q_dict["w"])
    return quaternion


def _pose_dict_from_xy_yaw(x: float, y: float, z: float, yaw: float) -> dict[str, Any]:
    half = math.radians(float(yaw)) / 2.0
    return {
        "position": {
            "x": float(x),
            "y": float(y),
            "z": float(z),
        },
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": float(math.sin(half)),
            "w": float(math.cos(half)),
        },
    }


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _pose_to_matrix(position, orientation) -> np.ndarray:
    quat = np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float64)
    rot_matrix = R.from_quat(quat).as_matrix()
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rot_matrix.astype(np.float32)
    matrix[:3, 3] = np.array([position.x, position.y, position.z], dtype=np.float32)
    return matrix


def _translation_quaternion_to_matrix(
    translation: np.ndarray,
    quaternion: np.ndarray,
) -> np.ndarray:
    rot_matrix = R.from_quat(np.asarray(quaternion, dtype=np.float64)).as_matrix()
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rot_matrix.astype(np.float32)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return matrix


def _wrap_angle_radians(angle: float) -> float:
    return float(math.atan2(math.sin(float(angle)), math.cos(float(angle))))


class NodeRunner:
    def __init__(self, node: Node, thread_name: str) -> None:
        self.node = node
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = Thread(target=self.executor.spin, name=thread_name, daemon=True)
        self.thread.start()

    def shutdown(self) -> None:
        self.executor.shutdown()
        self.thread.join(timeout=2.0)


class ObservationNode(Node):
    def __init__(self) -> None:
        super().__init__("navclaw_observation_node")
        self.camera_frame = "camera_head_left_link"
        self.base_frame = "base_link"
        self.odom_frame = "map"

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._lock = Lock()
        self._latest_observation_bundle: dict[str, Any] | None = None
        self._fixed_T_camera_base: np.ndarray | None = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.depth_info_sub = Subscriber(
            self,
            CameraInfo,
            "/hdas/camera_head/depth/camera_info",
            qos_profile=sensor_qos,
        )
        self.rgb_sub = Subscriber(
            self,
            ImageMsg,
            "/hdas/camera_head/rgb/image_rect_color",
            qos_profile=sensor_qos,
        )
        self.depth_sub = Subscriber(
            self,
            ImageMsg,
            "/hdas/camera_head/depth/depth_registered",
            qos_profile=sensor_qos,
        )
        self.odom_sub = Subscriber(
            self,
            Odometry,
            ODOM_TOPIC,
            qos_profile=sensor_qos,
        )
        self.obs_sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.depth_info_sub, self.odom_sub],
            queue_size=OBS_SYNC_QUEUE_SIZE,
            slop=OBS_SYNC_SLOP_SEC,
        )
        self.obs_sync.registerCallback(self.synced_observation_callback)

    def synced_observation_callback(
        self,
        rgb_msg: ImageMsg,
        depth_msg: ImageMsg,
        depth_info_msg: CameraInfo,
        odom_msg: Odometry,
    ) -> None:
        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if depth_msg.encoding == "32FC1":
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1").astype(np.float32)
        elif depth_msg.encoding == "16UC1":
            depth_u16 = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth = depth_u16.astype(np.float32) * 0.001
        else:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth, _depth_filter_elapsed_sec = filter_depth_edges(depth)
        bundle = {
            "stamp": depth_msg.header.stamp,
            "stamp_sec": _stamp_seconds(depth_msg.header.stamp),
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "depth": depth,
            "intrinsic": np.array(depth_info_msg.k, dtype=np.float32).reshape(3, 3),
            "odom_msg": odom_msg,
        }
        with self._lock:
            self._latest_observation_bundle = bundle

    def wait_first_obs(self, timeout_sec: float = 10.0) -> None:
        start_time = time.time()
        while rclpy.ok():
            with self._lock:
                obs_ready = self._latest_observation_bundle is not None
            camera_ready = self._fixed_T_camera_base is not None or bool(
                self.tf_buffer.can_transform(
                    self.camera_frame,
                    self.base_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )
            )
            if bool(obs_ready) and bool(camera_ready):
                self._ensure_fixed_camera_transform()
                return
            if time.time() - start_time > float(timeout_sec):
                raise RuntimeError(
                    "Timeout waiting for first synchronized observation and fixed camera extrinsic: "
                    f"camera_frame={self.camera_frame}, "
                    f"odom_frame={self.odom_frame}, "
                    f"base_frame={self.base_frame}"
                )
            time.sleep(POLL_STEP_SEC)

    def _lookup_transform_latest(self, target_frame: str, source_frame: str) -> Any:
        start_time = time.time()
        while rclpy.ok():
            ready = bool(
                self.tf_buffer.can_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )
            )
            if ready:
                return self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_SEC),
                )
            if time.time() - start_time > TF_READY_TIMEOUT_SEC:
                raise RuntimeError(
                    "Timeout waiting for latest transform: "
                    f"target_frame={target_frame}, source_frame={source_frame}"
                )
            time.sleep(POLL_STEP_SEC)
        raise RuntimeError("ROS shutdown while waiting for latest transform")

    def _ensure_fixed_camera_transform(self) -> np.ndarray:
        if self._fixed_T_camera_base is None:
            self._fixed_T_camera_base = _transform_to_matrix(
                self._lookup_transform_latest(
                    self.camera_frame,
                    self.base_frame,
                )
            )
        return self._fixed_T_camera_base

    def clear_fixed_camera_transform(self) -> None:
        self._fixed_T_camera_base = None

    def get_obs_snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._latest_observation_bundle is None:
                raise RuntimeError("Synchronized observation not ready")
            bundle = self._latest_observation_bundle
            rgb = np.copy(np.asarray(bundle["rgb"], dtype=np.uint8))
            depth = np.copy(np.asarray(bundle["depth"], dtype=np.float32))
            intrinsic = np.copy(np.asarray(bundle["intrinsic"], dtype=np.float32))
            odom_msg = bundle["odom_msg"]

        if str(odom_msg.header.frame_id) != self.odom_frame:
            raise ValueError(f"Expected odometry frame_id={self.odom_frame}, got {odom_msg.header.frame_id}")

        T_odom_imu = _pose_to_matrix(odom_msg.pose.pose.position, odom_msg.pose.pose.orientation)
        T_imu_base = _translation_quaternion_to_matrix(
            translation=IMU_TO_BASE_TRANSLATION,
            quaternion=IMU_TO_BASE_QUATERNION,
        )
        T_odom_base = T_odom_imu @ T_imu_base
        T_base_odom = np.linalg.inv(T_odom_base)
        T_cam_base = self._ensure_fixed_camera_transform()
        T_cam_odom = T_cam_base @ T_base_odom
        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsic": intrinsic,
            "T_cam_odom": T_cam_odom,
            "T_odom_base": T_odom_base,
        }

    def get_latest_base_pose(self) -> dict[str, float]:
        base_to_odom = self._lookup_transform_latest(
            self.odom_frame,
            self.base_frame,
        )
        T_odom_base = _transform_to_matrix(base_to_odom)
        rotation = T_odom_base[:3, :3]
        return {
            "x": float(T_odom_base[0, 3]),
            "y": float(T_odom_base[1, 3]),
            "z": float(T_odom_base[2, 3]),
            "yaw": _yaw_degrees_from_rotation(rotation),
        }


class NavigationNode(Node):
    def __init__(self) -> None:
        super().__init__("navclaw_navigation_node")

        pub_qos = QoSProfile(depth=10)
        self.torso_joint_state_pub = self.create_publisher(
            JointState,
            "/motion_target/target_joint_state_torso",
            pub_qos,
        )
        self.right_joint_state_pub = self.create_publisher(
            JointState,
            "/motion_target/target_joint_state_arm_right",
            pub_qos,
        )
        self.left_joint_state_pub = self.create_publisher(
            JointState,
            "/motion_target/target_joint_state_arm_left",
            pub_qos,
        )
        self.chassis_speed_pub = self.create_publisher(
            TwistStamped,
            "/motion_target/target_speed_chassis",
            pub_qos,
        )
        self.navigate_to_pose_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        self._stop_event = Event()
        self._current_nav_goal_lock = Lock()
        self._current_nav_goal_handle = None

    def pre_nav(self) -> None:
        torso = JointState()
        torso.position = [1.18, -2.10, -0.9, -0.1]
        self.torso_joint_state_pub.publish(torso)
        time.sleep(1.0)

        right = JointState()
        right.position = [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
        self.right_joint_state_pub.publish(right)
        time.sleep(1.0)

        left = JointState()
        left.position = [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
        self.left_joint_state_pub.publish(left)
        time.sleep(1.0)

    def send_zero_velocity(self) -> None:
        twist = self._build_chassis_twist(angular_z=0.0)
        for _ in range(5):
            self.chassis_speed_pub.publish(twist)
            time.sleep(0.02)

    def _build_chassis_twist(self, angular_z: float) -> TwistStamped:
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = "base_link"
        twist.twist.linear.x = 0.0
        twist.twist.linear.y = 0.0
        twist.twist.linear.z = 0.0
        twist.twist.angular.x = 0.0
        twist.twist.angular.y = 0.0
        twist.twist.angular.z = float(angular_z)
        return twist

    def publish_angular_velocity(self, angular_z: float) -> None:
        self.chassis_speed_pub.publish(self._build_chassis_twist(angular_z=angular_z))

    def dict_to_pose_stamped(self, pose_dict: dict[str, Any], frame_id: str = "odom") -> PoseStamped:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame_id
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = float(pose_dict["position"]["x"])
        pose_stamped.pose.position.y = float(pose_dict["position"]["y"])
        pose_stamped.pose.position.z = float(pose_dict["position"]["z"])
        pose_stamped.pose.orientation = _dict_to_quaternion(pose_dict["orientation"])
        return pose_stamped

    def clear_stop(self) -> None:
        self._stop_event.clear()

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop_requested(self) -> bool:
        return bool(self._stop_event.is_set())

    def _start_action(self, action_client, action_name: str, goal) -> tuple[Any, Any]:
        if not action_client.wait_for_server(timeout_sec=NAV_SERVER_WAIT_TIMEOUT_SEC):
            raise RuntimeError(f"{action_name} action server not available")
        send_goal_future = action_client.send_goal_async(goal)
        start_time = time.time()
        while rclpy.ok():
            if send_goal_future.done():
                break
            if time.time() - start_time > NAV_GOAL_RESPONSE_TIMEOUT_SEC:
                raise RuntimeError(f"Timed out waiting for {action_name} goal response")
            time.sleep(POLL_STEP_SEC)

        goal_handle = send_goal_future.result()
        if goal_handle is None:
            raise RuntimeError(f"{action_name} goal handle missing")
        if not goal_handle.accepted:
            raise RuntimeError(f"{action_name} goal rejected")

        with self._current_nav_goal_lock:
            self._current_nav_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        return goal_handle, result_future

    def start_navigation(self, pose_stamped: PoseStamped) -> tuple[Any, Any]:
        goal = NavigateToPose.Goal()
        goal.pose = pose_stamped
        goal.behavior_tree = ""
        return self._start_action(
            action_client=self.navigate_to_pose_client,
            action_name="NavigateToPose",
            goal=goal,
        )

    def cancel_goal(self, goal_handle, timeout_sec: float = NAV_CANCEL_TIMEOUT_SEC) -> None:
        cancel_future = goal_handle.cancel_goal_async()
        start_time = time.time()
        while rclpy.ok():
            if cancel_future.done():
                return
            if time.time() - start_time > float(timeout_sec):
                raise RuntimeError("Timed out waiting for Nav2 action cancellation")
            time.sleep(POLL_STEP_SEC)

    def clear_current_goal(self, goal_handle=None) -> None:
        with self._current_nav_goal_lock:
            if goal_handle is None or self._current_nav_goal_handle is goal_handle:
                self._current_nav_goal_handle = None


class RobotRPCController:
    def __init__(
        self,
        goal: str | None,
        observation_settle_time_sec: float,
        step_observation_interval_sec: float,
        pre_nav: bool,
    ) -> None:
        self.goal = "" if goal is None else str(goal)
        self.observation_settle_time_sec = float(observation_settle_time_sec)
        self.step_observation_interval_sec = float(step_observation_interval_sec)
        self._action_step_total = 0
        if self.step_observation_interval_sec <= 0.0:
            raise ValueError("step_observation_interval_sec must be positive")
        if not rclpy.ok():
            rclpy.init()

        self.observation_node = ObservationNode()
        self.navigation_node = NavigationNode()
        self.observation_runner = NodeRunner(self.observation_node, thread_name="navclaw_observation_spin")
        self.navigation_runner = NodeRunner(self.navigation_node, thread_name="navclaw_navigation_spin")

        self.observation_node.wait_first_obs(timeout_sec=10.0)
        if bool(pre_nav):
            self.navigation_node.pre_nav()
            self.observation_node.clear_fixed_camera_transform()
            self.observation_node.wait_first_obs(timeout_sec=10.0)

    def shutdown(self) -> None:
        self.navigation_runner.shutdown()
        self.observation_runner.shutdown()
        self.navigation_node.destroy_node()
        self.observation_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def _pump_observations(self) -> None:
        if self.observation_settle_time_sec <= 0.0:
            return
        time.sleep(self.observation_settle_time_sec)

    def _snapshot(self, pump: bool = True) -> dict[str, Any]:
        if pump:
            self._pump_observations()
        return self.observation_node.get_obs_snapshot()

    def _pose_from_snapshot(self, snapshot: dict[str, Any]) -> dict[str, float]:
        T_odom_base = np.asarray(snapshot["T_odom_base"], dtype=np.float32)
        rotation = T_odom_base[:3, :3]
        return {
            "x": float(T_odom_base[0, 3]),
            "y": float(T_odom_base[1, 3]),
            "z": float(T_odom_base[2, 3]),
            "yaw": _yaw_degrees_from_rotation(rotation),
        }

    @staticmethod
    def _empty_observation_payload_timing() -> dict[str, Any]:
        return {
            "elapsed_seconds": 0.0,
            "pose_seconds": 0.0,
            "rgb_encode_seconds": 0.0,
            "depth_encode_seconds": 0.0,
            "intrinsics_encode_seconds": 0.0,
            "T_cam_odom_encode_seconds": 0.0,
            "T_odom_base_encode_seconds": 0.0,
            "payload_approx_bytes": 0,
        }

    @classmethod
    def _accumulate_observation_payload_timing(
        cls,
        totals: dict[str, Any],
        timing: dict[str, Any],
    ) -> None:
        for key in cls._empty_observation_payload_timing():
            if key == "payload_approx_bytes":
                totals[key] = int(totals.get(key, 0)) + int(timing.get(key, 0))
            else:
                totals[key] = float(totals.get(key, 0.0)) + float(timing.get(key, 0.0))

    def _observation_payload(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        payload, _timing = self._timed_observation_payload(snapshot)
        return payload

    def _timed_observation_payload(self, snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        total_start = time.perf_counter()

        pose_start = time.perf_counter()
        pose = self._pose_from_snapshot(snapshot)
        pose_seconds = time.perf_counter() - pose_start

        rgb_start = time.perf_counter()
        rgb_payload = encode_rgb_image(np.asarray(snapshot["rgb"], dtype=np.uint8))
        rgb_encode_seconds = time.perf_counter() - rgb_start

        depth_start = time.perf_counter()
        depth_payload = encode_depth_meters(np.asarray(snapshot["depth"], dtype=np.float32))
        depth_encode_seconds = time.perf_counter() - depth_start

        intrinsics_start = time.perf_counter()
        intrinsics_payload = encode_array(np.asarray(snapshot["intrinsic"], dtype=np.float32))
        intrinsics_encode_seconds = time.perf_counter() - intrinsics_start

        t_cam_odom_start = time.perf_counter()
        t_cam_odom_payload = encode_array(np.asarray(snapshot["T_cam_odom"], dtype=np.float32))
        t_cam_odom_encode_seconds = time.perf_counter() - t_cam_odom_start

        t_odom_base_start = time.perf_counter()
        t_odom_base_payload = encode_array(np.asarray(snapshot["T_odom_base"], dtype=np.float32))
        t_odom_base_encode_seconds = time.perf_counter() - t_odom_base_start

        return {
            "pose": pose,
            "rgb": rgb_payload,
            "depth": depth_payload,
            "intrinsics": intrinsics_payload,
            "T_cam_odom": t_cam_odom_payload,
            "T_odom_base": t_odom_base_payload,
            "text_hint": "",
            "visible_objects": [],
            "goal": self.goal,
        }, {
            "elapsed_seconds": float(time.perf_counter() - total_start),
            "pose_seconds": float(pose_seconds),
            "rgb_encode_seconds": float(rgb_encode_seconds),
            "depth_encode_seconds": float(depth_encode_seconds),
            "intrinsics_encode_seconds": float(intrinsics_encode_seconds),
            "T_cam_odom_encode_seconds": float(t_cam_odom_encode_seconds),
            "T_odom_base_encode_seconds": float(t_odom_base_encode_seconds),
            "payload_approx_bytes": int(
                len(rgb_payload)
                + len(str(depth_payload.get("data", "")))
                + len(str(intrinsics_payload.get("data", "")))
                + len(str(t_cam_odom_payload.get("data", "")))
                + len(str(t_odom_base_payload.get("data", "")))
            ),
        }

    def _episode_metadata_payload(self) -> dict[str, Any]:
        return {
            "episode_id": 0,
            "dataset_index": None,
            "scene_id": "robot",
            "scene_name": "robot",
            "goal": self.goal,
            "goals": [],
            "ground_height_offset": 0.0,
            "episode_over": False,
            "pointnav_step_total": int(self._action_step_total),
        }

    def _goal_pose_stamped(self, x: float, y: float, z: float, yaw: float) -> PoseStamped:
        pose_dict = _pose_dict_from_xy_yaw(x=x, y=y, z=z, yaw=yaw)
        return self.navigation_node.dict_to_pose_stamped(
            pose_dict=pose_dict,
            frame_id=self.observation_node.odom_frame,
        )

    def _run_nav_action(
        self,
        start_action,
        sample_intermediate_observations: bool,
        action_name: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        total_start = time.perf_counter()
        self.navigation_node.clear_stop()
        step_payloads: list[dict[str, Any]] = []
        timing: dict[str, Any] = {
            "sample_intermediate_observations": bool(sample_intermediate_observations),
            "start_action_seconds": 0.0,
            "navigation_result_wait_seconds": 0.0,
            "intermediate_snapshot_seconds": 0.0,
            "intermediate_payload_encode": self._empty_observation_payload_timing(),
            "sampled_intermediate_observation_count": 0,
            "returned_intermediate_observation_count": 0,
            "final_snapshot_seconds": 0.0,
        }
        start_action_start = time.perf_counter()
        goal_handle, result_future = start_action()
        timing["start_action_seconds"] = float(time.perf_counter() - start_action_start)
        next_sample_time = time.time() + self.step_observation_interval_sec
        wait_start = time.perf_counter()

        try:
            while not result_future.done():
                if self.navigation_node.stop_requested():
                    self.navigation_node.cancel_goal(goal_handle)
                    self.navigation_node.send_zero_velocity()
                    raise NavigationStoppedException("Navigation stopped")
                now = time.time()
                if sample_intermediate_observations and now >= next_sample_time:
                    snapshot_start = time.perf_counter()
                    snapshot = self._snapshot(pump=False)
                    timing["intermediate_snapshot_seconds"] = float(
                        timing["intermediate_snapshot_seconds"]
                    ) + (time.perf_counter() - snapshot_start)
                    payload, payload_timing = self._timed_observation_payload(snapshot)
                    self._accumulate_observation_payload_timing(
                        timing["intermediate_payload_encode"],
                        payload_timing,
                    )
                    timing["sampled_intermediate_observation_count"] = int(
                        timing["sampled_intermediate_observation_count"]
                    ) + 1
                    step_payloads.append(payload)
                    next_sample_time = now + self.step_observation_interval_sec
                time.sleep(POLL_STEP_SEC)

            timing["navigation_result_wait_seconds"] = float(time.perf_counter() - wait_start)
            wrapped = result_future.result()
            if wrapped.status != 2 and wrapped.status != 4:
                raise RuntimeError(f"{action_name} failed with status: {wrapped.status}")
        finally:
            self.navigation_node.clear_current_goal(goal_handle)
            self.navigation_node.clear_stop()

        final_snapshot_start = time.perf_counter()
        final_snapshot = self._snapshot(pump=True)
        timing["final_snapshot_seconds"] = float(time.perf_counter() - final_snapshot_start)
        final_pose = self._pose_from_snapshot(final_snapshot)
        if step_payloads != []:
            last_pose = step_payloads[-1]["pose"]
            if (
                abs(float(last_pose["x"]) - float(final_pose["x"])) <= 1e-4
                and abs(float(last_pose["y"]) - float(final_pose["y"])) <= 1e-4
                and abs(float(last_pose["yaw"]) - float(final_pose["yaw"])) <= 1e-3
            ):
                step_payloads = step_payloads[:-1]
        timing["returned_intermediate_observation_count"] = int(len(step_payloads))
        timing["elapsed_seconds"] = float(time.perf_counter() - total_start)
        return final_snapshot, step_payloads, timing

    def _run_navigation(
        self,
        pose_stamped: PoseStamped,
        sample_intermediate_observations: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        return self._run_nav_action(
            start_action=lambda: self.navigation_node.start_navigation(pose_stamped),
            sample_intermediate_observations=sample_intermediate_observations,
            action_name="NavigateToPose",
        )

    def _run_turn_controller(self, target_yaw_degrees: float) -> dict[str, float]:
        self.navigation_node.clear_stop()
        start_time = time.time()
        stable_start_time: float | None = None
        try:
            while True:
                if self.navigation_node.stop_requested():
                    self.navigation_node.send_zero_velocity()
                    raise NavigationStoppedException("Turn stopped")

                pose = self.observation_node.get_latest_base_pose()
                current_yaw_radians = math.radians(float(pose["yaw"]))
                target_yaw_radians = math.radians(float(target_yaw_degrees))
                yaw_error_radians = _wrap_angle_radians(target_yaw_radians - current_yaw_radians)
                yaw_error_degrees = math.degrees(yaw_error_radians)

                if abs(float(yaw_error_degrees)) <= float(TURN_YAW_TOLERANCE_DEGREES):
                    if stable_start_time is None:
                        stable_start_time = time.time()
                    self.navigation_node.send_zero_velocity()
                    if time.time() - stable_start_time >= float(TURN_STABLE_TIME_SEC):
                        return pose
                else:
                    stable_start_time = None
                    commanded_speed = float(TURN_CONTROLLER_KP) * float(yaw_error_radians)
                    commanded_speed = max(
                        -float(TURN_MAX_ANGULAR_SPEED_RAD_PER_SEC),
                        min(float(TURN_MAX_ANGULAR_SPEED_RAD_PER_SEC), commanded_speed),
                    )
                    if abs(commanded_speed) < float(TURN_MIN_ANGULAR_SPEED_RAD_PER_SEC):
                        commanded_speed = math.copysign(float(TURN_MIN_ANGULAR_SPEED_RAD_PER_SEC), commanded_speed)
                    self.navigation_node.publish_angular_velocity(commanded_speed)

                if time.time() - start_time > float(TURN_TIMEOUT_SEC):
                    self.navigation_node.send_zero_velocity()
                    raise RuntimeError(
                        "Turn controller timed out: "
                        f"target_yaw={float(target_yaw_degrees):.3f}, "
                        f"current_yaw={float(pose['yaw']):.3f}, "
                        f"yaw_error_degrees={float(yaw_error_degrees):.3f}"
                    )
                time.sleep(POLL_STEP_SEC)
        finally:
            self.navigation_node.send_zero_velocity()
            self.navigation_node.clear_stop()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "goal": self.goal,
            "camera_frame": self.observation_node.camera_frame,
            "base_frame": self.observation_node.base_frame,
            "odom_frame": self.observation_node.odom_frame,
        }

    def reset(self, goal: str | None) -> dict[str, Any]:
        if goal is not None:
            self.goal = str(goal)
        self._action_step_total = 0
        snapshot = self._snapshot(pump=True)
        payload = self._observation_payload(snapshot)
        payload.update(self._episode_metadata_payload())
        return payload

    def current_episode_info(self) -> dict[str, Any]:
        return self._episode_metadata_payload()

    def get_obs(self) -> dict[str, Any]:
        return self._observation_payload(self._snapshot(pump=True))

    def turn(self, direction: str) -> dict[str, Any]:
        if direction not in {"left", "right"}:
            raise ValueError(f"Unsupported turn direction: {direction}")
        pose = self.observation_node.get_latest_base_pose()
        delta_yaw = float(TURN_ANGLE_DEGREES) if direction == "left" else -float(TURN_ANGLE_DEGREES)
        self._run_turn_controller(target_yaw_degrees=float(pose["yaw"]) + delta_yaw)
        self._action_step_total += 1
        final_snapshot = self._snapshot(pump=True)
        return {
            "pose": self._pose_from_snapshot(final_snapshot),
            "observation": self._observation_payload(final_snapshot),
            "pointnav_step_total": int(self._action_step_total),
        }

    def move(self, x: float, y: float, yaw: float, z: float | None = None) -> dict[str, Any]:
        total_start = time.perf_counter()
        current_snapshot_start = time.perf_counter()
        current_snapshot = self._snapshot(pump=True)
        current_snapshot_seconds = time.perf_counter() - current_snapshot_start
        goal_pose_build_start = time.perf_counter()
        current_pose = self._pose_from_snapshot(current_snapshot)
        pose_stamped = self._goal_pose_stamped(
            x=float(x),
            y=float(y),
            z=float(current_pose["z"] if z is None else z),
            yaw=float(yaw),
        )
        goal_pose_build_seconds = time.perf_counter() - goal_pose_build_start
        final_snapshot, step_payloads, navigation_timing = self._run_navigation(
            pose_stamped=pose_stamped,
            sample_intermediate_observations=True,
        )
        self._action_step_total += max(1, len(step_payloads) + 1)
        return {
            "pose": self._pose_from_snapshot(final_snapshot),
            "intermediate_observations": step_payloads,
            "pointnav_step_total": int(self._action_step_total),
            "timing": {
                "server": {
                    "elapsed_seconds": float(time.perf_counter() - total_start),
                    "current_snapshot_seconds": float(current_snapshot_seconds),
                    "goal_pose_build_seconds": float(goal_pose_build_seconds),
                    "navigation": navigation_timing,
                    "intermediate_observation_count": int(len(step_payloads)),
                },
            },
        }

    def stop(self) -> dict[str, Any]:
        self.navigation_node.request_stop()
        self.navigation_node.send_zero_velocity()
        snapshot = self._snapshot(pump=True)
        return {
            "pose": self._pose_from_snapshot(snapshot),
            "success": True,
            "episode_success": False,
            "pointnav_step_total": int(self._action_step_total),
            "episode_over": False,
            "metrics": {},
            "goal": self.goal,
        }

    def finalize_run(self, reason: str) -> dict[str, Any]:
        return {
            "saved": False,
            "reason": str(reason),
            "goal": self.goal,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robot RPC server for navclaw agent.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1877)
    parser.add_argument("--goal", default=None)
    parser.add_argument("--pre-nav", action="store_true")
    parser.add_argument("--observation-settle-time-sec", type=float, default=0.1)
    parser.add_argument("--step-observation-interval-sec", type=float, default=0.5)
    return parser


def create_app(controller: RobotRPCController) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Any:
        return jsonify(controller.health())

    @app.post("/reset")
    def reset() -> Any:
        payload = request.get_json(silent=True) or {}
        return jsonify(controller.reset(goal=payload.get("goal")))

    @app.get("/current_episode")
    def current_episode() -> Any:
        return jsonify(controller.current_episode_info())

    @app.post("/get_obs")
    def get_obs() -> Any:
        return jsonify(controller.get_obs())

    @app.post("/turn")
    def turn() -> Any:
        payload = request.get_json(silent=True) or {}
        return jsonify(controller.turn(direction=str(payload.get("direction", ""))))

    @app.post("/move")
    def move() -> Any:
        payload = request.get_json(silent=True) or {}
        z = payload.get("z")
        return jsonify(
            controller.move(
                x=float(payload["x"]),
                y=float(payload["y"]),
                yaw=float(payload["yaw"]),
                z=None if z is None else float(z),
            )
        )

    @app.post("/stop")
    def stop() -> Any:
        return jsonify(controller.stop())

    @app.post("/finalize_run")
    def finalize_run() -> Any:
        payload = request.get_json(silent=True) or {}
        return jsonify(controller.finalize_run(reason=str(payload.get("reason", ""))))

    return app


def main() -> None:
    args = build_parser().parse_args()
    controller = RobotRPCController(
        goal=args.goal,
        observation_settle_time_sec=args.observation_settle_time_sec,
        step_observation_interval_sec=args.step_observation_interval_sec,
        pre_nav=bool(args.pre_nav),
    )
    app = create_app(controller)
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
