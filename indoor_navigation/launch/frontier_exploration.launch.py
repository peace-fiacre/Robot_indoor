from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="indoor_navigation",
            executable="frontier_explorer.py",
            name="frontier_explorer",
            output="screen",
            parameters=[
                {"map_topic": "/map"},
                {"min_cluster_size": 10},      # baisse pour tester
                {"publish_period_s": 1.0},
                {"use_sim_time": True},

                {"global_frame": "map"},
                {"base_frame": "base_link"},
                {"nav_action_name": "/navigate_to_pose"},

                {"min_goal_separation_m": 0.8},
                {"goal_cooldown_s": 3.0},

                {"score_distance_weight": 1.0},
                {"score_size_weight": 0.05},

                {"goal_backoff_m": 0.6},
                {"goal_search_radius_m": 1.5},
                {"goal_clearance_cells": 2},
                {"min_cluster_size": 15},
            ],
        ),
    ])