# src/robotic_4dof_arm/test/test_scene_manager.py
import sys
import os
import pytest
from unittest.mock import MagicMock

# Point Python at the scripts folder BEFORE importing from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from planning_scene_manager import PlanningSceneManager  # noqa: E402


@pytest.fixture
def manager():
    """Creates a PlanningSceneManager with a fully mocked ROS node.

    No ament patch is needed any more: scene_config resolves robot_params.yaml
    (ament share first, repo copy as the fallback), so the manager no longer
    looks the share directory up itself.
    """
    mock_node = MagicMock()
    mock_node.get_clock.return_value.now.return_value.to_msg.return_value = MagicMock()

    mgr = PlanningSceneManager(node=mock_node)
    return mgr


def test_params_come_from_scene_config(manager):
    """The manager must read geometry from robot_params.yaml, not literals."""
    import scene_config as _scene
    assert manager.params is _scene.PARAMS
    assert manager.params['scene']['tables'] == _scene.SCENE['tables']


def test_add_box_publishes(manager):
    manager.add_box_object('test_box', (0.1, 0.1, 0.1), (0.0, 0.0, 0.5))
    assert manager.collision_object_pub.publish.called

def test_remove_object_publishes(manager):
    manager.remove_object('test_box')
    manager.collision_object_pub.publish.assert_called()
    # Verify the operation is REMOVE (val = 3)
    published_obj = manager.collision_object_pub.publish.call_args[0][0]
    from moveit_msgs.msg import CollisionObject
    assert published_obj.operation == CollisionObject.REMOVE

def test_attach_publishes(manager):
    manager.attach_target_box_to_gripper(link_name='gripper_base_link')
    assert manager.attached_object_pub.publish.called

def test_detach_publishes(manager):
    manager.detach_target_box_from_gripper(drop_position=(0.0, 0.0, 0.1))
    assert manager.attached_object_pub.publish.called

def test_init_full_scene_calls_all(manager):
    manager.add_static_environment = MagicMock()
    manager.add_target_box = MagicMock()
    manager.init_full_scene()
    manager.add_static_environment.assert_called_once()
    manager.add_target_box.assert_called_once()
