#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject, PlanningScene, ObjectColor

# scene_config resolves the same robot_params.yaml (ament share first, then the
# repo copy) and exposes the geometry that used to be hardcoded in this file:
# table poses/sizes, the obstacle, the demo box, per-class object sizes.
import scene_config as _scene


class PlanningSceneManager:
    def __init__(self, node: Node = None):
        """
        Allows PlanningSceneManager to either:
        1. Run as its own standalone ROS 2 Node (for testing).
        2. Attach to an existing PickAndPlace node instance.

        Every geometric value (tables, obstacle, demo box, object sizes, world
        offsets) comes from robot_params.yaml through scene_config.py — nothing
        in this file re-types a table size or an object pose.
        """
        # robot_params.yaml is loaded once by scene_config (ament share path
        # when the workspace is sourced, repo path for bare pytest).  self.params
        # is kept for callers that read the raw YAML.
        self.params = _scene.PARAMS
        self.own_node = False
        if node is None:
            if not rclpy.ok():
                rclpy.init()
            self.node = rclpy.create_node(
                'planning_scene_manager',
                parameter_overrides=[
                    rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, self.params['robot']['use_sim_time'])
                ]
            )
            self.own_node = True
        else:
            self.node = node

        # Publishers for MoveIt planning scene
        self.collision_object_pub = self.node.create_publisher(
            CollisionObject, '/collision_object', 10
        )
        self.attached_object_pub = self.node.create_publisher(
            AttachedCollisionObject, '/attached_collision_object', 10
        )
        self.planning_scene_pub = self.node.create_publisher(
            PlanningScene, '/planning_scene', 10
        )

        # Track every collision-object id we published so clear_all_objects()
        # can remove exactly what we added (tables, obstacle, pick objects).
        self._published_ids = set()

        # Allow subscribers time to connect
        time.sleep(0.5)

    def _create_header(self, frame_id='world') -> Header:
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = frame_id
        return header

    # ── Frame helpers: every position passed in is GAZEBO WORLD coords ───────
    @staticmethod
    def _local_z(world_z: float) -> float:
        """Gazebo world z → arm-local z ('world' link sits at the arm base)."""
        return float(world_z) - _scene.ARM_BASE_WORLD_Z

    @classmethod
    def _table_box_local(cls, centre) -> tuple:
        """Table box centre in the arm frame, lowered by table_top_clearance."""
        clearance = float(_scene.SCENE['table_top_clearance'])
        return (float(centre[0]), float(centre[1]),
                cls._local_z(centre[2]) - clearance)

    @staticmethod
    def _colour(values, alpha: float = 0.95) -> ColorRGBA:
        r, g, b = (float(v) for v in values[:3])
        return ColorRGBA(r=r, g=g, b=b, a=alpha)

    def add_solid_object(self, name: str, shape: str, size: tuple, position: tuple,
                         orientation=(0.0, 0.0, 0.0, 1.0), frame_id='world',
                         color: ColorRGBA = None):
        """Publish a box or cylinder CollisionObject (world coords, arm 'world' frame).

        shape: 'box'      → size = (x, y, z)
               'cylinder' → size = (diameter, diameter, length), axis = local +Z
        """
        primitive = SolidPrimitive()
        if shape == 'cylinder':
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [float(size[2]), float(size[0])]
        else:
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [float(size[0]), float(size[1]), float(size[2])]

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(orientation[0])
        pose.orientation.y = float(orientation[1])
        pose.orientation.z = float(orientation[2])
        pose.orientation.w = float(orientation[3])

        obj = CollisionObject()
        obj.header = self._create_header(frame_id)
        obj.id = name
        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD

        self._published_ids.add(name)
        self.collision_object_pub.publish(obj)

        # If custom color is provided, publish it via PlanningScene diff
        if color is not None:
            scene_diff = PlanningScene()
            scene_diff.is_diff = True
            obj_color = ObjectColor()
            obj_color.id = name
            obj_color.color = color
            scene_diff.object_colors = [obj_color]
            self.planning_scene_pub.publish(scene_diff)

        self.node.get_logger().info(
            f"Added {shape} collision object '{name}' at "
            f"({float(position[0]):.3f}, {float(position[1]):.3f}, "
            f"{float(position[2]):.3f}) in frame '{frame_id}'"
        )
        time.sleep(0.1)

    def add_box_object(self, name: str, size: tuple, position: tuple, orientation=(0.0, 0.0, 0.0, 1.0), frame_id='world', color: ColorRGBA = None):
        """Creates and publishes a box CollisionObject with optional custom color."""
        return self.add_solid_object(name, 'box', size, position, orientation,
                                     frame_id, color)

    def add_cylinder_object(self, name: str, size: tuple, position: tuple, orientation=(0.0, 0.0, 0.0, 1.0), frame_id='world', color: ColorRGBA = None):
        """Creates and publishes a cylinder CollisionObject (axis = local +Z)."""
        return self.add_solid_object(name, 'cylinder', size, position, orientation,
                                     frame_id, color)


    def remove_object(self, name: str):
        """Removes a CollisionObject by ID."""
        obj = CollisionObject()
        obj.header = self._create_header()
        obj.id = name
        obj.operation = CollisionObject.REMOVE
        self.collision_object_pub.publish(obj)
        self._published_ids.discard(name)
        self.node.get_logger().info(f"Removed collision object '{name}'")
        time.sleep(0.1)

    def add_static_environment(self):
        """
        Publishes every collision box listed in robot_params.yaml scene.tables
        (plus scene.obstacle when enabled), converted from Gazebo world coords
        to the arm-local 'world' frame.

        The values are no longer written here: they come from the YAML block
        that test_params.py cross-checks against worlds/vision_sorting.sdf, so
        the collision scene can never silently drift from the simulated world
        (the old hardcoded table was 1.4 m wide at x = -0.5, which did not
        match either table in vision_sorting.sdf).
        """
        table_color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.85)
        for table in _scene.SCENE['tables']:
            self.add_box_object(
                name=table['name'],
                size=tuple(table['size']),
                position=self._table_box_local(table['centre']),
                color=table_color,
            )
        if _scene.SCENE['obstacle'].get('enabled', False):
            self.add_obstacle()

    def add_obstacle(self):
        """Adds the divider wall from scene.obstacle (disabled by default:
        warehouse.sdf geometry — vision_sorting.sdf has no such wall)."""
        obstacle = _scene.SCENE['obstacle']
        obstacle_color = ColorRGBA(r=1.0, g=0.5, b=0.5, a=0.9)
        centre = obstacle['centre']
        self.add_box_object(
            name=obstacle['name'],
            size=tuple(obstacle['size']),
            position=(float(centre[0]), float(centre[1]), self._local_z(centre[2])),
            color=obstacle_color,
        )

    # ── Pick objects (boxes / cylinders) from the scene.objects block ────────
    def add_object_from_params(self, name: str, class_id: str, position: tuple,
                               orientation=(0.0, 0.0, 0.0, 1.0)):
        """Publish a detected object's collision geometry at a WORLD-frame pose.

        class_id indexes scene.objects (shape + size per class); unknown classes
        fall back to the demo box spec.  position z must be the GAZEBO world z
        (e.g. table_top_world_z for an object resting on the surface).
        """
        spec = _scene.SCENE['objects'].get(class_id)
        if spec is None:
            self.node.get_logger().warning(
                f"'{class_id}' is not in scene.objects — using the demo box spec"
            )
            spec = _scene.SCENE['target_box']
        colour = spec.get('colour') or _scene.SCENE['target_box']['colour']
        self.add_solid_object(
            name=name,
            shape=spec.get('shape', 'box'),
            size=tuple(spec['size']),
            position=(float(position[0]), float(position[1]),
                      self._local_z(position[2])),
            orientation=orientation,
            color=self._colour(colour),
        )

    def add_target_box(self, position=None, size=None):
        """Adds the configured demo box (scene.target_box) — see `position` in
        GAZEBO world coords; defaults to the configured pose."""
        box = _scene.SCENE['target_box']
        if position is None:
            position = (box['position'][0], box['position'][1], box['position'][2])
        if size is None:
            size = tuple(box['size'])
        self.add_box_object(
            name=box['id'],
            size=size,
            position=(float(position[0]), float(position[1]),
                      self._local_z(position[2])),
            color=self._colour(box['colour']),
        )

    def init_scene_from_params(self):
        """Initialize the collision scene (tables + optional obstacle) from YAML.

        Pick objects are added later, at the pose perception reports, via
        add_object_from_params().
        """
        self.node.get_logger().info('Initializing MoveIt planning scene from robot_params.yaml...')
        self.add_static_environment()
        tables = ', '.join(t['name'] for t in _scene.SCENE['tables'])
        obstacle = 'enabled' if _scene.SCENE['obstacle'].get('enabled') else 'disabled'
        self.node.get_logger().info(
            f'Planning scene ready: tables [{tables}], obstacle {obstacle} '
            f'(arm base at world z={_scene.ARM_BASE_WORLD_Z:.2f})'
        )

    def init_full_scene(self):
        """Legacy wrapper: static environment + the configured demo box."""
        self.node.get_logger().info('Initializing complete MoveIt planning scene...')
        self.add_static_environment()
        self.add_target_box()
        self.node.get_logger().info('Planning scene successfully initialized.')

    def attach_object_to_gripper(self, object_id: str, size: tuple,
                                 link_name: str = 'gripper_base_link',
                                 shape: str = 'box'):
        """
        Attaches a collision object to the robot gripper.
        Sets touch_links to avoid false collision aborts with fingers.
        """
        attached_object = AttachedCollisionObject()
        attached_object.link_name = link_name
        attached_object.object.header = self._create_header(frame_id=link_name)
        attached_object.object.id = object_id
        attached_object.object.operation = CollisionObject.ADD

        # Define the grasped geometry relative to the gripper link
        grasp_shape = SolidPrimitive()
        if shape == 'cylinder':
            grasp_shape.type = SolidPrimitive.CYLINDER
            grasp_shape.dimensions = [float(size[2]), float(size[0])]
        else:
            grasp_shape.type = SolidPrimitive.BOX
            grasp_shape.dimensions = [float(size[0]), float(size[1]), float(size[2])]

        box_pose = Pose()
        box_pose.position.x = 0.0
        box_pose.position.y = 0.0
        # E5: jaw span centre (grasp plane) — prong mid-plane sits 0.018 m
        # above gripper_base_link (arm.urdf.xacro prong origins), NOT at the
        # 0.075 m TCP.  The IK grasps at TCP_Z_OFFSET = 0.113 = d5 + 0.018
        # (Finding B), so the carried object must be attached at 0.018.
        box_pose.position.z = float(_scene.KINEMATICS['jaw_centre_offset'])
        box_pose.orientation.w = 1.0

        attached_object.object.primitives = [grasp_shape]
        attached_object.object.primitive_poses = [box_pose]

        # Allowed contact links (vital to prevent self-collision errors)
        attached_object.touch_links = [
            'right_prong_link',
            'left_prong_link',
            'gripper_base_link',
            'gripper_tcp',
            'wrist_gripper_link'
        ]

        self.attached_object_pub.publish(attached_object)
        self.node.get_logger().info(f"Attached '{object_id}' to '{link_name}'")
        time.sleep(0.2)

    def detach_object_from_gripper(self, object_id: str, world_position: tuple,
                                   size: tuple, colour=None, shape: str = 'box'):
        """
        Detaches a collision object from the gripper and re-adds it in the arm
        'world' frame at a GAZEBO world position (z converted here).

        size/shape must describe the object that was attached (scene.objects);
        colour defaults to the demo box colour when not supplied.
        """
        # 1. Detach from gripper
        detached_object = AttachedCollisionObject()
        detached_object.object.header = self._create_header(frame_id='world')
        detached_object.object.id = object_id
        detached_object.object.operation = CollisionObject.REMOVE
        self.attached_object_pub.publish(detached_object)
        self.node.get_logger().info(f"Detached '{object_id}' from gripper")
        time.sleep(0.1)

        # 2. Re-add as static object at the drop location
        self.add_solid_object(
            name=object_id,
            shape=shape,
            size=size,
            position=(float(world_position[0]), float(world_position[1]),
                      self._local_z(world_position[2])),
            color=self._colour(colour or _scene.SCENE['target_box']['colour']),
        )
        self.node.get_logger().info(
            f"Re-added '{object_id}' at Gazebo world position {world_position}"
        )

    # ── Legacy wrappers (config-driven demo box) ─────────────────────────────
    def attach_target_box_to_gripper(self, link_name='gripper_base_link', size=None):
        """Attaches the configured demo box ('target_box') to the gripper."""
        box = _scene.SCENE['target_box']
        if size is None:
            size = tuple(box['size'])
        self.attach_object_to_gripper(box['id'], size, link_name,
                                      shape=box.get('shape', 'box'))

    def detach_target_box_from_gripper(self, drop_position=None, size=None):
        """Detaches 'target_box' and re-adds it at the drop pose (world coords).

        Default drop pose = the first red sort-zone slot on the table surface
        (robot_params.yaml sort_zones.red + scene.table_top_world_z), i.e. the
        same place the vision mission would put a red cube.
        """
        box = _scene.SCENE['target_box']
        if drop_position is None:
            slot = (_scene.SORT_ZONES.get('red') or [[0.0, 0.0]])[0]
            drop_position = (float(slot[0]), float(slot[1]), _scene.TABLE_TOP_WORLD_Z)
        if size is None:
            size = tuple(box['size'])
        self.detach_object_from_gripper(box['id'], drop_position, size,
                                        colour=box['colour'],
                                        shape=box.get('shape', 'box'))

    def clear_all_objects(self):
        """Removes every collision object this manager published."""
        if not self._published_ids:
            self.node.get_logger().info('No collision objects to clear.')
            return
        for name in sorted(self._published_ids):
            self.remove_object(name)
        self.node.get_logger().info('Cleared all objects from planning scene.')


def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)
    manager = PlanningSceneManager()
    manager.init_full_scene()

    try:
        manager.node.get_logger().info('PlanningSceneManager is active with custom colors. Press Ctrl+C to stop.')
        rclpy.spin(manager.node)
    except KeyboardInterrupt:
        pass
    finally:
        if manager.own_node:
            manager.node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
