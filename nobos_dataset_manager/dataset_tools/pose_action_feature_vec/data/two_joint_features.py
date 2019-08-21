class TwoJointFeatures(object):
    __slots__ = ['normalized_distance_x', 'normalized_distance_y', 'normalized_distance_euclidean', 'angle_rad']
    normalized_distance_x: float
    normalized_distance_y: float
    normalized_distance_euclidean: float
    angle_rad: float

    def __init__(self, normalized_distance_x: float, normalized_distance_y: float, normalized_distance_euclidean: float,
                 angle_rad: float):
        """
        Contains feature between two 2D joints.
        :param normalized_distance_x: the distance between the joint's x coordinates, normalized by human hight
        :param normalized_distance_y: the distance between the joint's y coordinates, normalized by human hight
        :param normalized_distance_euclidean: the euclidean distance between the joints 2D coordinates, normalized by human hight
        :param angle_rad: the angle between the two joints
        """
        self.normalized_distance_x = normalized_distance_x
        self.normalized_distance_y = normalized_distance_y
        self.normalized_distance_euclidean = normalized_distance_euclidean
        self.angle_rad = angle_rad
