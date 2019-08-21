class ThreeJointFeatures(object):
    __slots__ = ['angle_alpha_rad', 'angle_beta_rad', 'angle_gamma_rad']
    angle_alpha_rad: float
    angle_beta_rad: float
    angle_gamma_rad: float

    def __init__(self, angle_alpha_rad: float, angle_beta_rad: float, angle_gamma_rad: float):
        """
        Contains features between three 3D joints.
        :param angle_alpha_rad: the alpha angle of the joint triangle.
        :param angle_beta_rad:
        :param angle_gamma_rad:
        """
        self.angle_alpha_rad = angle_alpha_rad
        self.angle_beta_rad = angle_beta_rad
        self.angle_gamma_rad = angle_gamma_rad
