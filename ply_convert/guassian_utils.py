# Adapted from the version at https://github.com/limacv/GaussianSplattingViewer/blob/main/util_gau.py
import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
import scipy as sp

@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)

    def __len__(self):
        return len(self.xyz)

    @property
    def sh_dim(self):
        return self.sh.shape[-1]


def naive_gaussian():
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)
    gau_s = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = np.array([
        1, 0, 1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = (gau_c - 0.5) / 0.28209
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)
    return GaussianData(
        gau_xyz,
        gau_rot,
        gau_s,
        gau_a,
        gau_c
    )




scale_modifier = 1.0
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2_0 = 1.0925484305920792
SH_C2_1 = -1.0925484305920792
SH_C2_2 = 0.31539156525252005
SH_C2_3 = -1.0925484305920792
SH_C2_4 = 0.5462742152960396
SH_C3_0 = -0.5900435899266435
SH_C3_1 = 2.890611442640554
SH_C3_2 = -0.4570457994644658
SH_C3_3 = 0.3731763325901154
SH_C3_4 = -0.4570457994644658
SH_C3_5 = 1.445305721320277
SH_C3_6 = -0.5900435899266435


class Gaussian:
    def __init__(self, pos, scale, rot, opacity, sh):
        self.pos = np.array(pos)
        self.scale = np.array(scale_modifier * scale)
        # Initialize scipy Quaternion from rot (s, x, y, z)
        self.rot = sp.spatial.transform.Rotation.from_quat([rot[1], rot[2], rot[3], rot[0]])
        self.opacity = opacity[0]
        self.sh = np.array(sh)
        self.cov3D = self.compute_cov3d()
        self.bbox = self.compute_bbox()
        self.iso_color = self.get_iso_color()

    def compute_cov3d(self):
        cov3D = np.diag(self.scale**2)
        cov3D = self.rot.as_matrix().T @ cov3D @ self.rot.as_matrix()
        return cov3D

    def compute_bbox(self):
        # Eigen decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov3D)
        
        # Compute the standard deviations along the principal axes
        std_devs = np.sqrt(eigenvalues)
        
        # Compute the 3-sigma values for each principal axis
        three_sigma_values = 3 * std_devs
        
        # Compute the bounding box in the principal axis coordinates
        # This assumes the Gaussian is oriented along the principal axes
        bounding_box_offsets = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], 
                                        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])
        bounding_box_offsets *= three_sigma_values
        
        # Rotate the bounding box back to the original coordinate system
        # and translate it to the Gaussian's center
        vertices = np.dot(bounding_box_offsets, eigenvectors) + self.pos
        return np.array(vertices)
  
    def get_iso_color(self) -> np.ndarray:
        """Samples spherical harmonics to get color for given view direction"""
        c0 = self.sh[0:3]   # f_dc_* from the ply file)
        color = SH_C0 * c0
        color += 0.5
        return np.clip(color, 0.0, 1.0)

    def compute_3d_gaussian_prob(self, position):
        prob = np.exp(-0.5 * (position - self.pos).T @ np.linalg.inv(self.cov3D) @ (position - self.pos))
        return prob
    
    def compute_blob_ellipsoid(self):
        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov3D)
        
        # Get the radii of the ellipsoid
        radii = np.sqrt(eigenvalues)*3
        
        # Generate data for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        
        # Rotate and translate the points to the correct position
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j], y[i,j], z[i,j]] = np.dot([x[i,j], y[i,j], z[i,j]], eigenvectors) + self.pos
        
        return x,y,z

    def get_cov2d(self, camera):
        view_mat = camera.get_view_matrix()
        g_pos_w = np.append(self.pos, 1.0)
        # g_pos_cam = camera.world_to_cam(self.pos)
        g_pos_cam = view_mat @ g_pos_w
        view_matrix = camera.get_view_matrix()
        [htan_fovx, htan_fovy, focal] = camera.get_htanfovxy_focal()
        focal_x = focal_y = focal

        t = np.copy(g_pos_cam)

        limx = 1.3 * htan_fovx
        limy = 1.3 * htan_fovy
        txtz = t[0]/t[2]
        tytz = t[1]/t[2]

        tx = min(limx, max(-limx, txtz)) * t[2]
        ty = min(limy, max(-limy, tytz)) * t[2]
        tz = t[2]

        J = np.array([
            [focal_x/tz, 0.0, -(focal_x * tx)/(tz * tz)],
            [0.0, focal_y/tz, -(focal_y * ty)/(tz * tz)],
            [0.0, 0.0, 0.0]
        ])
        W = view_matrix[:3, :3].T
        T = W @ J
        cov = T.T @ self.cov3D.T @ T

        cov[0,0] += 0.3
        cov[1,1] += 0.3
        return cov[:2, :2]

    def get_depth(self, camera):
        view_matrix = camera.get_view_matrix()
        
        position4 = np.append(self.pos, 1.0)
        g_pos_view = view_matrix @ position4
        depth = g_pos_view[2]
        return depth

    def get_conic_and_bb(self, camera):
        cov2d = self.get_cov2d(camera)

        det = np.linalg.det(cov2d)
        if det == 0.0:
            return None
        
        det_inv = 1.0 / det
        cov = [cov2d[0,0], cov2d[0,1], cov2d[1,1]]
        conic = np.array([cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv])
        # cov_inv= np.linalg.inv(cov2d)
        # conic = np.array([cov_inv[0,0], cov_inv[0,1], cov_inv[1,1]])
        # compute 3-sigma bounding box size
        bboxsize_cam = np.array([3.0 * np.sqrt(cov2d[0,0]), 3.0 * np.sqrt(cov2d[1,1])])
        # bboxsize_cam = np.array([3.0 * np.sqrt(cov[0]), 3.0 * np.sqrt(cov[2])])        
        # Divide out camera plane size to get bounding box size in NDC
        wh = np.array([camera.w, camera.h])
        bboxsize_ndc = np.divide(bboxsize_cam, wh) * 2

        vertices = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])
        # Four coordxy values (used to evaluate gaussian, also in camera space coordinates)
        bboxsize_cam = np.multiply(vertices, bboxsize_cam)

        # compute g_pos_screen and gl_position
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix()

        position4 = np.append(self.pos, 1.0)
        g_pos_view = view_matrix @ position4
        g_pos_screen = projection_matrix @ g_pos_view
        g_pos_screen = g_pos_screen / g_pos_screen[3]
        
        bbox_ndc = np.multiply(vertices, bboxsize_ndc) + g_pos_screen[:2]
        bbox_ndc = np.hstack((bbox_ndc, np.zeros((vertices.shape[0],2))))
        bbox_ndc[:,2:4] = g_pos_screen[2:4]

        return conic, bboxsize_cam, bbox_ndc

    def get_color(self, dir) -> np.ndarray:
        """Samples spherical harmonics to get color for given view direction"""
        c0 = self.sh[0:3]   # f_dc_* from the ply file)
        color = SH_C0 * c0

        shdim = len(self.sh)

        if shdim > 3:
            # Add the first order spherical harmonics
            c1 = self.sh[3:6]
            c2 = self.sh[6:9]
            c3 = self.sh[9:12]
    
            x = dir[0]
            y = dir[1]
            z = dir[2]
            color = color - SH_C1 * y * c1 + SH_C1 * z * c2 - SH_C1 * x * c3
            
        if shdim > 12:
            c4 = self.sh[12:15]
            c5 = self.sh[15:18]
            c6 = self.sh[18:21]
            c7 = self.sh[21:24]
            c8 = self.sh[24:27]
    
            (xx, yy, zz) = (x * x, y * y, z * z)
            (xy, yz, xz) = (x * y, y * z, x * z)
            
            color = color +	SH_C2_0 * xy * c4 + \
                SH_C2_1 * yz * c5 + \
                SH_C2_2 * (2.0 * zz - xx - yy) * c6 + \
                SH_C2_3 * xz * c7 + \
                SH_C2_4 * (xx - yy) * c8

        if shdim > 27:
            c9 = self.sh[27:30]
            c10 = self.sh[30:33]
            c11 = self.sh[33:36]
            c12 = self.sh[36:39]
            c13 = self.sh[39:42]
            c14 = self.sh[42:45]
            c15 = self.sh[45:48]
    
            color = color + \
                SH_C3_0 * y * (3.0 * xx - yy) * c9 + \
                SH_C3_1 * xy * z * c10 + \
                SH_C3_2 * y * (4.0 * zz - xx - yy) * c11 + \
                SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * c12 + \
                SH_C3_4 * x * (4.0 * zz - xx - yy) * c13 + \
                SH_C3_5 * z * (xx - yy) * c14 + \
                SH_C3_6 * x * (xx - 3.0 * yy) * c15
        
        color += 0.5
        return np.clip(color, 0.0, 1.0)
