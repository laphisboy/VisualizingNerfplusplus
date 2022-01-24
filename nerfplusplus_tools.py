import numpy as np
import torch

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6  

# from autonomousvision/giraffe

# 0.
def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.
    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor).float()
        is_numpy = True
    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor

# 1. get camera intrinsic
def get_camera_mat(fov=49.13, invert=True):
    # fov = 2 * arctan( sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))
    # in our case, sensor = 2 as pixels are in [-1, 1]
    focal = 1. / np.tan(0.5 * fov * np.pi/180.)
    focal = focal.astype(np.float32)
    mat = torch.tensor([
        [focal, 0., 0., 0.],
        [0., focal, 0., 0.],
        [0., 0., 1, 0.],
        [0., 0., 0., 1.]
    ]).reshape(1, 4, 4)

    if invert:
        mat = torch.inverse(mat)
    return mat

# 2. get camera position with camera pose (theta & phi)
def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return np.stack([cx, cy, cz], axis=-1)

# 3. get camera coordinate system assuming it points to the center of the sphere
def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5,
            to_pytorch=True):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)
    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis,
                                              axis=1, keepdims=True), eps]))

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis,
                                              axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis,
                                              axis=1, keepdims=True), eps]))

    r_mat = np.concatenate(
        (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(
            -1, 3, 1)), axis=2)

    if to_pytorch:
        r_mat = torch.tensor(r_mat).float()

    return r_mat

# 5. arange 2d array of pixel coordinate and give depth of 1
def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None, invert_y_axis=False):
    ''' Arranges pixels for given resolution in range image_range.
    The function returns the unscaled pixel locations as integers and the
    scaled float values.
    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = scale / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and
            subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    if invert_y_axis:
        assert(image_range == (-1, 1))
        pixel_scaled[..., -1] *= -1.
        pixel_locations[..., -1] = (h - 1) - pixel_locations[..., -1]

    return pixel_locations, pixel_scaled

# 6. mat_mul with intrinsic and then extrinsic gives you p_world (pixels in world) 
def image_points_to_world(image_points, camera_mat, world_mat, scale_mat=None,
                          invert=False, negative_depth=True):
    ''' Transforms points on image plane to world coordinates.
    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.
    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: False)
    '''
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    d_image = torch.ones(batch_size, n_pts, 1)
    if negative_depth:
        d_image *= -1.
    return transform_to_world(image_points, d_image, camera_mat, world_mat,
                              scale_mat, invert=invert)

def transform_to_world(pixels, depth, camera_mat, world_mat, scale_mat=None,
                       invert=True, use_absolute_depth=True):
    ''' Transforms pixel positions p with given depth value d to world coordinates.
    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)

    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(0).repeat(
            camera_mat.shape[0], 1, 1)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)

    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    if use_absolute_depth:
        pixels[:, :2] = pixels[:, :2] * depth.permute(0, 2, 1).abs()
        pixels[:, 2:3] = pixels[:, 2:3] * depth.permute(0, 2, 1)
    else:
        pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)
        
    # Transform pixels to world space
    p_world = scale_mat @ world_mat @ camera_mat @ pixels

    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world


# 7. mat_mul zeros with intrinsic&extrinsic for camera pos (which we alread obtained as loc)
def origin_to_world(n_points, camera_mat, world_mat, scale_mat=None,
                    invert=False):
    ''' Transforms origin (camera location) to world coordinates.
    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: false)
    '''
    
    batch_size = camera_mat.shape[0]
    device = camera_mat.device
    # Create origin in homogen coordinates
    p = torch.zeros(batch_size, 4, n_points).to(device)
    p[:, -1] = 1.

    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(
            0).repeat(batch_size, 1, 1).to(device)

    # Invert matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)
        
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)
    
    # Apply transformation
    p_world = scale_mat @ world_mat @ camera_mat @ p

    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world


# from Kai-46/nerfplusplus

# 8. intersect sphere for distinguishing fg and bg
def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception('Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2

# 9. inverse sphere sampling for bg
def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new.squeeze(), depth.squeeze().unsqueeze(-1)), dim=-1) # (modified) added .squeeze()

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real

# 10. perturb sample z values (depth) for some randomness (stratified sampling)
def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    return z_vals

# 11. return fg (just uniform) and bg (uniform inverse sphere) samples
def uniformsampling(ray_o, ray_d, min_depth, N_samples, device):
    
    dots_sh = list(ray_d.shape[:-1])

    # foreground depth
    fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
    fg_near_depth = min_depth  # [..., ]
    step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
    fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]
    fg_depth = perturb_samples(fg_depth)   # random perturbation during training

    # background depth
    bg_depth = torch.linspace(0., 1., N_samples).view(
                [1, ] * len(dots_sh) + [N_samples,]).expand(dots_sh + [N_samples,]).to(device)
    bg_depth = perturb_samples(bg_depth)   # random perturbation during training


    fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
    fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])

    bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
    bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
    
    # sampling foreground
    fg_pts = fg_ray_o + fg_depth.unsqueeze(-1) * fg_ray_d

    # sampling background
    bg_pts, bg_depth_real = depth2pts_outside(bg_ray_o, bg_ray_d, bg_depth)

    return fg_pts, bg_pts, bg_depth_real

# 12 convert bg pts (x', y', z' 1/r) to real bg pts (x, y, z)
def pts2realpts(bg_pts, bg_depth_real):
    return bg_pts[:, :, :3] * bg_depth_real.squeeze().unsqueeze(-1)

def nerfpp(u = 1,
           v = 0.5,
           fov = 49.13,
           depth_range=[0.5, 6.],
           n_ray_samples=16,
           resolution_vol = 4,
           batch_size = 1,
           device = torch.device('cpu')
           ):
    
    r = 1
    range_radius=[r, r]
    
    res = resolution_vol
    n_points = res * res

    # 1. get camera intrinsic - fiddle around with fov this time
    camera_mat = get_camera_mat(fov=fov)

    # 2. get camera position with camera pose (theta & phi)
    loc = to_sphere(u, v)
    loc = torch.tensor(loc).float()
    
    radius = range_radius[0] + \
        torch.rand(batch_size) * (range_radius[1] - range_radius[0])
    
    loc = loc * radius.unsqueeze(-1)

    # 3. get camera coordinate system assuming it points to the center of the sphere
    R = look_at(loc)

    # 4. The carmera coordinate is the rotational matrix and with camera loc, it is camera extrinsic
    RT = np.eye(4).reshape(1, 4, 4)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc
    world_mat = RT

    # 5. arange 2d array of pixel coordinate and give depth of 1
    pixels = arange_pixels((res, res), 1, invert_y_axis=False)[1]
    pixels[..., -1] *= -1. # still dunno why this is here

    # 6. mat_mul with intrinsic and then extrinsic gives you p_world (pixels in world) 
    pixels_world = image_points_to_world(pixels, camera_mat, world_mat)

    # 7. mat_mul zeros with intrinsic&extrinsic for camera pos (which we alread obtained as loc)
    camera_world = origin_to_world(n_points, camera_mat, world_mat)

    # 8. ray = pixel - camera origin (in world)
    ray_vector = pixels_world - camera_world

    # 9. sample fg and bg points according to nerfpp (uniform and inverse sphere)
    fg_pts, bg_pts, bg_depth_real = uniformsampling(ray_o=camera_world, ray_d=ray_vector, min_depth=depth_range[0], N_samples=n_ray_samples, device=device)
    
    #10. convert bg pts (x', y', z' 1/r) to real bg pts (x, y, z)
    bg_pts_real = pts2realpts(bg_pts, bg_depth_real)
    
    return pixels_world, camera_world, world_mat, fg_pts, bg_pts_real
