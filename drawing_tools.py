import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# draw sphere with radius r 
# also draw contours and vertical lines
def draw_sphere(r, sphere_colorscale, sphere_opacity):
    # sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # vertical lines on sphere
    u2 = np.linspace(0, 2 * np.pi, 20)
    x2 = r * np.outer(np.cos(u2), np.sin(v))
    y2 = r * np.outer(np.sin(u2), np.sin(v))
    z2 = r * np.outer(np.ones(np.size(u2)), np.cos(v))
    
    # create sphere and draw sphere with contours
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, 
                                 colorscale=sphere_colorscale, opacity=sphere_opacity,
                                 contours = {
                                     'z' : {'show' : True, 'start' : -r,
                                           'end' : r, 'size' : r/10,
                                           'color' : 'white',
                                           'width' : 1}
                                 }
                                , showscale=False)])
    
    # vertical lines on sphere
    for i in range(len(u2)):
        fig.add_scatter3d(x=x2[i], y=y2[i], z=z2[i], 
                          line=dict(
                              color='white',
                              width=1
                          ),
                         mode='lines',
                         showlegend=False)
    
    return fig

# draw xyplane
def draw_XYplane(fig, xy_plane_colorscale, xy_plane_opacity, x_range = [-2, 2], y_range = [-2, 2]):
    x3 = np.linspace(x_range[0], x_range[1], 100)
    y3 = np.linspace(y_range[0], y_range[1], 100)
    z3 = np.zeros(shape=(100,100))
    
    fig.add_surface(x=x3, y=y3, z=z3,
                colorscale =xy_plane_colorscale, opacity=xy_plane_opacity,
                showscale=False
    )
    
    return fig
    

def draw_XYZworld(fig, world_axis_size):
    # x, y, z positive direction (world)
    X_axis = [0, world_axis_size]
    X_text = [None, "X"]
    X0 = [0, 0]
    Y_axis = [0, world_axis_size]
    Y_text = [None, "Y"]
    Y0 = [0, 0]
    Z_axis = [0, world_axis_size]
    Z_text = [None, "Z"]
    Z0 = [0, 0]
    
    fig.add_scatter3d(x=X_axis, y=Y0, z=Z0, 
                      line=dict(
                          color='red',
                          width=10
                      ),
                    mode='lines+text',
                    text=X_text,
                    textposition='top center',
                    textfont=dict(
                        color="red",
                        size=18
                    ),
                    showlegend=False)

    fig.add_scatter3d(x=X0, y=Y_axis, z=Z0, 
                          line=dict(
                              color='green',
                              width=10
                          ),
                         mode='lines+text',
                        text=Y_text,
                        textposition='top center',
                        textfont=dict(
                            color="green",
                            size=18
                        ),
                        showlegend=False)

    fig.add_scatter3d(x=X0, y=Y0, z=Z_axis, 
                          line=dict(
                              color='blue',
                              width=10
                          ),
                         mode='lines+text',
                        text=Z_text,
                        textposition='top center',
                        textfont=dict(
                            color="blue",
                            size=18
                        ),
                        showlegend=False)
    
    return fig

# draw cam and cam coordinate system
def draw_cam_init(fig, world_mat, camera_axis_size, camera_color):
    # camera at init

    Xc = [world_mat[0, : ,3][0]]
    Yc = [world_mat[0, : ,3][1]]
    Zc = [world_mat[0, : ,3][2]]
    text_c = ["Camera"]

    # camera axis
    Xc_Xaxis = Xc + [world_mat[0, : ,0][0]*camera_axis_size+Xc[0]]
    Yc_Xaxis = Yc + [world_mat[0, : ,0][1]*camera_axis_size+Yc[0]]
    Zc_Xaxis = Zc + [world_mat[0, : ,0][2]*camera_axis_size+Zc[0]]
    text_Xaxis = [None, "Xc"]
    
    # -z in world perspective
    Xc_Yaxis = Xc + [world_mat[0, : ,1][0]*camera_axis_size+Xc[0]]
    Yc_Yaxis = Yc + [world_mat[0, : ,1][1]*camera_axis_size+Yc[0]]
    Zc_Yaxis = Zc + [world_mat[0, : ,1][2]*camera_axis_size+Zc[0]]
    text_Yaxis = [None, "Yc"]

    # y in world perspective
    Xc_Zaxis = Xc + [world_mat[0, : ,2][0]*camera_axis_size+Xc[0]]
    Yc_Zaxis = Yc + [world_mat[0, : ,2][1]*camera_axis_size+Yc[0]]
    Zc_Zaxis = Zc + [world_mat[0, : ,2][2]*camera_axis_size+Zc[0]]
    text_Zaxis = [None, "Zc"]
        
    # cam pos
    fig.add_scatter3d(x=Xc, y=Yc, z=Zc, 
                     mode='markers',
                  marker=dict(
                      color=camera_color,
                      size=4,
                      sizemode='diameter'
                  ),
                    showlegend=False)

    # camera axis
    fig.add_scatter3d(x=Xc_Xaxis, y=Yc_Xaxis, z=Zc_Xaxis, 
                          line=dict(
                              color='red',
                              width=10
                          ),
                        mode='lines+text',
                        text=text_Xaxis,
                        textposition='top center',
                        textfont=dict(
                            color="red",
                            size=18
                        ),
                        showlegend=False)

    fig.add_scatter3d(x=Xc_Yaxis, y=Yc_Yaxis, z=Zc_Yaxis, 
                          line=dict(
                              color='green',
                              width=10
                          ),
                        mode='lines+text',
                        text=text_Yaxis,
                        textposition='top center',
                        textfont=dict(
                            color="green",
                            size=18
                        ),
                        showlegend=False)

    fig.add_scatter3d(x=Xc_Zaxis, y=Yc_Zaxis, z=Zc_Zaxis, 
                          line=dict(
                              color='blue',
                              width=10
                          ),
                        mode='lines+text',
                        text=text_Zaxis,
                        textposition='top center',
                        textfont=dict(
                            color="blue",
                            size=18
                        ),
                        showlegend=False)
    
    return fig

# draw all rays
def draw_all_rays(fig, p_i, ray_color):
    for i in range(p_i.shape[1]):
        Xray = p_i[0, i, :, 0]
        Yray = p_i[0, i, :, 1]
        Zray = p_i[0, i, :, 2]
        
        fig.add_scatter3d(x=Xray, y=Yray, z=Zray, 
                          line=dict(
                              color=ray_color,
                              width=5
                          ),
                         mode='lines',
                        showlegend=False)
        
    return fig

# draw all rays
def draw_all_rays_with_marker(fig, p_i, marker_size, ray_color):
    
    # convert colorscale string to px.colors.seqeuntial
    # default color is set to Viridis in case of mismatch
    c = px.colors.sequential.Viridis

    for c_name in [ray_color, ray_color.capitalize()]:
        try:
            c = getattr(px.colors.sequential, c_name)
        except:
            continue
    
    for i in range(p_i.shape[1]):
        Xray = p_i[0, i, :, 0]
        Yray = p_i[0, i, :, 1]
        Zray = p_i[0, i, :, 2]
        
        fig.add_scatter3d(x=Xray, y=Yray, z=Zray, 
                          
                          marker=dict(
#                               color=np.arange(len(Xray)),
                              color=c,
#                               colorscale='Viridis',
                              size=marker_size
                          ),
                          
                          line=dict(
#                               color=np.arange(len(Xray)),
                              color=c,
#                               colorscale='Viridis',
                              width=3
                          ),
                         mode="lines+markers",
                        showlegend=False)
        
    return fig

# draw near&far frustrum with rays connecting the corners (changed for nerfpp)
def draw_ray_frus(fig, p_i, frustrum_color, frustrum_opacity, at=[0, -1]):
    
    for i in at:
#         Xfrus = p_i[0, :, i, 0][[0,1,2,3,7,11,15,14,13,12,8,4,0]]
#         Yfrus = p_i[0, :, i, 1][[0,1,2,3,7,11,15,14,13,12,8,4,0]]
#         Zfrus = p_i[0, :, i, 2][[0,1,2,3,7,11,15,14,13,12,8,4,0]]

        Xfrus = p_i[0, :, i, 0]
        Yfrus = p_i[0, :, i, 1]
        Zfrus = p_i[0, :, i, 2]
        
        fig.add_scatter3d(x=Xfrus, y=Yfrus, z=Zfrus, 
                        line=dict(
                              color=frustrum_color,
                              width=5
                          ),
                         mode='lines',
                          surfaceaxis=0,
                          surfacecolor=frustrum_color,
                          opacity=frustrum_opacity,
                        showlegend=False)
    
    return fig

# draw foreground sample points, ray and frustrum
def draw_foreground(fig, fg_pts, fg_color, marker_size, at=[0, -1]):
    fig = draw_all_rays_with_marker(fig, fg_pts, marker_size, fg_color)
    
    return fig

# draw background sample points, ray and frustrum
def draw_background(fig, bg_pts, bg_color, marker_size, at=[0, -1]):
    fig = draw_all_rays_with_marker(fig, bg_pts, marker_size, bg_color)
    
    return fig