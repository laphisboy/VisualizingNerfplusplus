import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# for colors
import matplotlib.colors as mcolors

from drawing_tools import *
from nerfplusplus_tools import *

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Nerfplusplus ray sampling visualization"),
    dcc.Graph(id='graph'),
    
    html.Div([
        html.Div([
            
            # changes to setting and ray casted 
            html.Label([ "u (theta)",
                dcc.Slider(
                    id='u-slider', 
                    min=0, max=1,
                    value=0.00,
                    marks={str(val) : str(val) for val in [0.00, 0.25, 0.50, 0.75]},
                    step=0.01, tooltip = { 'always_visible': True }
                ), ]),
            html.Label([ "v (phi)",
                dcc.Slider(
                    id='v-slider', 
                    min=0, max=1,
                    value=0.25,
                    marks={str(val) : str(val) for val in [0.00, 0.25, 0.50, 0.75]},
                    step=0.01, tooltip = { 'always_visible': True }
                ), ]),
            html.Label([ "fov (field-of-view))",
                dcc.Slider(
                    id='fov-slider', 
                    min=0, max=100,
                    value=50,
                    marks={str(val) : str(val) for val in [0, 20, 40, 60, 80, 100]},
                    step=5, tooltip = { 'always_visible': True }
                ), ]),
            html.Label([ "foreground near depth",
                dcc.Slider(
                    id='foreground-near-depth-slider', 
                    min=0, max=2,
                    value=0.5,
                    marks={f"{val:.1f}" : f"{val:.1f}" for val in [0.1 * i for i in range(21)]},
                    step=0.1, tooltip = { 'always_visible': True }
                ), ])
        ], style = {'width' : '48%', 'display' : 'inline-block'}),
        
        html.Div([
            # changes to visual appearance
            
            # axis scale
            html.Div([
                html.Label([ "world axis size",
                    html.Div([
                        dcc.Input(id='world-axis-size-input',
                                  value=1.5,
                                  type='number', style={'width': '50%'}
                                 )
                    ]),
                ], style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                html.Label([ "camera axis size",
                    html.Div([
                        dcc.Input(id='camera-axis-size-input',
                                  value=0.3,
                                  type='number', style={'width': '50%'}
                                 )
                    ]),
                ], style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                html.Label([ "sample marker size",
                    html.Div([
                        dcc.Input(id='sample-marker-size-input',
                                  value=2,
                                  type='number', style={'width': '50%'}
                                 )
                    ]),
                ], style = {'width' : '32%', 'float' : 'left', 'display' : 'inline-block'}),
            ]),
            
            # opacity 
            html.Div([
                html.Label([ "sphere opacity",
                    html.Div([
                        dcc.Input(id='sphere-opacity-input',
                                  value=0.2,
                                  type='number', style={'width': '50%'}
                                 )
                    ])
                ], style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                            
                html.Label([ "xy-plane opacity",            
                    html.Div([
                        dcc.Input(id='xy-plane-opacity-input',
                                  value=0.8,
                                  type='number', style={'width': '50%'}
                                 )
                    ])
                ],  style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                
            ]),
            
            # color
            html.Div([
                html.Label([ "camera color",
                html.Div([
                    dcc.Dropdown(id='camera-color-input',
                                 clearable=False,
                              value='yellow',
                              options=[
                                     {'label': c, 'value': c}
                                     for (c, _) in mcolors.CSS4_COLORS.items()
                                 ], style={'width': '80%'}
                             )
                ])
                ],  style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                
                html.Label([ "foreground color",
                    html.Div([
                        dcc.Dropdown(id='fg-color-input',
                                     clearable=False,
                                  value='plotly3',
                                  options=[
                                         {'label': c, 'value': c}
                                         for c in px.colors.named_colorscales()
                                     ], style={'width': '80%'}
                                 )
                    ])
                ],  style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                
                html.Label([ "background color",
                    html.Div([
                        dcc.Dropdown(id='bg-color-input',
                                     clearable=False,
                                  value='plotly3',
                                  options=[
                                         {'label': c, 'value': c}
                                         for c in px.colors.named_colorscales()
                                     ], style={'width': '80%'}
                                 )
                    ])
                ],  style = {'width' : '32%', 'float' : 'left', 'display' : 'inline-block'}),
            ]),
            
            # colorscale
            html.Div([
                html.Label([ "sphere colorscale",
                    html.Div([
                        dcc.Dropdown(id='sphere-colorscale-input',
                                     clearable=False,
                                     value='greys',
                                     options=[
                                         {'label': c, 'value': c}
                                         for c in px.colors.named_colorscales()
                                     ], style={'width': '80%'}
                                    )
                    ])
                ], style = {'width' : '32%', 'float' : 'left', 'display' : 'inline-block'}),
                
                html.Label([ "xy-plane colorscale",
                    html.Div([
                        dcc.Dropdown(id='xy-plane-colorscale-input',
                                     clearable=False,
                                     value='greys',
                                     options=[
                                         {'label': c, 'value': c}
                                         for c in px.colors.named_colorscales()
                                     ], style={'width': '80%'}
                                    )
                    ])
                ],  style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                
                html.Label([ " ",
                    html.Div([
                        dcc.Checklist(id='show-background-checklist',
                                      
                                     options=[
                                         {'label': 'show background', 'value': 'show_background'}
                                     ], style={'width': '80%'},
                                      value=[ ],
                                    )
                    ])
                ],  style = {'width' : '34%', 'float' : 'left', 'display' : 'inline-block'}),
                
            ]),
            
            
            
        ], style = {'width' : '48%', 'float' : 'right', 'display' : 'inline-block'}),
            
    ]),
    
])

@app.callback(
    Output('graph', 'figure'),
    Input("u-slider", "value"),
    Input("v-slider", "value"),
    
    Input("fov-slider", "value"),
    Input("foreground-near-depth-slider", "value"),
    
    Input("world-axis-size-input", "value"),
    Input("camera-axis-size-input", "value"),
    Input("sample-marker-size-input", "value"),
    
    Input("camera-color-input", "value"),
    Input("fg-color-input", "value"),
    Input("bg-color-input", "value"),
    
    Input('sphere-colorscale-input', "value"),
    Input('xy-plane-colorscale-input', "value"),
    Input('show-background-checklist', "value"),
       
    Input("sphere-opacity-input", "value"),
    Input("xy-plane-opacity-input", "value"),
)

def update_figure(u, v, 
                  fov, foreground_near_depth,
                  world_axis_size, camera_axis_size, sample_marker_size,
                  camera_color, fg_color, bg_color,
                  sphere_colorscale, xy_plane_colorscale, show_background,
                  sphere_opacity, xy_plane_opacity                  
                 ):
    
    depth_range = [foreground_near_depth, 2]
    
    # sphere
    fig = draw_sphere(r=1, sphere_colorscale=sphere_colorscale, sphere_opacity=sphere_opacity)

    # change figure size
#     fig.update_layout(autosize=False, width = 500, height=500)

    # draw axes in proportion to the proportion of their ranges
    fig.update_layout(scene_aspectmode='data')

    # xy plane
    fig = draw_XYplane(fig, xy_plane_colorscale, xy_plane_opacity,
                       x_range=[-depth_range[1], depth_range[1]], y_range=[-depth_range[1], depth_range[1]])

    # show world coordinate system (X, Y, Z positive direction)
    fig = draw_XYZworld(fig, world_axis_size=world_axis_size)

    pixels_world, camera_world, world_mat, fg_pts, bg_pts = nerfpp(u=u, v=v, fov=fov, depth_range=depth_range)

    #  draw camera at init (with its cooridnate system)
    fig = draw_cam_init(fig, world_mat, 
                        camera_axis_size=camera_axis_size, camera_color=camera_color)

    # draw foreground and background sample point, ray, and frustrum
    fig = draw_foreground(fig, fg_pts, fg_color, sample_marker_size, at=[0, -1])
    
    if show_background:
        fig = draw_background(fig, bg_pts.unsqueeze(0), bg_color, sample_marker_size, at=[0, -1])
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)