import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import base64

app = dash.Dash(external_stylesheets=['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/sketchy/bootstrap.min.css'])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

NAVBAR_STYLE = {
    "font-size": "25px",
}


navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(dbc.NavbarBrand("SVARA.AI", style=NAVBAR_STYLE, )), #className="ml-1"
                ],
                align="center",
                no_gutters=True,
            ),
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        #dbc.Collapse(search_bar, id="navbar-collapse", navbar=True),
    ],
    color="dark",
    dark=True,
)

composition_card_content = [
    
    dbc.CardHeader("Composition"),
    
    dbc.CardBody(
        [
            html.P(id='selected-styles'),
            html.P(id='selected-nb-bars'),
            html.P(id='ref-selected'),
            html.H5("Work in progress...", className="card-title"),
            dbc.Button("Play", color="primary"),
            dbc.Button("Download", color="primary"),
        ]
    ),
]

style_cards = [
    dbc.CardHeader("Select style"),
    
    dbc.CardBody(
        [
        html.P('Select one or more styles by assigning weights, summing upto 100.'),
        dbc.CardGroup(
            [
                dbc.Card(
                        [
                            dbc.CardImg(src='data:image/png;base64,{}'.format(base64.b64encode(open('images/Haydn.jpg', 'rb').read()).decode()), top=True),
                            dbc.CardBody(
                                [
                                    html.H6('Joseph Haydn', style={'font-size': '1em', 'font-weight': 'bold'}, className="card-title"),
                        
                                    dbc.Row(
                                        [
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupAddon(dbc.Checkbox(id='haydn-select')),
                                                    dbc.Input(type='number', min=0, max=100, id='haydn-weight'),
                                                ]
                                            ),
                                        ]
                                    ),
                                    #dbc.Button("Select", color="primary"),
                                    #dbc.Input(placeholder="Select weight",type="number", min=0, max=10, step=5),
                                ]
                                ),   
                        ], color="primary", inverse=True),
                dbc.Card(
                        [
                            dbc.CardImg(src='data:image/png;base64,{}'.format(base64.b64encode(open('images/Mozart.jpg', 'rb').read()).decode()), top=True),
                            dbc.CardBody(
                                [
                                    html.H6('Wolfgang Amadeus Mozart', style={'font-size': '1em', 'font-weight': 'bold'}, className="card-title"),
                        
                                    dbc.Row(
                                        [   
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupAddon(dbc.Checkbox(id='mozart-select')),
                                                    dbc.Input(type='number', min=0, max=100, id='mozart-weight'),
                                                ]
                                            ),
                                        ]
                                    ),
                                    #dbc.Button("Select", color="primary"),
                                    #dbc.Input(placeholder="Select weight",type="number", min=0, max=10, step=5),
                                ]
                                ),   
                        ], color="success", inverse=True),
                dbc.Card(
                        [
                            dbc.CardImg(src='data:image/png;base64,{}'.format(base64.b64encode(open('images/Beethoven.jpg', 'rb').read()).decode()), top=True),
                            dbc.CardBody(
                                [
                                    html.H6('Ludwig van Beethoven', style={'font-size': '1em', 'font-weight': 'bold'}, className="card-title"),
                        
                                    dbc.Row(
                                        [
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupAddon(dbc.Checkbox(id='beethoven-select')),
                                                    dbc.Input(type='number', min=0, max=100, id='beethoven-weight'),
                                                ]
                                            ),
                                        ]
                                    ),
                                    #dbc.Button("Select", color="primary"),
                                    #dbc.Input(placeholder="Select weight",type="number", min=0, max=10, step=5),
                                ]
                                ),   
                        ], color="info", inverse=True),
                dbc.Card(
                        [
                            dbc.CardImg(src='data:image/png;base64,{}'.format(base64.b64encode(open('images/Chopin.jpg', 'rb').read()).decode()), top=True),
                            dbc.CardBody(
                                [
                                    html.H6('Frédéric Chopin', style={'font-size': '1em', 'font-weight': 'bold'}, className="card-title"),
                        
                                    dbc.Row(
                                        [
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupAddon(dbc.Checkbox(id='chopin-select')),
                                                    dbc.Input(type='number', min=0, max=100, id='chopin-weight'),
                                                ]
                                            ),
                                        ]
                                    ),
                                    #dbc.Button("Select", color="primary"),
                                    #dbc.Input(placeholder="Select weight",type="number", min=0, max=10, step=5),
                                ]
                                ),   
                        ], color="warning", inverse=True),
                dbc.Card(
                        [
                            dbc.CardImg(src='data:image/png;base64,{}'.format(base64.b64encode(open('images/Liszt.jpg', 'rb').read()).decode()), top=True),
                            dbc.CardBody(
                                [
                                    html.H6('Franz Liszt', style={'font-size': '1em', 'font-weight': 'bold'}, className="card-title"),
                        
                                    dbc.Row(
                                        [
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupAddon(dbc.Checkbox(id='liszt-select')),
                                                    dbc.Input(type='number', min=0, max=100, id='liszt-weight'),
                                                ]
                                            ),
                                        ]
                                    ),
                                    #dbc.Button("Select", color="primary"),
                                    #dbc.Input(placeholder="Select weight",type="number", min=0, max=10, step=5),
                                ]
                                ),   
                        ], color="dark", inverse=True),
                dbc.Card(
                        [
                            dbc.CardImg(src='data:image/png;base64,{}'.format(base64.b64encode(open('images/Tchaikovsky.jpg', 'rb').read()).decode()), top=True),
                            dbc.CardBody(
                                [
                                    html.H6('Pyotr Ilyich Tchaikovsky', style={'font-size': '1em', 'font-weight': 'bold'}, className="card-title"),
                        
                                    dbc.Row(
                                        [
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupAddon(dbc.Checkbox(id='tchaikovsky-select')),
                                                    dbc.Input(type='number', min=0, max=100, id='tchaikovsky-weight'),
                                                ]
                                            ),
                                        ]
                                    ),
                                    #dbc.Button("Select", color="primary"),
                                    #dbc.Input(placeholder="Select weight",type="number", min=0, max=10, step=5),
                                ]
                                ),   
                        ], color="danger", inverse=True),
            ]
        )
        ]
    ),
]

more_options = [
    dbc.CardHeader("Options"),
    dbc.CardBody(
        [
            dbc.Row(
                    [
                        dbc.Col([
                            dbc.InputGroup(
                                    [
                                        dbc.InputGroupAddon("Name", addon_type="prepend"),
                                        dbc.Input(placeholder='file name...'),
                                    ],
                                    className="mb-3",
                                ),
                            dbc.InputGroup(
                                    [
                                        dbc.InputGroupAddon("Number of Bars", addon_type="prepend"),
                                        dbc.Input(id='nb-bars', placeholder=4, type="number", min=4, max=256, step=4),
                                    ],
                                    className="mb-3",
                                ),
                            dbc.InputGroup(
                                    [
                                        dbc.InputGroupAddon("Upload reference", addon_type="prepend"),
                                        dcc.Upload(
                                            id='upload_id',
                                            children=dbc.Input(placeholder='browse...',size=73.5),
                                            #style={'height': '90%', 'width': '100%'}
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                            dbc.Button("Compose", color="primary", className="mr-1"),
                                ]
                            ),
                    ],
                    className="mb-4",
                )
        ]
    ),
]

gen_layout = html.Div(
    [
        navbar,
        dbc.Row(
            [
                dbc.Col(dbc.Card(style_cards, color="light", inverse=False)),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(more_options, color="light", inverse=False)),
                dbc.Col(dbc.Card(composition_card_content, color="light", inverse=False)),

            ],
            className="mb-4",
        ),
    ]
)


#app.layout = html.Div([dcc.Location(id="url"), gen_layout, content])
app.layout = html.Div(gen_layout)

@app.callback([Output("selected-styles", "children")], 
              [Input("mozart-select", "checked"),
               Input("mozart-weight", "value"),
               Input("haydn-select", "checked"),
               Input("haydn-weight", "value"),
               Input("beethoven-select", "checked"),
               Input("beethoven-weight", "value"),
               Input("chopin-select", "checked"),
               Input("chopin-weight", "value"),
               Input("liszt-select", "checked"),
               Input("liszt-weight", "value"),
               Input("tchaikovsky-select", "checked"),
               Input("tchaikovsky-weight", "value")],)
def selected_weights(mozart_checked, mozart_weight, 
                     haydn_checked, haydn_weight, 
                     beethoven_checked, beethoven_weight, 
                     chopin_checked, chopin_weight, 
                     liszt_checked, liszt_weight, 
                     tchaikovsky_checked, tchaikovsky_weight):
    print('Weight callback initiated')
    
    output_text = 'Styles selected: '
    if haydn_checked:
        output_text += f'Haydn: {haydn_weight}%, '
    if mozart_checked:
        output_text += f'Mozart: {mozart_weight}%, '
    if beethoven_checked:
        output_text += f'Beethoven: {beethoven_weight}%, '
    if chopin_checked:
        output_text += f'Chopin: {chopin_weight}%, '
    if liszt_checked is not None:
        output_text += f'Liszt: {liszt_weight}%, '
    if tchaikovsky_checked is not None:
        output_text += f'Tchaikovsky: {tchaikovsky_weight}%, '
    if output_text=='':
        output_text = 'Styles selected: None'
    
    return [output_text]


@app.callback([Output("selected-nb-bars", "children")], 
              [Input("nb-bars", "value")],)
def selected_num_bars(nb_bars):
    
    return [f'Number of bars selected: {nb_bars}']


@app.callback([Output("ref-selected", "children")], 
              [Input("upload_id", "contents")],)
def selected_num_bars(ref_selected):
    
    return [f'Reference used: {ref_selected}']

if __name__ == "__main__":
    app.run_server(port=8888)