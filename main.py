import numpy as np
import os
import midi
import base64
import pickle
import time

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input as dash_Input
from dash.dependencies import Output, State
from dash_bootstrap_components import ListGroupItem

from keras.models import load_model

from model import *
from google.cloud import storage

MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_FILENAME = os.environ['MODEL_FILENAME']
OUTPUT_DIR = 'output/'

models = build_models()
models[0].summary()
client = storage.Client()
bucket = client.get_bucket(MODEL_BUCKET)
blob = bucket.get_blob(MODEL_FILENAME)
s = blob.download_as_string()
models[0].load_weights(s)
#OUTPUT_DIR = blob.upload_from_filename('myfile')

app = dash.Dash(external_stylesheets=['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/lux/bootstrap.min.css'])

composing = 0

#open('generation_progress.txt', 'w').write('0')

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

'''
composition_card_content = [
    
    dbc.CardHeader("Compositions"),
    
    dbc.CardBody(
        [
            html.P(id='selected-styles'),
            html.P(id='selected-nb-bars'),
            html.P(id='ref-selected'),
            html.P(id='generate-output'),
            html.P(id='can-download'),
            #dcc.Interval(id="progress-interval", n_intervals=0, interval=1500),
            #dbc.Progress(id="progress", color='warning', striped=True, animated=True),
            dbc.Button(
                [" Download"],
                color="primary",
                disabled=False,
                target='_blank',
                id='download-button'
            ),
        ]
    ),
]
'''


style_cards = [
    dbc.CardHeader("Select style"),
    
    dbc.CardBody(
        [
        html.P('Select one or more styles by assigning weights.'),
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
                                        dbc.Input(placeholder='file name...', id='file-name'),
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
                                            children=dbc.Input(id='upload-name', placeholder='browse...',size=73.5),
                                            #style={'height': '90%', 'width': '100%'}
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                            dbc.Button("Compose", color="primary", className="mr-1", id='compose-clicked'),
                            html.Div(id='hidden-div', style={'display': 'none'}), 
                            dbc.Toast(
                                id="positioned-toast",
                                header="Status",
                                is_open=False,
                                dismissable=True,
                                icon="warning",
                                # top: 66 positions the toast below the navbar
                                style={"position": "fixed", "top": 66, "right": 10, "width": 350},
                                duration=4000
                            ),
                            dcc.Interval(id="progress-interval", n_intervals=0, interval=1500),
                                ]
                            ),
                    ],
                    className="mb-4",
                )
        ]
    ),
]

composition_card_content = [
    
    dbc.CardHeader("Compositions"),
    
    dbc.CardBody(
        [
            dbc.ListGroup(id='compositions-list')
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

'''
@app.callback([Output("selected-styles", "children")], 
              [dash_Input("mozart-select", "checked"),
               dash_Input("mozart-weight", "value"),
               dash_Input("haydn-select", "checked"),
               dash_Input("haydn-weight", "value"),
               dash_Input("beethoven-select", "checked"),
               dash_Input("beethoven-weight", "value"),
               dash_Input("chopin-select", "checked"),
               dash_Input("chopin-weight", "value"),
               dash_Input("liszt-select", "checked"),
               dash_Input("liszt-weight", "value"),
               dash_Input("tchaikovsky-select", "checked"),
               dash_Input("tchaikovsky-weight", "value")],)
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
              [dash_Input("nb-bars", "value")],)
def file_name(nb_bars):
    
    if nb_bars is None:
        nb_bars=4
    return [f'Number of bars selected: {nb_bars}']


@app.callback([Output("output-midi-filename", "children")], 
              [Input("file-name", "value")],)
def selected_num_bars(filename):
    
    output_name = filename
    return [f'Output file name: {filename}']
    
@app.callback([Output("progress", "value"), 
               Output("progress", "children")],
              [dash_Input("progress-interval", "n_intervals")],
)
def update_progress(n):
    generation_progress = int(open('generation_progress.txt', 'r').read())
    print("Calling progress:", generation_progress)
    return generation_progress, f"{generation_progress} %"    
    
    


@app.callback([Output("download-button", "href"),
               Output("download-button", "disabled"),
               Output("can-download", "children"),], 
              [dash_Input("file-name", "value")],)
def download_option(filename):
    
    print("Download option callback initiated")
    
    if filename is not None:
        downloadable_files = os.listdir('output')
        for f in downloadable_files:
            if filename in f.split('.'):
                return f'file:///C:/Users/sunny/Downloads/Dash UI/output/{filename}.mid', False, f'{filename}.mid is available for download'
            else:
                return '', True, f'{filename}.mid is not yet generated'
    else:
        return '', True, f'{filename}.mid is not yet generated'
       
'''
@app.callback([Output("compose-clicked", "children"), 
               Output("compose-clicked", "disabled"), ],
              [dash_Input("progress-interval", "n_intervals")],
)
def update_compose_btn(n):
    global composing
    
    if composing==1:
        return [dbc.Spinner(size="sm"), " Composing..."], True
    else:
        return "Compose", False
    
@app.callback([Output("upload-name", "placeholder")],
              [dash_Input("upload_id", "contents"),
               dash_Input("upload_id", "filename")], prevent_initial_call=True)
def uploaded(contents, filename):
    
    midi_data = ''.join(contents.split(',')[1:])
    midi_data = midi_data.encode('utf-8')

    f = open('samples/sample.mid','wb')
    f.write(base64.decodebytes(midi_data))
    f.close()
    return [f"{filename}"]


@app.callback([Output("compositions-list", "children")], 
              [dash_Input("compose-clicked", "n_clicks")],)
def download_option(clicks):
    
    print("Download option callback initiated")
    global composing
    if composing==0:
        files = os.listdir('output')
        midi_files = []
        for file in files:
            if file.split('.')[1]=='mid':
                midi_files.append(ListGroupItem(f"{file}", href=f'file:///C:/Users/sunny/Downloads/Dash UI/output/{file}'))
    
        if len(midi_files)==0:
            return [dbc.ListGroupItemText("No compositions to download.")]
        else:
            return [midi_files]
        
@app.callback([Output("positioned-toast", "children"),
               Output("positioned-toast", "is_open")], 
              [dash_Input("compose-clicked", "n_clicks"), 
               State("file-name", "value"), 
               State("upload_id", "contents"),
               State("nb-bars", "value"),
               State("mozart-select", "checked"),
               State("mozart-weight", "value"),
               State("haydn-select", "checked"),
               State("haydn-weight", "value"),
               State("beethoven-select", "checked"),
               State("beethoven-weight", "value"),
               State("chopin-select", "checked"),
               State("chopin-weight", "value"),
               State("liszt-select", "checked"),
               State("liszt-weight", "value"),
               State("tchaikovsky-select", "checked"),
               State("tchaikovsky-weight", "value")], prevent_initial_call=True)
def generate_sequence(compose_clicked, filename, uploaded_ref, nb_bars,
                      mozart_checked, mozart_weight, 
                      haydn_checked, haydn_weight, 
                      beethoven_checked, beethoven_weight, 
                      chopin_checked, chopin_weight, 
                      liszt_checked, liszt_weight, 
                      tchaikovsky_checked, tchaikovsky_weight):
    
    '''
    compose_clicked = 1
    filename = 'utest_120602'
    uploaded_ref = None
    nb_bars = 4
    mozart_checked = None
    mozart_weight = None
    haydn_checked = None
    haydn_weight = None
    beethoven_checked = None 
    beethoven_weight = None
    chopin_checked = None
    chopin_weight = None
    liszt_checked = None
    liszt_weight = None
    tchaikovsky_checked = None
    tchaikovsky_weight = None
    '''
    global composing
    
    composing = 1
    
    if compose_clicked is not None:
        
        print('Setting up generation')
        #sample_seq = np.load('samples/thunderstruck.npy')
        #styles_gen = [np.mean([one_hot(i, NUM_STYLES) for i in [3]], axis=0)]
        
        print('Selection status:', mozart_checked, mozart_weight, 
                      haydn_checked, haydn_weight, 
                      beethoven_checked, beethoven_weight, 
                      chopin_checked, chopin_weight, 
                      liszt_checked, liszt_weight, 
                      tchaikovsky_checked, tchaikovsky_weight, '\n\n\n')
        
        start_time = time.time()
        styles_gen = []
        if beethoven_checked:
            styles_gen.append(beethoven_weight/100)
        else:
            styles_gen.append(0)
        if haydn_checked:
            styles_gen.append(haydn_weight/100)
        else:
            styles_gen.append(0)
        if mozart_checked:
            styles_gen.append(mozart_weight/100)
        else:
            styles_gen.append(0)
        if chopin_checked:
            styles_gen.append(chopin_weight/100)
        else:
            styles_gen.append(0)
        if liszt_checked:
            styles_gen.append(liszt_weight/100)
        else:
            styles_gen.append(0)
        if tchaikovsky_checked:
            styles_gen.append(tchaikovsky_weight/100)
        else:
            styles_gen.append(0)
        
        nb_bars_selected = nb_bars
        if nb_bars is None:
            nb_bars_selected = 4

        print("Styles selected", sum(styles_gen), styles_gen, liszt_checked, filename, '...')

        if sum(styles_gen)>0:
            
            if filename is not None:
                styles_gen = np.array(styles_gen)/sum(styles_gen)
                if uploaded_ref is not None:
                    try:
                        
                        sample_seq = midi.read_midifile('samples/sample.mid')
                        sample_seq = midi_decode(sample_seq)
                        temp = sample_seq.shape[0]//256
                        if temp>2:
                            temp = temp*128
                            sample_seq = sample_seq[temp:temp+256, :, :]
                        elif temp>=1:
                            sample_seq = sample_seq[-256:, :, :]
                        else:
                            print('Require longer reference file')
                            composing = 0
                            return ['Please provide a bigger reference file, with atleast 16 bars', True]
                        
                        sample_seq = clamp_midi(sample_seq)
                    
                        results = zip(*list(generate_with_seq(models, nb_bars_selected, styles_gen, sample_seq)))
                    except Exception as e:
                        composing = 0
                        return [f'{e}', True]
                else:
                    results = zip(*list(generate(models, nb_bars_selected, [styles_gen])))
                
                pickle.dump(results, open(OUTPUT_DIR+f'{filename}.pkl', 'wb'))
            
                results = pickle.load(open(OUTPUT_DIR+f'{filename}.pkl', 'rb'))
                
                for i, result in enumerate(results):
                    fpath = os.path.join(OUTPUT_DIR, f'{filename}.mid')
                    print('Writing file', fpath)
                    os.makedirs(os.path.dirname(fpath), exist_ok=True)
                    mf = midi_encode(unclamp_midi(result))
                    midi.write_midifile(fpath, mf)
                    blob.upload_from_filename(OUTPUT_DIR, f'{filename}.mid')
                
                print('Composition Done')
                
                time_taken = int(time.time() - start_time)
                
                composing = 0
            
                return [f'Composition of {filename} has been completed in {time_taken} seconds', True]
            else:
                composing = 0
                return ['Please fill name', True]
        else:
            composing = 0
            return ['Please select a style.', True]


'''
@app.callback(Output('output-image2', 'children'),
              [Input('upload_id', 'contents')])
def load_img(contents):
    open('uploaded_image.txt','w').write(contents)
    new_img = open('uploaded_image.txt','r').read()
    return html.Div([html.P(predict_caption(new_img))])
'''


#sample_seq = np.load('samples/thunderstruck.npy')

#styles_gen = [np.mean([one_hot(i, NUM_STYLES) for i in [3]], axis=0)]
#results = zip(*list(generate_with_seq(models, 8, styles_gen[0], sample_seq)))

#results = zip(*list(generate(models, 8, styles_gen)))





if __name__ == "__main__":
    #app.run_server(port=8888) - This was from Sahith
    app.run(host='127.0.0.1', port=8080, debug=True)
