import logging
from ipywidgets import Output, HBox, Text, Label, Box, VBox, RadioButtons, Button, Layout
from IPython.display import display
from datetime import datetime
import os
from functools import partial

def set_alpha(alpha, alpha_):
    alpha_[0] = alpha

def set_beta(beta, beta_):
    beta_[0] = beta
    
def set_freq_pen(freq_penalty, freq_pen):
    freq_pen[0] = freq_penalty

def set_hub_word_bias(hub_word_bias, hub_bias):
    hub_bias[0] = hub_word_bias

def set_input_bar():
    inp = Text(placeholder='a dark house, broken window, weird voice', description='Keywords:', layout = Layout(width='18cm'))
    button = Button(description='Run!',
                    layout = {'width':'3cm'},
                    tooltip='Click me')
    button.style.button_color = 'lightgreen'
    box = HBox([inp, button])
    return box, inp, button

class OutputWidgetHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {
            'width': '85%',
            'height': '300px',
            'border': '1px solid black',
            'margin': '10px 50px 10px 50px',
            'overflow': 'hidden scroll'
        }
        self.out = Output(layout=layout)

    def emit(self, record):
        """ Overload of logging.Handler method """
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record+'\n'
        }
        # self.out.outputs = (new_output, ) + self.out.outputs  # new outputs are one the top
        self.out.outputs = self.out.outputs + (new_output, )

    def show_logs(self):
        """ Show the logs """
        display(self.out)

    def clear_logs(self):
        """ Clear the current logs """
        # self.out.clear_output()
        self.out.outputs = []


def set_logging():
    if not os.path.isdir('logs/'):
        os.mkdir('logs/')
    filename = datetime.now().strftime('log_%Y%m%d_%H_%M.log')

    logFormatter = logging.Formatter("%(levelname)s-%(asctime)s  %(message)s")
    logger = logging.getLogger("requests")
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler('logs/'+filename)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    # consoleHandler = logging.StreamHandler(sys.stdout)
    # consoleHandler.setFormatter(logging.Formatter(""))
    # logger.addHandler(consoleHandler)

    widgetsHandler = OutputWidgetHandler()
    widgetsHandler.setFormatter(logging.Formatter(None))
    logger.addHandler(widgetsHandler)

    return widgetsHandler, logger

def set_outputBox():
    lal1 = Label('Steered Story')
    lal2 = Label('GPT3 Story')

    box1_label = Box([lal1], layout={'width': '10cm',
                                    'margin': '10px 25px 0px 50px'})
    box2_label = Box([lal2], layout={'width': '10cm',
                                    'margin': '10px 50px 0px 25px'})
    
    
    box1_layout= {
            'width': '10.4cm',
            'height': '400px',
            'border': '1px solid black',
            'margin': '5px 25px 10px 50px',
            'overflow': 'hidden scroll'
        }
    box2_layout= {
            'width': '10.4cm',
            'height': '400px',
            'border': '1px solid black',
            'margin': '5px 50px 10px 25px',
            'overflow': 'hidden scroll'
        }
    
    box1 = Output(layout=box1_layout)
    box2 = Output(layout=box2_layout)

    vbox1 = VBox([box1_label, box1])
    vbox2 = VBox([box2_label, box2])
    boxes = HBox([vbox1, vbox2])

    return boxes, box1, box2

def set_imageBox():
    imageLabel = Label('Generated Keywords Knowledge Graph:')
    box_imageLabel = Box([imageLabel], layout={'width': '10cm',
                                               'margin': '15px 50px 0px 50px'})
    imageOutPut = Output(layout = {
                                'width': '20cm',
                                'height': '600px',
                                'border': 'white',
                                'margin': '0px 50px 0px 50px'
                            })
    imageBoxes = VBox([box_imageLabel, imageOutPut])
    return imageBoxes, imageOutPut

prompt_button = RadioButtons(
                    options=['text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-002'],
                    value= 'text-curie-001', # Defaults to 'pineapple'
                    layout={'description_width': 'initial'}, # If the items' names are long
                    description='Prompt:',
                    disabled=False
                    )

story_button = RadioButtons(
                    options=['text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-002'],
                    value= 'text-curie-001', # Defaults to 'pineapple'
                    layout={'description_width': 'initial'}, # If the items' names are long
                    description='Story:',
                    disabled=False
                    )

def change_prompt_model(model, change):
    if change['type'] == 'change' and change['name'] == 'value':
        model[0] = change['new']

def change_story_model(model, change):
    if change['type'] == 'change' and change['name'] == 'value':
        model[0] = change['new']

def set_models(prompt_model, story_model):
    prompt_button.observe(partial(change_prompt_model, prompt_model))
    story_button.observe(partial(change_story_model, story_model))
    return prompt_button, story_button

def show_parameters(logger, alpha_, beta_, freq_pen, prompt_model, story_model, hub_word_bias):
    logger.info('-----Parameters-----')
    logger.info(f"alpha = {alpha_}")
    logger.info(f"beta = {beta_}")
    logger.info(f"freq_penalty = {freq_pen}")
    logger.info(f"hub_word_bias = {hub_word_bias}")
    logger.info(f"prompt_model = \"{prompt_model}\"")
    logger.info(f"story_model = \"{story_model}\"")
    logger.info("")