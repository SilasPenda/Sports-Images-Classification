import streamlit as st
from PIL import Image
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model

model = load_model('SIC.h5')


labels = {
    0: 'air hockey',
    1: 'ampute football',
    2: 'archery',
    3: 'arm wrestling',
    4: 'axe throwing',
    5: 'balance beam',
    6: 'barell racing',
    7: 'baseball',
    8: 'basketball',
    9: 'baton twirling',
    10: 'bike polo',
    11: 'billiards',
    12: 'bmx',
    13: 'bobsled',
    14: 'bowling',
    15: 'boxing',
    16: 'bull riding',
    17: 'bungee jumping',
    18: 'canoe slamon',
    19: 'cheerleading',
    20: 'chuckwagon racing',
    21: 'cricket',
    22: 'croquet',
    23: 'curling',
    24: 'disc golf',
    25: 'fencing',
    26: 'field hockey',
    27: 'figure skating men',
    28: 'figure skating pairs',
    29: 'figure skating women',
    30: 'fly fishing',
    31: 'football',
    32:	'formula 1 racing',
    33:	'frisbee',
    34:	'gaga',
    35:	'giant slalom',
    36:	'golf',
    37:	'hammer throw',
    38:	'hang gliding',
    39:	'harness racing',
    40:	'high jump',
    41:	'hockey',
    42:	'horse jumping',
    43:	'horse racing',
    44:	'horseshoe pitching',
    45:	'hurdles',
    46:	'hydroplane racing',
    47:	'ice climbing',
    48:	'ice yachting',
    49:	'jai alai',
    50:	'javelin',
    51:	'jousting',
    52:	'judo',
    53:	'lacrosse',
    54:	'log rolling',
    55:	'luge',
    56:	'motorcycle racing',
    57:	'mushing',
    58:	'nascar racing',
    59:	'olympic wrestling',
    60:	'parallel bar',
    61:	'pole climbing',
    62:	'pole dancing',
    63:	'pole vault',
    64:	'polo',
    65:	'pommel horse',
    66:	'rings',
    67:	'rock climbing',
    68:	'roller derby',
    69:	'rollerblade racing',
    70:	'rowing',
    71:	'rugby',
    72:	'sailboat racing',
    73:	'shot put',
    74:	'shuffleboard',
    75:	'sidecar racing',
    76:	'ski jumping',
    77:	'sky surfing',
    78:	'skydiving',
    79:	'snow boarding',
    80:	'snowmobile racing',
    81:	'speed skating',
    82:	'steer wrestling',
    83:	'sumo wrestling',
    84:	'surfing',
    85:	'swimming',
    86:	'table tennis',
    87:	'tennis',
    88:	'track bicycle',
    89:	'trapeze',
    90:	'tug of war',
    91:	'ultimate',
    92:	'uneven bars',
    93:	'volleyball',
    94:	'water cycling',
    95:	'water polo',
    96:	'weightlifting',
    97:	'wheelchair basketball',
    98:	'wheelchair racing',
    99:	'wingsuit flying'
}


def processed_img(imagepath):
    img = load_img(imagepath, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img/255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


def run():
    img_file = st.file_uploader('Choose an Image', type=['jpg', 'png'])
    if img_file is not None:
        img = Image.open(img_file).resize((250,250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, 'wb') as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            result = processed_img(save_image_path)
            print(result)
            st.success('This is '+result)

run()