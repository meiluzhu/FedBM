# source: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb

Kvasir_BASIC_TEMPLATES = ['this is an image of {}']

Kvasir_TEMPLATES= [
    "an image of {}.",
    "a photo of {}.",
    "a endoscopic image of {}.",
    "a endoscopic photo of {}.",
    
    "this is a photo of {}.",
    "this is an image of {}.",
    "this is a endoscopic photo of {}.",
    "this is a endoscopic image of {}.",    

    "{} presented in photo.",
    "{} presented in image.",
    "{} presented in endoscopic photo.",
    "{} presented in endoscopic image.",
    
    "the image shows {}.",
    "the photo shows {}.",
    "the endoscopic image shows {}.",
    "the endoscopic photo shows {}.",
    
    "the image shows the presence of {}.",
    "the photo shows the presence of {}.",
    "the endoscopic image shows the presence of {}.",
    "the endoscopic photo shows the presence of {}.",
    
    "the presence of {} in image.",
    "the presence of {} in photo.",
    "the presence of {} in endoscopic image.",
    "the presence of {} in endoscopic photo.",
]


##https://arxiv.org/pdf/2303.00915v1.pdf
OCT_BASIC_TEMPLATES = ['this is an image of {}']

OCT_TEMPLATES = [
    "an image of {}.",
    "a photo of {}.",
    "an OCT scan of {}.",
    
    "this is a photo of {}.",
    "this is an image of {}.",
    "this is an OCT scan of {}.",
    "this is an OCT photo of {}.",
    "this is an OCT image of {}.",
    
    "{} presented in photo.",
    "{} presented in image.",
    "{} presented in OCT scan.",
    "{} presented in OCT image.",
    "{} presented in OCT photo.",
    
    "the image shows {}.",
    "the photo shows {}.",
    "the OCT scan shows {}.",
    "the OCT image shows {}.",
    "the OCT photo shows {}.",
    
    "the image shows the presence of {}.",
    "the photo shows the presence of {}.",
    "the OCT scan shows the presence of {}.",
    "the OCT image shows the presence of {}.",
    "the OCT photo shows the presence of {}.",
    
    "the presence of {} in image.",
    "the presence of {} in photo.",
    "the presence of {} in OCT image.",
    "the presence of {} in OCT photo.",
    "the presence of {} in OCT scan.",
]
