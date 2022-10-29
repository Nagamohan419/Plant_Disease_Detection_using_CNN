#import requirements
# Imporiting Necessary Libraries
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

#streamlit page configuration
st.set_page_config(
     page_title="Plant disease App",
     page_icon=":shark:",
     layout="wide"
 )

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2014/07/16/05/18/beach-394503_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

#loading the model
model= tf.keras.models.load_model(r'C:\Users\hp\Desktop\plant disease\object_classification\Model_resnet101V2.h5')

# Get the calss names programmtically
import pathlib
import numpy as np
data_dir = pathlib.Path(r"C:\Users\hp\Desktop\plant disease\data\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")])) # Created a list of class_names from the sundirectorie
df_class_names = pd.DataFrame(class_names, columns =["Disease Classification Classes"])

 #this path is the path where the uplaoded picture is stored
path = r'C:\Users\hp\Desktop\plant disease\object_classification\tested\out.jpg'
  

activities = ["About" ,"Image Segmentation","Plant Disease","Remedies"]
choice = st.sidebar.selectbox("Select Activty",activities)

def header(url):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)	
if choice =='About':
    st.title("Plant Disease Detection & Classification")
    st.markdown("An application that makes it easier for farmers, biologists, and botanists to identify plant or crop types and spot any problems in them. The software uploads the plant image to the server for analysis using the CNN classifier model. When a sickness is found, the user is shown the problem and the solutions.", unsafe_allow_html=True)
    st.header('Model')
    st.markdown('Trained to identify 38 classes for classification of 14 plants')
    #added the data of the disease name
    st.dataframe(df_class_names,1000,1400)
            
if choice == 'Image Segmentation':
    st.title("Image Segmentation Information")
    st.markdown("Image segmentation is a method in which a digital image is broken down into various subgroups called Image segments which helps in reducing the complexity of the image to make further processing or analysis of the image simpler.")
    st.header("Types of Image segmentation:")
    st.markdown("- Image Segmentation using edge detection")
    st.markdown("- Image Segmentation using K means")
    st.markdown("- Image Segmentation using Contour Detection")
    st.markdown("- Image Segmentation using Color Masking")

    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
      list-style-position: inside;
    }
</style>
''', unsafe_allow_html=True)

    image1 = Image.open(r'C:\Users\hp\Desktop\plant disease\images\img1.png')
    image2 = Image.open(r'C:\Users\hp\Desktop\plant disease\images\img2.png')
    image3 = Image.open(r'C:\Users\hp\Desktop\plant disease\images\img3.png')
    st.image(image1)
    st.image(image2)
    st.image(image3)

if choice == 'Plant Disease':
        st.title("Plant Disease Classification")
        st.write("Just Upload your Plant's Leaf Image and get predictions if the plant is healthy or not")
        # Setting the files that can be uploaded
        image_input = st.file_uploader("Upload Image",type=['jpg'])
        st.markdown("* * *")
        

        if image_input is None:
            st.text("Please upload an image file")
        else:
            image = Image.open(image_input)
            image = image.resize((200, 200))
            st.image(image)
            img_array = np.array(image)
            img = tf.image.resize(img_array, size=(256, 256))
            img = img/255.0
            img = tf.expand_dims(img, axis=0)
            st.write("filename:", image_input.name)
            st.markdown("* * *")
            if st.button('Predict'):
                 with st.spinner('Your image is processing'):
                    prediction = model.predict(img)
                    st.success('Done!')
                    st.write(prediction.shape)
                    st.write(prediction)
                    st.write("Prediction_Class: ", class_names[np.argmax(prediction)])

if choice == 'Remedies':
        st.title("Remedies for the diseases")
        data={
            'Diseases':['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Cherry_(including_sour)___Powdery_mildew','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Pepper,_bell___Bacterial_spot','Potato___Early_blight','Potato___Late_blight','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus'],
            'Remedies': ['Treated by the fungicide portion of an all-purpose fruit tree spray, not the insecticide portion, so a fungicide-only spray is all you need','Pick all dried and shriveled fruits remaining on the trees.Remove infected plant material from the area.All infected plant parts should be burned, buried or sent to a municipal composting site.','Spraying apple trees with copper can be done to treat cedar apple rust and prevent other fungal infections,Neem oil and horticultual oils are also safe and effective in controling rust on apple and crabapple trees','Avoid early irrigation as this may cause premature powdery mildew infections to rise,Pruning your cherry tree will encourage airflow and leaf dryness.','Products containing chlorothalonil, myclobutanil or thiophanate-methyl are most effective when applied prior to or at the first sign of leaf spots,Two products that seem to work well against Cercospora diseases are SYSTHANE (myclobutanil) and HERITAGE (azoxystrobin),Management strategies for gray leaf spot include tillage, crop rotation and planting resistant hybrids',' If the corn begins to show symptoms of infection, immediately spray with a fungicide,Mancozeb as a protectant and myclobutanil as an eradicant are generally effective against all rusts while triadimefon is effective against only specific rusts,The best management practice is to use resistant corn hybrids.','A one-year rotation away from corn, followed by tillage is recommended to prevent disease development in the subsequent corn crop,Prune or stake plants to improve air circulation and reduce fungal problems,Drip irrigation and soaker hoses can be used to help keep the foliage dry.','Sanitation is extremely important. Destroy mummies, remove diseased tendrils from the wires, and select fruiting canes without lesions','Presently, there are no effective management strategies for measles. Wine grape growers with small vineyards will often have field crews remove infected fruit prior to harvest.',' All the fungicides/botanicals were found significantly superior over the control in reducing the disease intensity.','It is incredibly important to remove trees that have citrus greening disease,In citrus-producing areas with little or no HLB incidence, early detection and removal of infected trees are critical to prevent spread of the disease,Inspect trees for the Asian citrus psyllid and Huanglongbing monthly, and whenever watering, spraying, pruning or tending trees.','Bactericides are available for suppression of bacterial spot,Removal of gummy-blackened branch tips typical of bacterial spot during spring pruning may help somewhat to reduce inoculum levels.','Treat seeds by soaking them for 2 minutes in a 10% Chlorine bleach solution,Avoid overwatering.','Early blight can be minimized by maintaining optimum growing conditions, including proper fertilization, irrigation, and management of other pests,Mancozeb and chlorothalonil are perhaps the most frequently used protectant fungicides for early blight management,Prune or stake plants to improve air circulation and reduce fungal problems.',' Late blight is controlled by eliminating cull piles and volunteer potatoes, using proper harvesting and storage practices and applying fungicides when necessary,Acrobat used later in the season reduces late blight spores. Use just before topkilling if there is blight in the crop.','Sulfur and “Stylet” oil are effective products for powdery mildew control. Fixed copper fungicides have also shown results in managing powdery mildew. Neem oil is also an effective combatant for managing powdery mildew.','Remove infected leaves and debris. Increase air circulation to encourage leaf drying. Consider resistant cultivars,Prevention of scorch needs to begin with winter watering.','Gardeners should always use either a sterilized soil medium or one that is commercially made. Practice crop rotation','Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable.	Do not spray pesticides, fungicides, fertilizers, or herbicides when it is in the high 80 or 90; you can damage your plants.','Spraying fungicides is the most effective way to prevent late blight.','When planting, use only certified disease-free seeds or treated seeds,Remove and destroy all crop debris post-harvest,Sanitize the greenhouse between crop seasons.','Remove diseased leaves. If caught early, the lower infected leaves can be removed and burned or destroyed.Improve air circulation around the plants.Mulch around the base of the plants.Do not use overhead watering','Check plants regularly for spider mites.Keep plants healthy. Spider mites thrive on plants under stress.Physically remove them. Use a high-pressure water spray to dislodge twospotted spider mites.','Many fungicides are registered to control target spots on tomatoes.Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spots in research trials.','Spread reflective mulch around your plants to repel whiteflies.,Place row covers over your plants to prevent whitefly infections.Avoid importing plants from virus-prone areas.','To avoid seed-borne mosaic viruses, soak the seeds of susceptible plants in a 10% Bleach solution before planting.You can try covering your plants with a floating row cover or aluminum foil mulches to prevent these insects from infecting your plants.']
        }
        df= pd.DataFrame(data)
        # st.dataframe(df,1200,1400)
        st.table(df)

