# 🩺 Disease Classification with Deep Learning

This project focuses on **disease classification from medical images** using **deep learning models**.  
We experimented with multiple pretrained architectures and a CNN model from scratch, then deployed the **best-performing model (ResNet50)** using **Streamlit** for interactive predictions.

---

## 🚀 Project Overview

- **Goal:** Classify medical images into 7 disease categories:  
  `CaS, CoS, Gum, MC, OC, OLP, OT`  

- **Models Applied:**
  - 🧩 **ResNet50** ✅ (Best performing, used for deployment)
  - 🔬 **DenseNet121**
  - 📱 **MobileNet**
  - 🧠 **Vision Transformer (ViT)**
  - 🏗️ **CNN from scratch**

- **Deployment:**
  - Implemented with **Streamlit**  
  - Model weights stored on **Google Drive** and downloaded dynamically at runtime  
  - User uploads an image → model predicts **disease class + confidence + class probabilities**  

---

## 📂 Repository Structure

``` bash
├── deployment/ # Deployment app files
│ ├── app.py # Streamlit app for prediction
│ ├── requirements.txt # Required dependencies
│
├── pretrained_models/ # Pretrained model experiments
│ ├── ResNet50.ipynb
│ ├── DenseNet121.ipynb
│ ├── MobileNet.ipynb
│ ├── vit_model.ipynb
│
├── cnn_from_scratch.ipynb # CNN built from scratch
├── gum.jpeg # Sample test image
├── .gitignore
├── .gitattributes

```


---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ahmed-dawood10/Teeth_classification.git
   cd disease-classification/deployment
   ```

2. Create virtual environment (optional but recommended)

        python -m venv venv
        
        source venv/bin/activate     # On Linux/Mac
        
        venv\Scripts\activate        # On Windows

3. Install dependencies

        pip install -r requirements.txt

4.Run the app

        streamlit run app.py


## 🏆 Model Deployment

The trained model weights are stored on **Google Drive**.  
On first run, the Streamlit app automatically downloads the **best model (ResNet50)**.  
You can also access the weights of the other trained models below:

| Model            | Weights Link                                                                 |
|------------------|-------------------------------------------------------------------------------|
| 🧩 ResNet50 (Best, used in deployment) | [Download](https://drive.google.com/uc?id=1Gvs0ZuMX1UQi6SPaNhekh_C7jJ505r9N) |
| 🔬 DenseNet121   | [Download](https://drive.google.com/uc?id=11gvBNuEDsG-TeW0suH0pz-0NH88xB1Y2) |
| 📱 MobileNet     | [Download](https://drive.google.com/uc?id=1wFf8zw8qI1ocRv02lm9PvLzCraypjKI6) |
| 🧠 Vision Transformer (ViT) | [Download](https://drive.google.com/uc?id=1nT2zuvx6jvkcmujiFQ7-ZMrsB_QnCU2w) |
| 🏗️ CNN from scratch | [Download](https://drive.google.com/uc?id=1drxz-YdfFybW_01ZyWFRubysislr3qKa) |


---

### Example (ResNet50 in `app.py`)
```python
import gdown, os

model_file = "model.ResNet50.keras"

if not os.path.exists(model_file):
    url = "[https://drive.google.com/uc?id=YOUR_RESNET_FILE_ID](https://drive.google.com/uc?id=YOUR_RESNET_FILE_ID](https://drive.google.com/file/d/1Gvs0ZuMX1UQi6SPaNhekh_C7jJ505r9N/view?usp=drive_link)"
    gdown.download(url, model_file, quiet=False)

```

## 📊 Example Prediction

1. Upload an image (e.g., gum.jpeg)

2.The app displays:

    Predicted disease class
    
    Confidence score
    
    Probabilities for all classes with progress bars
## 🔮Results

📌 ResNet50 (Best Model)
Test Accuracy: 99.71%
Test Loss: 0.0067

ResNet50 achieved the best balance between accuracy and generalization, making it the primary choice for deployment.

Other models (DenseNet121, MobileNet, ViT, CNN from scratch) were also trained and compared for benchmarking.
