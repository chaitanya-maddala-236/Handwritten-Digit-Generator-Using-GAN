import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .digit-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin: 20px 0;
    }
    .generated-image {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 5px;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator Network
class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.label_emb = nn.Embedding(num_classes, noise_dim)
        
        # Generator layers
        self.model = nn.Sequential(
            nn.Linear(noise_dim * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Embed labels
        label_emb = self.label_emb(labels)
        # Concatenate noise and label embedding
        gen_input = torch.cat([noise, label_emb], dim=1)
        # Generate image
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img

@st.cache_resource
def load_model():
    """Load the trained generator model"""
    generator = Generator().to(device)
    try:
        generator.load_state_dict(torch.load('generator.pth', map_location=device))
        generator.eval()
        return generator, True
    except FileNotFoundError:
        st.warning("⚠️ Trained model not found. Using mock generation for demonstration.")
        return generator, False

def generate_mock_images(digit, num_samples=5):
    """Generate mock images when trained model is not available"""
    images = []
    
    for i in range(num_samples):
        # Create a 28x28 image
        img = np.ones((28, 28)) * 0.1  # Light background
        
        # Add some randomness to make each image unique
        np.random.seed(digit * 100 + i)  # Ensure reproducible but varied results
        
        # Create digit patterns
        center_x, center_y = 14, 14
        
        if digit == 0:
            # Draw oval
            y, x = np.ogrid[:28, :28]
            mask = ((x - center_x)/8)**2 + ((y - center_y)/10)**2 <= 1
            img[mask] = 0.8 + np.random.normal(0, 0.1, np.sum(mask))
            inner_mask = ((x - center_x)/5)**2 + ((y - center_y)/7)**2 <= 1
            img[inner_mask] = 0.1
            
        elif digit == 1:
            # Draw vertical line with slight angle
            angle_offset = (i - 2) * 2  # Vary angle slightly
            for y in range(6, 22):
                x_pos = center_x + angle_offset + int((y - 14) * 0.1)
                if 0 <= x_pos < 28:
                    img[y, max(0, x_pos-1):min(28, x_pos+2)] = 0.8
                    
        elif digit == 2:
            # Draw 2-like shape with variations
            # Top horizontal
            img[8:11, 10:18] = 0.8
            # Right vertical
            img[11:15, 15:18] = 0.8
            # Middle horizontal
            img[14:17, 10:18] = 0.8
            # Left vertical
            img[17:21, 10:13] = 0.8
            # Bottom horizontal
            img[20:23, 10:18] = 0.8
            
        elif digit == 3:
            # Draw 3-like shape
            img[8:11, 10:18] = 0.8  # Top
            img[13:16, 10:18] = 0.8  # Middle
            img[20:23, 10:18] = 0.8  # Bottom
            img[11:20, 15:18] = 0.8  # Right side
            
        elif digit == 4:
            # Draw 4-like shape
            img[8:16, 10:13] = 0.8   # Left vertical
            img[13:16, 10:18] = 0.8  # Horizontal
            img[8:23, 15:18] = 0.8   # Right vertical
            
        elif digit == 5:
            # Draw 5-like shape
            img[8:11, 10:18] = 0.8   # Top
            img[11:15, 10:13] = 0.8  # Left upper
            img[13:16, 10:18] = 0.8  # Middle
            img[16:20, 15:18] = 0.8  # Right lower
            img[19:22, 10:18] = 0.8  # Bottom
            
        elif digit == 6:
            # Draw 6-like shape
            img[8:22, 10:13] = 0.8   # Left side
            img[8:11, 10:18] = 0.8   # Top
            img[13:16, 10:18] = 0.8  # Middle
            img[19:22, 10:18] = 0.8  # Bottom
            img[16:22, 15:18] = 0.8  # Right lower
            
        elif digit == 7:
            # Draw 7-like shape
            img[8:11, 10:18] = 0.8   # Top
            img[11:22, 14:17] = 0.8  # Diagonal
            
        elif digit == 8:
            # Draw 8-like shape
            img[8:11, 11:16] = 0.8   # Top
            img[13:16, 11:16] = 0.8  # Middle
            img[19:22, 11:16] = 0.8  # Bottom
            img[11:19, 10:13] = 0.8  # Left
            img[11:19, 14:17] = 0.8  # Right
            
        elif digit == 9:
            # Draw 9-like shape
            img[8:11, 10:18] = 0.8   # Top
            img[11:15, 10:13] = 0.8  # Left upper
            img[13:16, 10:18] = 0.8  # Middle
            img[8:22, 15:18] = 0.8   # Right side
        
        # Add noise and variations
        noise = np.random.normal(0, 0.03, (28, 28))
        img += noise
        img = np.clip(img, 0, 1)
        
        images.append(img)
    
    return np.array(images)

def generate_digit_images(generator, digit, num_samples=5, use_model=True):
    """Generate images for a specific digit"""
    if use_model:
        with torch.no_grad():
            noise = torch.randn(num_samples, 100).to(device)
            labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
            fake_imgs = generator(noise, labels)
            
            # Denormalize images
            fake_imgs = fake_imgs * 0.5 + 0.5
            fake_imgs = torch.clamp(fake_imgs, 0, 1)
            
            return fake_imgs.cpu().numpy().squeeze()
    else:
        return generate_mock_images(digit, num_samples)

def main():
    # Title and description
    st.title("🎨 Handwritten Digit Generator")
    st.markdown("""
    This app generates handwritten-style digits using a Generative Adversarial Network (GAN) trained on the MNIST dataset.
    Select a digit below and click **Generate** to create 5 unique variations!
    """)
    
    # Load model
    generator, model_loaded = load_model()
    
    if model_loaded:
        st.success("✅ GAN model loaded successfully!")
    else:
        st.info("ℹ️ Using mock generation for demonstration purposes.")
    
    # Digit selection
    st.subheader("Select Digit to Generate")
    
    # Create digit selection buttons in a grid
    cols = st.columns(10)
    selected_digit = None
    
    for i in range(10):
        with cols[i]:
            if st.button(f"{i}", key=f"digit_{i}", help=f"Generate digit {i}"):
                selected_digit = i
    
    # Store selected digit in session state
    if selected_digit is not None:
        st.session_state.selected_digit = selected_digit
    
    # Display current selection
    if 'selected_digit' in st.session_state:
        st.info(f"Selected digit: **{st.session_state.selected_digit}**")
        
        # Generate button
        if st.button("🎯 Generate 5 Images", type="primary"):
            with st.spinner("Generating handwritten digits..."):
                # Generate images
                images = generate_digit_images(
                    generator, 
                    st.session_state.selected_digit, 
                    num_samples=5,
                    use_model=model_loaded
                )
                
                # Display results
                st.subheader(f"Generated Images for Digit {st.session_state.selected_digit}")
                
                # Create columns for images
                cols = st.columns(5)
                
                for i, img in enumerate(images):
                    with cols[i]:
                        # Convert to PIL Image for display
                        img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
                        
                        # Resize for better visibility
                        img_resized = img_pil.resize((112, 112), Image.NEAREST)
                        
                        st.image(img_resized, caption=f"Sample {i+1}", use_column_width=True)
                
                # Display statistics
                st.subheader("Generation Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Images Generated", "5")
                with col2:
                    st.metric("Image Size", "28×28 pixels")
                with col3:
                    st.metric("Selected Digit", st.session_state.selected_digit)
                
                # Show raw images as a grid
                if st.checkbox("Show Original 28×28 Images"):
                    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
                    for i, img in enumerate(images):
                        axes[i].imshow(img, cmap='gray')
                        axes[i].set_title(f'Sample {i+1}')
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
    
    else:
        st.info("👆 Please select a digit (0-9) to generate images.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **Model Architecture:**
        - Conditional GAN (cGAN)
        - Generator: 100D noise + digit label → 28×28 image
        - Trained on MNIST dataset
        
        **Features:**
        - Generates unique variations
        - MNIST-style 28×28 grayscale images
        - Supports all digits 0-9
        
        **Technical Details:**
        - Framework: PyTorch
        - Training: Google Colab (T4 GPU)
        - Loss: Binary Cross Entropy
        """)
        
        if st.button("🔄 Reset Session"):
            if 'selected_digit' in st.session_state:
                del st.session_state.selected_digit
            st.rerun()

if __name__ == "__main__":
    main()
