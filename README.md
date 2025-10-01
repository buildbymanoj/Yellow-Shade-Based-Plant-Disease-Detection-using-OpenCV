# 🌱 Plant Disease Detection System

A real-time plant disease detection application that identifies diseases based on yellow discoloration in plant leaves. The system uses computer vision and HSV color space analysis to detect and classify plant diseases.

## 🎯 Features

- **Multiple Input Methods**
  - 📷 Upload images of plant leaves
  - 🎥 Upload and process video files
  - 📹 Real-time webcam detection with streaming

- **Disease Detection**
  - Nitrogen Deficiency (light yellow leaves)
  - Fungal Infection (dark yellow patches)
  - Potassium Deficiency (bright yellow discoloration)

- **Advanced Controls**
  - Adjustable HSV thresholds for fine-tuning detection
  - Minimum area filtering to reduce noise
  - Brightness adjustment for low-light conditions
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) for image enhancement

- **Visual Output**
  - Annotated images with bounding boxes
  - Detection masks showing affected areas
  - Detailed summary with HSV values

## 📋 Requirements

- Python 3.8 or higher
- Webcam (optional, for real-time detection)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd plant-disease-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. **Run the application**
   ```bash
   python app.py
   ```

2. **Access the interface**
   - The application will launch in your default browser
   - A local URL will be displayed (typically `http://127.0.0.1:7860`)
   - A public shareable link will also be generated

3. **Choose your input method**
   - **Image Tab**: Upload a plant image
   - **Video Tab**: Upload a video file (processes first 10 seconds)
   - **Webcam Tab**: Use your camera for real-time detection

4. **Adjust settings (optional)**
   - Expand "Advanced Settings" to fine-tune detection parameters
   - Modify HSV thresholds if detection is not accurate
   - Adjust minimum area to filter small artifacts

## 🎛️ Parameter Guide

### HSV Thresholds
- **Hue Min/Max (0-179)**: Controls the yellow color range
  - Default: 20-35 (captures yellow-green to yellow-orange)
- **Saturation Min/Max (0-255)**: Controls color intensity
  - Default: 50-255 (filters out pale/washed colors)
- **Value Min/Max (0-255)**: Controls brightness
  - Default: 50-255 (filters out dark regions)

### Other Parameters
- **Min Area**: Minimum contour size to detect (default: 500 pixels)
- **Brightness**: Adds brightness to dark images (0-100)

## 🔬 Disease Classification Logic

The system analyzes the average HSV values of detected yellow regions:

| Disease | HSV Characteristics |
|---------|-------------------|
| **Nitrogen Deficiency** | High Value (>180), Low Saturation (<100) - Light, pale yellow |
| **Fungal Infection** | High Saturation (>150), Low Value (<180) - Dark, intense yellow |
| **Potassium Deficiency** | High Saturation (>100), High Value (>180) - Bright, vivid yellow |

## 📁 Project Structure

```
plant-disease-detection/
│
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── output_video.mp4      # Generated processed video (temporary)
```

## 🛠️ Technologies Used

- **OpenCV**: Image processing and computer vision
- **NumPy**: Numerical computations
- **Gradio**: Web interface for ML applications
- **PIL/Pillow**: Image handling

## 📸 Tips for Best Results

1. **Image Quality**
   - Use well-lit images with clear yellow discoloration
   - Avoid shadows and reflections
   - Keep the camera steady for video/webcam input

2. **Threshold Adjustment**
   - If no yellow is detected, decrease S_min and V_min
   - If too many false positives, increase S_min and V_min
   - Adjust H_min and H_max to narrow/widen the yellow range

3. **Lighting**
   - For dark images, increase brightness adjustment
   - Natural lighting works best
   - Avoid direct flash or harsh lighting

## ⚠️ Limitations

- Video processing is limited to the first 10 seconds to optimize performance
- Detection accuracy depends on lighting conditions and image quality
- The system is calibrated for yellow-based diseases only
- Disease classification is based on color analysis and should be verified by experts

## 🔮 Future Enhancements

- [ ] Add more disease types (brown spots, white mold, etc.)
- [ ] Machine learning-based classification
- [ ] Support for multiple plant species
- [ ] Database integration for tracking disease history
- [ ] Mobile app development
- [ ] Batch processing for multiple images

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

Your Name - buildbymanoj

## 🙏 Acknowledgments

- OpenCV community for excellent documentation
- Gradio team for the intuitive interface framework
- Agricultural research community for disease classification insights

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: your.email@example.com

---

**Note**: This system is intended as a diagnostic aid and should not replace professional agricultural consultation. Always verify disease identification with qualified experts.
