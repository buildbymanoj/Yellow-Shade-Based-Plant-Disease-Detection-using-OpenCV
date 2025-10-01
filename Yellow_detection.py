import cv2
import numpy as np
import gradio as gr
from PIL import Image

def detect_disease_from_yellow_shade(hsv_color):
    h, s, v = hsv_color
    if v > 180 and s < 100:
        return "Nitrogen Deficiency (light yellow)"
    elif s > 150 and v < 180:
        return "Fungal Infection (dark yellow)"
    elif s > 100 and v > 180:
        return "Potassium Deficiency (bright yellow)"
    else:
        return "Unknown yellow shade"

def enhance_low_light(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def process_frame(frame, h_min=20, h_max=35, s_min=50, s_max=255, v_min=50, v_max=255, min_area=500, brightness=0):
    """Process a single frame and return annotated image with disease detection"""
    
    # Convert PIL Image to numpy array if needed
    if isinstance(frame, Image.Image):
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Apply brightness adjustment
    if brightness > 0:
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
    
    # Enhance low light
    enhanced = enhance_low_light(frame)
    
    # Convert to HSV
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    
    # Create mask for yellow regions
    lower_yellow = np.array([h_min, s_min, v_min])
    upper_yellow = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    diseases_detected = []
    
    if large_contours:
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = hsv[y:y+h, x:x+w]
            
            if roi.size > 0:
                avg_color = np.mean(roi.reshape(-1, 3), axis=0)
                disease = detect_disease_from_yellow_shade(avg_color)
                diseases_detected.append(disease)
                
                # Draw rectangle and text
                cv2.rectangle(enhanced, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(enhanced, disease, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(enhanced, f"H:{avg_color[0]:.1f} S:{avg_color[1]:.1f} V:{avg_color[2]:.1f}",
                            (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(enhanced, "No yellow detected - adjust thresholds", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Convert back to RGB for display
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    # Create summary
    if diseases_detected:
        summary = f"Detected {len(diseases_detected)} affected area(s):\n" + "\n".join(f"- {d}" for d in set(diseases_detected))
    else:
        summary = "No disease detected. Try adjusting the HSV thresholds."
    
    return enhanced_rgb, mask_rgb, summary

def process_image(image, h_min, h_max, s_min, s_max, v_min, v_max, min_area, brightness):
    """Process uploaded image"""
    return process_frame(image, h_min, h_max, s_min, s_max, v_min, v_max, min_area, brightness)

def process_video(video_path, h_min, h_max, s_min, s_max, v_min, v_max, min_area, brightness):
    """Process uploaded video and return processed video"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    max_frames = 300  # Limit to 10 seconds at 30fps to avoid long processing
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed, _, _ = process_frame(frame, h_min, h_max, s_min, s_max, v_min, v_max, min_area, brightness)
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        out.write(processed_bgr)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return output_path

def process_webcam(image, h_min, h_max, s_min, s_max, v_min, v_max, min_area, brightness):
    """Process webcam frame"""
    if image is None:
        return None, None, "No image captured"
    
    return process_frame(image, h_min, h_max, s_min, s_max, v_min, v_max, min_area, brightness)

# Create Gradio interface
with gr.Blocks(title="Plant Disease Detection System") as demo:
    gr.Markdown("# 🌱 Plant Disease Detection from Yellow Shades")
    gr.Markdown("Upload an image, video, or use your webcam to detect plant diseases based on yellow discoloration.")
    
    with gr.Tab("Image Upload"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Plant Image")
                
                with gr.Accordion("Advanced Settings", open=False):
                    h_min_img = gr.Slider(0, 179, 20, label="Hue Min", step=1)
                    h_max_img = gr.Slider(0, 179, 35, label="Hue Max", step=1)
                    s_min_img = gr.Slider(0, 255, 50, label="Saturation Min", step=1)
                    s_max_img = gr.Slider(0, 255, 255, label="Saturation Max", step=1)
                    v_min_img = gr.Slider(0, 255, 50, label="Value Min", step=1)
                    v_max_img = gr.Slider(0, 255, 255, label="Value Max", step=1)
                    min_area_img = gr.Slider(100, 5000, 500, label="Min Area", step=100)
                    brightness_img = gr.Slider(0, 100, 0, label="Brightness Adjustment", step=1)
                
                analyze_btn = gr.Button("Analyze Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Detected Diseases")
                mask_image = gr.Image(label="Detection Mask")
                result_text = gr.Textbox(label="Detection Summary", lines=5)
        
        analyze_btn.click(
            process_image,
            inputs=[image_input, h_min_img, h_max_img, s_min_img, s_max_img, v_min_img, v_max_img, min_area_img, brightness_img],
            outputs=[output_image, mask_image, result_text]
        )
    
    with gr.Tab("Video Upload"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Plant Video")
                
                with gr.Accordion("Advanced Settings", open=False):
                    h_min_vid = gr.Slider(0, 179, 20, label="Hue Min", step=1)
                    h_max_vid = gr.Slider(0, 179, 35, label="Hue Max", step=1)
                    s_min_vid = gr.Slider(0, 255, 50, label="Saturation Min", step=1)
                    s_max_vid = gr.Slider(0, 255, 255, label="Saturation Max", step=1)
                    v_min_vid = gr.Slider(0, 255, 50, label="Value Min", step=1)
                    v_max_vid = gr.Slider(0, 255, 255, label="Value Max", step=1)
                    min_area_vid = gr.Slider(100, 5000, 500, label="Min Area", step=100)
                    brightness_vid = gr.Slider(0, 100, 0, label="Brightness Adjustment", step=1)
                
                process_vid_btn = gr.Button("Process Video", variant="primary")
                gr.Markdown("*Note: Video processing is limited to first 10 seconds*")
            
            with gr.Column():
                output_video = gr.Video(label="Processed Video")
        
        process_vid_btn.click(
            process_video,
            inputs=[video_input, h_min_vid, h_max_vid, s_min_vid, s_max_vid, v_min_vid, v_max_vid, min_area_vid, brightness_vid],
            outputs=output_video
        )
    
    with gr.Tab("Webcam"):
        with gr.Row():
            with gr.Column():
                webcam_input = gr.Image(sources=["webcam"], type="pil", label="Capture from Webcam", streaming=True)
                
                with gr.Accordion("Advanced Settings", open=False):
                    h_min_cam = gr.Slider(0, 179, 20, label="Hue Min", step=1)
                    h_max_cam = gr.Slider(0, 179, 35, label="Hue Max", step=1)
                    s_min_cam = gr.Slider(0, 255, 50, label="Saturation Min", step=1)
                    s_max_cam = gr.Slider(0, 255, 255, label="Saturation Max", step=1)
                    v_min_cam = gr.Slider(0, 255, 50, label="Value Min", step=1)
                    v_max_cam = gr.Slider(0, 255, 255, label="Value Max", step=1)
                    min_area_cam = gr.Slider(100, 5000, 500, label="Min Area", step=100)
                    brightness_cam = gr.Slider(0, 100, 0, label="Brightness Adjustment", step=1)
            
            with gr.Column():
                output_webcam = gr.Image(label="Detected Diseases")
                mask_webcam = gr.Image(label="Detection Mask")
                result_webcam = gr.Textbox(label="Detection Summary", lines=5)
        
        webcam_input.change(
            process_webcam,
            inputs=[webcam_input, h_min_cam, h_max_cam, s_min_cam, s_max_cam, v_min_cam, v_max_cam, min_area_cam, brightness_cam],
            outputs=[output_webcam, mask_webcam, result_webcam]
        )
    
    gr.Markdown("""
    ### Disease Detection Guide:
    - **Nitrogen Deficiency**: Light yellow (high brightness, low saturation)
    - **Fungal Infection**: Dark yellow (high saturation, lower brightness)
    - **Potassium Deficiency**: Bright yellow (high saturation, high brightness)
    
    ### Tips:
    - Adjust HSV thresholds if detection is not accurate
    - Increase minimum area to filter out small noise
    - Use brightness adjustment for poorly lit images
    """)

if __name__ == "__main__":
    demo.launch(share=True)
