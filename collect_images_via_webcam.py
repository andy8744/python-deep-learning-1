import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import os
from datetime import datetime
from PIL import Image, ImageTk
import threading
import time

class WebcamImageCapture:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Webcam Image Capture")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        
        # Initialize webcam
        self.cap = None
        self.webcam_running = False
        
        # Initialize counters for each category
        self.counters = {'rock': 0, 'paper': 0, 'scissors': 0, 'null': 0}
        
        # Update counters based on existing files
        self.update_counters_from_files()
        
        # Create GUI elements
        self.setup_gui()
        
        # Start webcam capture
        self.start_webcam()
        
    def setup_gui(self):
        # Counter display
        self.counter_text = tk.StringVar()
        self.update_counter_display()
        
        counter_label = tk.Label(
            self.root, 
            textvariable=self.counter_text,
            font=('Arial', 14, 'bold'),
            bg='lightblue',
            fg='darkblue',
            relief='raised',
            padx=10,
            pady=5
        )
        counter_label.pack(pady=10)
        
        # Video display frame
        self.video_frame = tk.Label(self.root, bg='black')
        self.video_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # Create buttons for each category
        categories = ['Rock', 'Paper', 'Scissors', 'Null']
        for i, category in enumerate(categories):
            btn = tk.Button(
                button_frame,
                text=category,
                font=('Arial', 12),
                width=10,
                height=2,
                command=lambda cat=category.lower(): self.capture_image(cat)
            )
            btn.pack(side='left', padx=10)
            
    def update_counters_from_files(self):
        """Update counters based on existing files in data folders"""
        for category in self.counters.keys():
            data_folder = os.path.join('data', category)
            if os.path.exists(data_folder):
                files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
                self.counters[category] = len(files)
                
    def update_counter_display(self):
        """Update the counter display string"""
        counter_str = " | ".join([f"{cat.capitalize()}: {count}" for cat, count in self.counters.items()])
        self.counter_text.set(counter_str)
        
    def start_webcam(self):
        """Initialize and start webcam capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
                
            self.webcam_running = True
            self.update_webcam_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {str(e)}")
            
    def update_webcam_image(self):
        """Update the webcam image display"""
        if self.webcam_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Get the display frame size
                self.video_frame.update_idletasks()
                frame_width = self.video_frame.winfo_width()
                frame_height = self.video_frame.winfo_height()
                
                # Skip if frame hasn't been rendered yet
                if frame_width <= 1 or frame_height <= 1:
                    self.root.after(50, self.update_webcam_image)
                    return
                
                # Calculate aspect ratio preserving resize
                original_height, original_width = frame.shape[:2]
                aspect_ratio = original_width / original_height
                
                # Calculate new dimensions that fit within the display frame
                if frame_width / frame_height > aspect_ratio:
                    # Frame is wider than image aspect ratio
                    new_height = frame_height
                    new_width = int(frame_height * aspect_ratio)
                else:
                    # Frame is taller than image aspect ratio
                    new_width = frame_width
                    new_height = int(frame_width / aspect_ratio)
                
                # Resize frame maintaining aspect ratio
                resized_frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert BGR to RGB for tkinter
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update the label
                self.video_frame.configure(image=photo)
                self.video_frame.image = photo  # Keep a reference
                
        # Schedule next update
        if self.webcam_running:
            self.root.after(50, self.update_webcam_image)  # ~20 FPS
            
    def capture_image(self, category):
        """Capture and save an image for the specified category"""
        if not self.webcam_running or self.cap is None:
            messagebox.showerror("Error", "Webcam not available")
            return
            
        try:
            # Ensure data directory exists
            data_folder = os.path.join('data', category)
            os.makedirs(data_folder, exist_ok=True)
            
            # Capture image
            ret, frame = self.cap.read()
            if ret:
                # Resize to 224x224
                resized_frame = cv2.resize(frame, (224, 224))
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')[:-3]  # Remove last 3 digits of microseconds
                filename = f"{category}_{timestamp}.jpg"
                filepath = os.path.join(data_folder, filename)
                
                # Save image
                cv2.imwrite(filepath, resized_frame)
                
                # Update counter
                self.counters[category] += 1
                self.update_counter_display()
                
                print(f"Captured {category} image: {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture image: {str(e)}")
            
    def close_app(self):
        """Clean up resources and close the application"""
        self.webcam_running = False
        
        if self.cap is not None:
            self.cap.release()
            
        cv2.destroyAllWindows()
        self.root.destroy()
        
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()

def main():
    """Main function to run the application"""
    app = WebcamImageCapture()
    app.run()

if __name__ == "__main__":
    main()
