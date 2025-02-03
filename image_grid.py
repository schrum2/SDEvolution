import tkinter as tk
from PIL import Image, ImageTk
from math import ceil, sqrt
import io

"""
This class was made by Claude: https://claude.ai/
"""

class ImageGridViewer:
    def __init__(self, root, callback_fn=None):
        self.root = root
        self.root.title("Generated Images")
        self.images = []  # Stores PIL Image objects
        self.photo_images = []  # Stores PhotoImage objects (needed to prevent garbage collection)
        self.selected_images = set()  # Tracks which images are selected
        self.buttons = []  # Stores the button widgets
        self.callback_fn = callback_fn        

        # Create frame for images
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10)
        
        # Create frame for control buttons
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=5)
        
        # Add Done button
        self.done_button = tk.Button(
            self.control_frame,
            text="Evolve",
            command=self._handle_done,
            width=20,
            height=2
        )
        self.done_button.pack(side=tk.LEFT, padx=5)
        
        # Add Close button
        self.close_button = tk.Button(
            self.control_frame,
            text="Close",
            command=self.root.destroy,
            width=20,
            height=2
        )
        self.close_button.pack(side=tk.LEFT, padx=5)

    def clear_images(self):
        """Clears all images from the grid and resets selections."""
        self.images.clear()
        self.selected_images.clear()
        self._update_grid()

    def add_image(self, pil_image):
        """Add a new image to the grid. Expects a PIL Image object."""
        self.images.append(pil_image)
        self._update_grid()
        
    def get_selected_images(self):
        """Returns list of selected PIL Image objects."""
        return [(i,self.images[i]) for i in self.selected_images]
    
    def _update_grid(self):
        # Clear existing buttons
        for button in self.buttons:
            button.destroy()
        self.buttons.clear()
        self.photo_images.clear()
        
        # Calculate grid dimensions
        n_images = len(self.images)
        if n_images == 0:
            return
            
        grid_size = ceil(sqrt(n_images))  # Make a square-ish grid
        
        # Resize images for display
        thumbnail_size = (256, 256)
        
        for idx, img in enumerate(self.images):
            # Create a copy and resize for thumbnail
            thumb = img.copy()
            thumb.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(thumb)
            self.photo_images.append(photo)
            
            # Create button
            btn = tk.Button(
                self.image_frame,
                image=photo,
                relief='solid',
                borderwidth=2
            )
            
            # Configure selection behavior
            btn.configure(
                command=lambda i=idx, b=btn: self._toggle_selection(i, b)
            )
            
            # Position in grid
            row = idx // grid_size
            col = idx % grid_size
            btn.grid(row=row, column=col, padx=5, pady=5)
            
            self.buttons.append(btn)
            
            # Update selected state if necessary
            if idx in self.selected_images:
                btn.configure(bg='blue')
    
    def _toggle_selection(self, idx, button):
        if idx in self.selected_images:
            self.selected_images.remove(idx)
            button.configure(bg='SystemButtonFace')  # Default background
        else:
            self.selected_images.add(idx)
            button.configure(bg='blue')  # Highlight selected

    def _handle_done(self):
        """Called when Evolve button is clicked"""
        if self.callback_fn:
            selected = self.get_selected_images()
            self.callback_fn(selected)