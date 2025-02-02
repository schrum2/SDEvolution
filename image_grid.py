import tkinter as tk
from PIL import Image, ImageTk
from math import ceil, sqrt
import io

"""
This class was made by Claude: https://claude.ai/
"""

class ImageGridViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Generated Images Grid")
        self.images = []  # Stores PIL Image objects
        self.photo_images = []  # Stores PhotoImage objects (needed to prevent garbage collection)
        self.selected_images = set()  # Tracks which images are selected
        self.buttons = []  # Stores the button widgets
        
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
        thumbnail_size = (150, 150)
        
        for idx, img in enumerate(self.images):
            # Create a copy and resize for thumbnail
            thumb = img.copy()
            thumb.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(thumb)
            self.photo_images.append(photo)
            
            # Create button
            btn = tk.Button(
                self.root,
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