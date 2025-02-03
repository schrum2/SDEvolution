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
        
        # Initial window sizing
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Set initial window size to 75% of screen
        window_width = int(screen_width * 0.75)
        window_height = int(screen_height * 0.75)
        root.geometry(f"{window_width}x{window_height}")

        # Create frame for images
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(expand=True, fill=tk.BOTH, pady=10)
        
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

        # Bind resize event
        self.root.bind('<Configure>', self._on_window_resize)

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
    
    def _calculate_thumbnail_size(self):
        """Calculate thumbnail size based on current window dimensions."""
        # Get current window size
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        
        # Calculate grid dimensions for 3x3 grid
        n_images = len(self.images)
        if n_images == 0:
            return (256, 256)  # Default size if no images
        
        grid_size = min(3, ceil(sqrt(n_images)))
        
        # Calculate thumbnail size to fit the grid with some padding
        padding = 50  # Additional padding for margins and buttons
        max_thumb_width = (window_width - (grid_size + 1) * 10) // grid_size
        max_thumb_height = (window_height - (grid_size + 1) * 10 - padding) // grid_size
        
        # Ensure thumbnail has equal width and height
        thumbnail_size = min(max_thumb_width, max_thumb_height)
        
        return (thumbnail_size, thumbnail_size)
    
    def _on_window_resize(self, event):
        """Handles window resize event."""
        # Only update if the resize is significant to prevent excessive redraws
        if event.widget == self.root:
            self._update_grid()
    
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
            
        # Dynamically calculate grid size
        grid_size = min(3, ceil(sqrt(n_images)))
        
        # Get dynamic thumbnail size
        thumbnail_size = self._calculate_thumbnail_size()
        thumbnail_size = (max(100,thumbnail_size[0]), max(100,thumbnail_size[1]))

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
            btn.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            # Configure grid weights to make buttons resize
            self.image_frame.grid_rowconfigure(row, weight=1)
            self.image_frame.grid_columnconfigure(col, weight=1)
            
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