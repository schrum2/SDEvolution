import tkinter as tk
from PIL import Image, ImageTk, PngImagePlugin
from math import ceil, sqrt
import io
import re
import json

"""
This class was made by Claude: https://claude.ai/
"""

class ImageGridViewer:
    def __init__(self, root, callback_fn=None, initial_prompt="", initial_neg_prompt="", back_fn=None):
        self.root = root
        self.root.title("Generated Images")
        self.images = []  # Stores PIL Image objects
        self.photo_images = []  # Stores PhotoImage objects (needed to prevent garbage collection)
        self.selected_images = set()  # Tracks which images are selected
        self.buttons = []  # Stores the button widgets
        self.tooltips = []  # Stores tooltip text for each image
        self.metadata = []  # to be embedded in PNG images
        self.callback_fn = callback_fn
        self.back_fn = back_fn
        
        # Initial window sizing
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Set initial window size to 75% of screen
        window_width = int(screen_width * 0.75)
        window_height = int(screen_height * 0.75)
        root.geometry(f"{window_width}x{window_height}")

        # Create main container frame
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(expand=True, fill=tk.BOTH)
        
        # Create frame for images with weight=1 to allow expansion
        self.image_frame = tk.Frame(self.main_container)
        self.image_frame.pack(expand=True, fill=tk.BOTH, pady=10)
        
        # Create frame for control buttons and text inputs
        self.control_frame = tk.Frame(self.main_container, height=120)  # Increased height for text fields
        self.control_frame.pack(fill=tk.X, pady=5, padx=10)
        self.control_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Create button frame
        self.button_frame = tk.Frame(self.control_frame)
        self.button_frame.pack(fill=tk.X)
        
        # Add Back button
        self.back_button = tk.Button(
            self.button_frame,
            text="Previous Generation",
            command=self._handle_back,
            width=20
        )
        self.back_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add Done button
        self.done_button = tk.Button(
            self.button_frame,
            text="Reset",
            command=self._handle_done,
            width=20
        )
        self.done_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Add Save button
        self.save_button = tk.Button(
            self.button_frame,
            text="Save Selected",
            command=self._save_selected,
            width=20
        )
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add Close button
        self.close_button = tk.Button(
            self.button_frame,
            text="Close",
            command=self.root.destroy,
            width=20
        )
        self.close_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create prompt input frame
        self.prompt_frame = tk.Frame(self.control_frame)
        self.prompt_frame.pack(fill=tk.X, padx=5)
        
        # Add prompt label and entry
        tk.Label(self.prompt_frame, text="Prompt: ").pack(side=tk.LEFT)
        self.prompt_entry = tk.Entry(self.prompt_frame)
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.prompt_entry.insert(0, initial_prompt)
        
        # Create negative prompt input frame
        self.neg_prompt_frame = tk.Frame(self.control_frame)
        self.neg_prompt_frame.pack(fill=tk.X, padx=5)
        
        # Add negative prompt label and entry
        tk.Label(self.neg_prompt_frame, text="Neg prompt: ").pack(side=tk.LEFT)
        self.neg_prompt_entry = tk.Entry(self.neg_prompt_frame)
        self.neg_prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.neg_prompt_entry.insert(0, initial_neg_prompt)

        # Bind resize event
        self.root.bind('<Configure>', self._on_window_resize)

    def clear_images(self):
        """Clears all images from the grid and resets selections."""
        self.images.clear()
        self.tooltips.clear()
        self.metadata.clear()
        self.selected_images.clear()
        self._update_grid()

    def add_image(self, pil_image, tooltip_text="", image_metadata=None):
        """
        Add a new image to the grid with an optional tooltip.
        
        Args:
            pil_image: PIL Image object
            tooltip_text: String to display when hovering over the image
        """
        self.images.append(pil_image)
        self.tooltips.append(tooltip_text)
        self.metadata.append(image_metadata)
        self._update_grid()
        
    def get_selected_images(self):
        """Returns list of selected PIL Image objects."""
        return [(i,self.images[i]) for i in self.selected_images]
    
    def _calculate_thumbnail_size(self):
        """Calculate thumbnail size based on current window dimensions."""
        # Get current window size
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height() - 120  # Adjusted for larger control frame
        
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
    
    def _create_tooltip(self, widget, text):
        """Create a tooltip for a widget."""
        def enter(event):
            # Create a toplevel window
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)  # Remove window decorations
            
            # Position tooltip near the mouse
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20
            
            # Create tooltip label
            label = tk.Label(tooltip, text=text, justify=tk.LEFT,
                           background="#ffffe0", relief=tk.SOLID, borderwidth=1)
            label.pack()
            
            tooltip.wm_geometry(f"+{x}+{y}")
            widget._tooltip = tooltip
            
        def leave(event):
            # Destroy tooltip when mouse leaves
            if hasattr(widget, '_tooltip'):
                widget._tooltip.destroy()
                del widget._tooltip
        
        if text:  # Only bind events if there's tooltip text
            widget.bind('<Enter>', enter)
            widget.bind('<Leave>', leave)
    
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
            
            # Add tooltip
            self._create_tooltip(btn, self.tooltips[idx])
            
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

        if len(self.selected_images) == 0:
            self.done_button.config(text="Reset")
        else:
            self.done_button.config(text="Evolve Selected")

    def _handle_done(self):
        """Called when Evolve button is clicked"""

        self.done_button.config(text="Reset")
        if self.callback_fn:
            selected = self.get_selected_images()
            prompt = self.prompt_entry.get()
            neg_prompt = self.neg_prompt_entry.get()
            self.callback_fn(selected, prompt, neg_prompt)

    def _save_selected(self):
        selected = self.get_selected_images()
        for (i,image) in selected:
            full_desc = self.tooltips[i]
            image_meta = self.metadata[i]

            metadata = PngImagePlugin.PngInfo()
            gen_meta_str = json.dumps(image_meta)
            metadata.add_text("SD_data",gen_meta_str)

            match = re.search(r"id=(\d+)", full_desc)
            output = f"Image_Id{match.group(1)}_Num{i}.png"
            image.save(output, "PNG", pnginfo=metadata)
            print(f"Saved {output}")

    def _handle_back(self):
        """Called when Back button is clicked"""

        if self.back_fn:
            self.back_fn()