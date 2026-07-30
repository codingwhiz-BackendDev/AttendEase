"""
Icon Generation Script for ClassMillia PWA
This script helps generate PWA icons from the SVG template.
Requires: pillow (PIL) package
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create a simple gradient icon with a checkmark for ClassMillia"""
    # Create image with gradient background
    img = Image.new('RGB', (size, size), color='#667eea')
    draw = ImageDraw.Draw(img)
    
    # Create gradient effect (simplified)
    for i in range(size):
        alpha = int(255 * (i / size))
        color = (
            int(102 + (118 - 102) * (i / size)),  # R gradient
            int(126 + (75 - 126) * (i / size)),   # G gradient  
            int(234 + (162 - 234) * (i / size))   # B gradient
        )
        draw.line([(0, i), (size, i)], fill=color)
    
    # Draw rounded rectangle background
    margin = size // 10
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=size // 5,
        fill='white',
        outline='#764ba2',
        width=size // 20
    )
    
    # Draw checkmark
    center = size // 2
    check_size = size // 4
    checkmark_points = [
        (center - check_size, center),
        (center, center + check_size),
        (center + check_size, center - check_size)
    ]
    draw.line(checkmark_points, fill='#667eea', width=size // 15)
    
    # Save image
    img.save(output_path, 'PNG')
    print(f"Created icon: {output_path} ({size}x{size})")

def main():
    """Generate all required icon sizes"""
    icon_sizes = [72, 96, 128, 144, 152, 192, 384, 512]
    icons_dir = os.path.join(os.path.dirname(__file__), 'static', 'icons')
    
    # Create icons directory if it doesn't exist
    os.makedirs(icons_dir, exist_ok=True)
    
    # Generate icons
    for size in icon_sizes:
        output_path = os.path.join(icons_dir, f'icon-{size}x{size}.png')
        create_icon(size, output_path)
    
    print("All icons generated successfully!")

if __name__ == '__main__':
    main()