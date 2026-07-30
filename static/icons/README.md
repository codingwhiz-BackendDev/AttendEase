# PWA Icons for ClassMillia

## Icon Generation

Since Python PIL is not available in the current environment, you have several options to generate the required PWA icons:

## Option 1: Use Online Tools (Recommended)

1. Visit https://www.favicon-generator.org/ or https://realfavicongenerator.net/
2. Upload the SVG template (`icon-template.svg`) or create a simple design
3. Download the generated icon pack
4. Extract the PNG files to this directory

## Option 2: Run the Script Locally

1. Make sure you have Python and Pillow installed:
   ```bash
   pip install pillow
   ```

2. Run the icon generation script:
   ```bash
   python generate-icons.py
   ```

## Required Icon Sizes

- icon-72x72.png
- icon-96x96.png  
- icon-128x128.png
- icon-144x144.png
- icon-152x152.png
- icon-192x192.png
- icon-384x384.png
- icon-512x512.png

## Icon Design

The icons should represent:
- Education/Classroom theme
- Checkmark (for attendance completion)
- Gradient colors: #667eea to #764ba2
- Clean, modern design

## Temporary Solution

For testing, you can use placeholder icons from:
- https://placehold.co/72x72/667eea/FFFFFF?text=CM
- https://placehold.co/96x96/667eea/FFFFFF?text=CM
- https://placehold.co/192x192/667eea/FFFFFF?text=CM
- https://placehold.co/512x512/667eea/FFFFFF?text=CM

Replace the URLs in manifest.json with actual icon files once generated.