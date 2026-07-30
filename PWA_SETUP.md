# ClassMillia PWA Setup Guide

## Overview
Your ClassMillia attendance system is now configured as a Progressive Web App (PWA), allowing users to install it from Chrome and use it like a native application.

## What Was Added

### 1. PWA Manifest (`static/manifest.json`)
- Defines app metadata (name, description, colors)
- Specifies icons for different sizes
- Sets app behavior (standalone mode, theme colors)
- Includes shortcuts for Student and Lecturer dashboards

### 2. Service Worker (`static/service-worker.js`)
- Enables offline functionality
- Caches static assets for faster loading
- Provides network fallback for cached content
- Automatically updates cache when new versions are available

### 3. PWA Icons
- Currently using placeholder icons from placehold.co
- Icons available in sizes: 72x72, 96x96, 128x128, 144x144, 152x152, 192x192, 384x384, 512x512
- Can be replaced with custom branded icons

### 4. HTML Updates
- **base.html**: Added manifest link, theme color, service worker registration
- **welcome_page.html**: Added PWA meta tags and service worker registration

### 5. Django Configuration
- Updated URLs to serve static files in development
- Configured proper static file serving for PWA files

## How to Install the PWA

### On Desktop Chrome:
1. Open your website in Chrome
2. Look for the install icon (⊕) in the address bar
3. Click "Install ClassMillia"
4. The app will be installed and available from your desktop/applications

### On Mobile Chrome:
1. Open your website in Chrome on Android
2. Tap the menu (three dots)
3. Select "Add to Home Screen" or "Install App"
4. The app will be added to your home screen

## PWA Features

✅ **Installable** - Can be installed from Chrome browser
✅ **Offline Support** - Works without internet connection (cached pages)
✅ **App-like Experience** - Runs in standalone mode (no browser UI)
✅ **Push Notifications Ready** - Can be extended for notifications
✅ **Fast Loading** - Caches static assets for performance
✅ **Responsive** - Works on all device sizes
✅ **Shortcuts** - Quick access to Student/Lecturer dashboards

## Testing the PWA

### 1. Local Testing
```bash
python manage.py runserver
```
Open Chrome DevTools (F12) and check:
- **Application Tab** → Manifest should be loaded
- **Service Workers** → Should be registered and active
- **Lighthouse** → Run PWA audit

### 2. Production Testing
Deploy to Render and test the installation process from the live URL.

## Customizing Icons

### Option 1: Use Online Generator
1. Visit https://realfavicongenerator.net/
2. Upload your logo/design
3. Download the generated icon pack
4. Replace placeholder URLs in `manifest.json` with local paths

### Option 2: Use the Provided Script
```bash
pip install pillow
python generate-icons.py
```
Then update `manifest.json` to use local icon paths.

### Option 3: Manual Creation
Create PNG icons in these sizes and place them in `static/icons/`:
- 72x72, 96x96, 128x128, 144x144, 152x152, 192x192, 384x384, 512x512

Update `manifest.json` to use:
```json
"src": "/static/icons/icon-192x192.png"
```

## Service Worker Details

The service worker currently caches:
- Main page
- Manifest file
- Basic CSS and JS files

### To Cache Additional Files:
Update `urlsToCache` array in `static/service-worker.js`:
```javascript
const urlsToCache = [
  '/',
  '/static/manifest.json',
  '/static/css/style.css',
  '/static/js/main.js',
  '/static/your-other-files.css'
];
```

## Troubleshooting

### PWA Not Installing:
1. Check that manifest.json is accessible: `https://yourdomain.com/static/manifest.json`
2. Verify service worker is registered (check browser console)
3. Ensure site is served over HTTPS (required for PWA)
4. Check that icons are loading correctly

### Service Worker Not Working:
1. Open DevTools → Application → Service Workers
2. Check for registration errors in console
3. Try unregistering and refreshing the page
4. Verify the service worker file path is correct

### Icons Not Showing:
1. Verify icon URLs are accessible
2. Check browser console for 404 errors
3. Ensure icon sizes match manifest specifications

## Deployment Notes

### Render Deployment:
- PWA files are automatically included in static files
- No additional configuration needed
- HTTPS is automatically provided

### Static File Collection:
```bash
python manage.py collectstatic
```
This ensures all PWA files are properly deployed.

## Future Enhancements

Consider adding:
- **Push Notifications** - For attendance reminders
- **Background Sync** - Sync attendance data offline
- **Offline Forms** - Allow marking attendance without connection
- **App Updates** - Prompt users when new versions are available
- **Share Target** - Allow sharing content to the app

## Browser Support

PWA installation works in:
- ✅ Chrome/Edge (Desktop & Android)
- ✅ Firefox (Android)
- ✅ Safari (iOS - limited support)
- ❌ Safari Desktop (not supported)

## Security Notes

- Service workers only work over HTTPS
- Manifest and service worker must be from same origin
- Ensure proper CORS headers if loading external resources

## Performance Benefits

- **Faster Load Times** - Cached assets load instantly
- **Reduced Bandwidth** - Only updates changed files
- **Better UX** - Native app-like experience
- **Offline Capability** - Works without internet

## Maintenance

### Update PWA:
1. Update version in service worker CACHE_NAME
2. Modify cached files list as needed
3. Test changes in development
4. Deploy and collect static files

### Monitor Usage:
- Check service worker registration in analytics
- Monitor install rates through PWA install prompts
- Track offline usage patterns

---

For more information, visit:
- [PWA Best Practices](https://web.dev/progressive-web-apps/)
- [Service Worker API](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API)
- [Web App Manifest](https://developer.mozilla.org/en-US/docs/Web/Manifest)