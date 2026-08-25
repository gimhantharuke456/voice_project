// Replaces electron-app/preload.js's contextBridge for the plain-web version.
// Since this is served by the same Flask app that exposes the API, an empty
// string means "same origin" - requests go to whatever host/port this page
// was loaded from, so it works both on localhost and when accessed from
// another device on the network.
window.APP_CONFIG = {
    apiUrl: '',
    version: '1.0.0',
};
