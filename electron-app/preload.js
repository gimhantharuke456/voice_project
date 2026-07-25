const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('appInfo', {
    apiUrl: 'http://localhost:5055',
    version: '1.0.0',
});
