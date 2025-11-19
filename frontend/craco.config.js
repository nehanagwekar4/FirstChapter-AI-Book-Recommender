const path = require('path');

module.exports = {
  webpack: {
    alias: {
      // Resolves the alias "@/App.css" to the actual location: src/App.css
      '@': path.resolve(__dirname, 'src/'),
    },
  },
};