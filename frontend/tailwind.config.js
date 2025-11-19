/** @type {import('tailwindcss').Config} */
module.exports = {
  // Scans all JavaScript and TypeScript files in src for utility classes
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html",
  ],
  theme: {
    extend: {
      // Custom fonts defined in the original App.css
      fontFamily: {
        'sans': ['Manrope', 'sans-serif'],
        'serif': ['Playfair Display', 'serif'],
      },
    },
  },
  plugins: [
    // Required plugin for animation classes
    require("tailwindcss-animate"), 
  ],
};