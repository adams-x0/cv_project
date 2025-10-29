/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        sage: {
          50:  '#F9FAF9',
          100: '#F2F7F1',
          200: '#E3E9E4',
          300: '#C8D3C5',
          400: '#A8BBA3',
          500: '#8EA68B',
          600: '#7A897C',
          700: '#5F6D63',
          800: '#4A554D',
          900: '#37443B',
        },
      },
      fontFamily: {
        sans: ['Outfit', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
