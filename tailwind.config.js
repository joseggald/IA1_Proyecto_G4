/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      keyframes: {
        'bounce': {
          '0%, 100%': { transform: 'translateY(-25%)' },
          '50%': { transform: 'translateY(0)' },
        },
        'pulse': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: .5 },
        }
      },
      animation: {
        'bounce': 'bounce 1s infinite',
        'pulse': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite'
      }
    }
  },
  plugins: [],
}