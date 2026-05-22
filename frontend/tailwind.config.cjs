/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#e6f7f5',
          100: '#ccefeb',
          200: '#99dfd7',
          300: '#66cfc3',
          400: '#33bfaf',
          500: '#00af9b',
          600: '#008c7c',
          700: '#00695d',
          800: '#00463e',
          900: '#00231f',
        },
        accent: {
          50: '#f3e8ff',
          100: '#e7d1ff',
          200: '#cfa3ff',
          300: '#b775ff',
          400: '#9f47ff',
          500: '#8719ff',
          600: '#6c14cc',
          700: '#510f99',
          800: '#360a66',
          900: '#1b0533',
        },
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: 'none',
            code: {
              backgroundColor: 'var(--code-bg)',
              padding: '0.2rem 0.4rem',
              borderRadius: '0.25rem',
              fontSize: '0.875em',
            },
            'code::before': { content: '""' },
            'code::after': { content: '""' },
          },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
    require('@tailwindcss/forms'),
  ],
};
