module.exports = {
  content: ["./pages/*.{html,js}", "./index.html", "./js/*.js", "./components/*.{html,js}"],
  theme: {
    extend: {
      colors: {
        // Primary Colors - Deep Navy (Analytical Authority)
        primary: {
          DEFAULT: "#1a365d", // blue-900 equivalent
          50: "#e6f3ff",
          100: "#b3d9ff", 
          200: "#80bfff",
          300: "#4da6ff",
          400: "#1a8cff",
          500: "#1a365d",
          600: "#153052",
          700: "#102947",
          800: "#0b223c",
          900: "#061b31"
        },
        // Secondary Colors - Teal (Innovation Moments)
        secondary: {
          DEFAULT: "#38b2ac", // teal-500 equivalent
          50: "#e6fffa",
          100: "#b2f5ea",
          200: "#81e6d9", 
          300: "#4fd1c7",
          400: "#38b2ac",
          500: "#319795",
          600: "#2c7a7b",
          700: "#285e61",
          800: "#234e52",
          900: "#1d4044"
        },
        // Accent Colors - Orange (Warm Action Prompts)
        accent: {
          DEFAULT: "#ed8936", // orange-400 equivalent
          50: "#fffaf0",
          100: "#feebc8",
          200: "#fbd38d",
          300: "#f6ad55",
          400: "#ed8936",
          500: "#dd6b20",
          600: "#c05621",
          700: "#9c4221",
          800: "#7b341e",
          900: "#652b19"
        },
        // Background Colors
        background: "#ffffff", // white
        surface: {
          DEFAULT: "#f7fafc", // gray-50 equivalent
          50: "#f7fafc",
          100: "#edf2f7",
          200: "#e2e8f0",
          300: "#cbd5e0"
        },
        // Text Colors
        text: {
          primary: "#2d3748", // gray-700 equivalent
          secondary: "#718096" // gray-500 equivalent
        },
        // Status Colors
        success: {
          DEFAULT: "#38a169", // green-600 equivalent
          50: "#f0fff4",
          100: "#c6f6d5",
          200: "#9ae6b4",
          300: "#68d391",
          400: "#48bb78",
          500: "#38a169",
          600: "#2f855a",
          700: "#276749",
          800: "#22543d",
          900: "#1c4532"
        },
        warning: {
          DEFAULT: "#d69e2e", // yellow-600 equivalent
          50: "#fffff0",
          100: "#fefcbf",
          200: "#faf089",
          300: "#f6e05e",
          400: "#ecc94b",
          500: "#d69e2e",
          600: "#b7791f",
          700: "#975a16",
          800: "#744210",
          900: "#5f370e"
        },
        error: {
          DEFAULT: "#e53e3e", // red-500 equivalent
          50: "#fed7d7",
          100: "#feb2b2",
          200: "#fc8181",
          300: "#f56565",
          400: "#e53e3e",
          500: "#c53030",
          600: "#9b2c2c",
          700: "#742a2a",
          800: "#63171b",
          900: "#521b1b"
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        inter: ['Inter', 'sans-serif'],
        code: ['Fira Code', 'monospace']
      },
      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.875rem', { lineHeight: '1.25rem' }],
        'base': ['1rem', { lineHeight: '1.5rem' }],
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
        '5xl': ['3rem', { lineHeight: '1' }],
        '6xl': ['3.75rem', { lineHeight: '1' }]
      },
      fontWeight: {
        normal: '400',
        medium: '500',
        semibold: '600',
        bold: '700'
      },
      boxShadow: {
        'subtle': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'interactive': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'data-card': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)'
      },
      borderRadius: {
        'sm': '0.125rem',
        DEFAULT: '0.25rem',
        'md': '0.375rem',
        'lg': '0.5rem',
        'xl': '0.75rem',
        '2xl': '1rem'
      },
      transitionDuration: {
        '300': '300ms'
      },
      transitionTimingFunction: {
        'smooth': 'ease-in-out'
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem'
      },
      maxWidth: {
        '8xl': '88rem',
        '9xl': '96rem'
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-in-out',
        'scale-in': 'scaleIn 0.3s ease-in-out'
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' }
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' }
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' }
        }
      }
    }
  },
  plugins: [
    function({ addUtilities }) {
      const newUtilities = {
        '.transition-smooth': {
          transition: '300ms ease-in-out'
        },
        '.text-balance': {
          'text-wrap': 'balance'
        }
      }
      addUtilities(newUtilities)
    }
  ]
}