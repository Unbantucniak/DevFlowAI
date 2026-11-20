import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#182d32",
        panel: "#f4f7f5",
        line: "#d9e2df",
        accent: "#116d61",
        warn: "#b45309"
      }
    }
  },
  plugins: []
};

export default config;
