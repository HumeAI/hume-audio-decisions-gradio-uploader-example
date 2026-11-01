// Gradio's dark theme follows the `dark` class on the body. Gradio's own theme
// setting reloads the page, which would discard the session's results.
const STORAGE_KEY = "audio-decisions-gradio-uploader:theme";

function showThemeIcon() {
  const dark = document.body.classList.contains("dark");
  element.querySelector(".light-icon").hidden = dark;
  element.querySelector(".dark-icon").hidden = !dark;
}

try {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) {
    document.body.classList.toggle("dark", saved === "dark");
  }
} catch {}
showThemeIcon();

// Gradio also switches the class when the system theme changes.
new MutationObserver(showThemeIcon).observe(document.body, {
  attributes: true,
  attributeFilter: ["class"],
});

element.querySelector("button").addEventListener("click", () => {
  const dark = document.body.classList.toggle("dark");
  try {
    localStorage.setItem(STORAGE_KEY, dark ? "dark" : "light");
  } catch {
    // The theme still applies until the page is reloaded.
  }
});
