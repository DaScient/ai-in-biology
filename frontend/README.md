# AI in Biological Sciences — Frontend

Static-first, progressive enhancement site built with [Astro](https://astro.build).

## Quick start

```bash
cd frontend
npm install
npm run dev      # local dev server
npm run build    # static build to dist/
npm run preview  # preview production build
```

## Stack

- **Astro 4** with island architecture (zero JS by default)
- **Preact** for interactive islands
- **Tailwind CSS** + Open Props (CSS variables in `src/styles/theme.css`)
- **MDX** for content (textbook chapters, tutorials)
- **Pagefind** for static full-text search
- **Service worker** (`public/sw.js`) for offline support

## Structure

See the top-level `frontend/` development plan for the full directory layout.
Key entry points:

- `src/pages/index.astro` — landing page
- `src/pages/api/playground.astro` — interactive API tester
- `src/components/interactive/NotebookLauncher.tsx` — Preact island for notebooks
- `src/components/common/SearchBar.astro` — Pagefind-backed search modal

## Deployment

Pushes to `main` that touch `frontend/**`, `api/notebooks/**`, or `docs/source/**`
trigger `.github/workflows/deploy-frontend.yml`, which builds the site and
publishes to GitHub Pages.
