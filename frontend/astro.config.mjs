import { defineConfig } from 'astro/config';
import preact from '@astrojs/preact';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import compress from 'astro-compress';
import { remarkAlert } from 'remark-github-alerts';

// https://astro.build/config
export default defineConfig({
  site: 'https://dascient.github.io/ai-in-biological-sciences',
  base: '/ai-in-biological-sciences',

  integrations: [
    preact({ compat: true }),
    tailwind({
      applyBaseStyles: false,
    }),
    mdx({
      remarkPlugins: [remarkAlert],
      rehypePlugins: [],
    }),
    sitemap(),
    compress({
      css: true,
      html: true,
      js: true,
      img: false, // handled by CloudFlare
    }),
  ],

  markdown: {
    shikiConfig: {
      theme: 'github-dark',
      langs: ['python', 'bash', 'json', 'yaml', 'r', 'julia'],
      wrap: true,
    },
    remarkPlugins: [
      'remark-math',
      'remark-gfm',
    ],
    rehypePlugins: [
      'rehype-katex',
      'rehype-slug',
      'rehype-autolink-headings',
    ],
  },

  vite: {
    optimizeDeps: {
      include: ['preact', 'preact/hooks', 'katex'],
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            katex: ['katex'],
            mermaid: ['mermaid'],
          },
        },
      },
    },
  },

  output: 'static', // Static site for GitHub Pages
  trailingSlash: 'ignore',
});
