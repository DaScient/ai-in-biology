/** @jsxImportSource preact */
import { useState, useEffect } from 'preact/hooks';

interface NotebookLauncherProps {
  notebookPath: string;
  title: string;
}

export default function NotebookLauncher({ notebookPath, title }: NotebookLauncherProps) {
  const [mode, setMode] = useState<'view' | 'edit' | 'run'>('view');
  const [loading, setLoading] = useState(true);
  const [, setContent] = useState<unknown>(null);

  useEffect(() => {
    // Load notebook content
    fetch(`/notebooks/${notebookPath}`)
      .then(res => res.json())
      .then(data => {
        setContent(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [notebookPath]);

  if (loading) {
    return <div class="notebook-loading">Loading notebook...</div>;
  }

  return (
    <div class="notebook-container">
      <div class="notebook-toolbar">
        <h3>{title}</h3>
        <div class="notebook-controls">
          <button
            class={mode === 'view' ? 'active' : ''}
            onClick={() => setMode('view')}
          >
            📖 View
          </button>
          <button
            class={mode === 'edit' ? 'active' : ''}
            onClick={() => setMode('edit')}
          >
            ✏️ Edit
          </button>
          <button
            class={mode === 'run' ? 'active' : ''}
            onClick={() => setMode('run')}
          >
            ▶️ Run in Browser
          </button>
        </div>
        <div class="notebook-actions">
          <a
            href={`https://colab.research.google.com/github/DaScient/ai-in-biological-sciences/blob/main/notebooks/${notebookPath}`}
            target="_blank"
            rel="noopener noreferrer"
            class="colab-link"
          >
            <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" />
          </a>
          <a
            href={`https://mybinder.org/v2/gh/DaScient/ai-in-biological-sciences/main?filepath=notebooks%2F${notebookPath}`}
            target="_blank"
            rel="noopener noreferrer"
            class="binder-link"
          >
            <img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder" />
          </a>
        </div>
      </div>

      {mode === 'run' && (
        <div class="thebe-container" data-thebe>
          <div class="notebook-preview">
            <div class="markdown-cell">
              <h4>Live Notebook Environment</h4>
              <p>
                <strong>📢 Note:</strong> This runs Python in your browser via
                <a href="https://jupyterlite.readthedocs.io/" target="_blank" rel="noopener noreferrer"> JupyterLite</a>.
                No server required!
              </p>
              <button id="activate-thebe" class="btn-secondary">
                🚀 Activate Interactive Mode
              </button>
            </div>
          </div>
        </div>
      )}

      {mode === 'view' && (
        <div class="notebook-preview">
          <div class="markdown-cells">
            {/* This would be populated with rendered markdown from the notebook */}
            <div class="alert alert-info">
              <strong>📘 Full notebook content available on GitHub</strong>
              <p>
                For the complete interactive experience, open this notebook in
                Google Colab or Binder using the buttons above.
              </p>
            </div>
          </div>
        </div>
      )}

      {mode === 'edit' && (
        <div class="notebook-preview">
          <div class="alert alert-info">
            <strong>✏️ Edit on GitHub</strong>
            <p>
              Open the notebook on GitHub to propose changes via pull request.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
