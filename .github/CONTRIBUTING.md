# Contributing to AI in Biological Sciences

Thank you for your interest in contributing! This guide will help you get started.

## Ways to Contribute

- **Report bugs** — Open a [GitHub issue](https://github.com/DaScient/ai-in-biology/issues)
- **Suggest features** — Use the feature request issue template
- **Fix bugs / add features** — Fork → branch → PR
- **Improve documentation** — Edit files in `docs/source/`
- **Add notebooks** — Submit new Jupyter notebooks to `api/notebooks/`

## Development Setup

```bash
git clone https://github.com/DaScient/ai-in-biology.git
cd ai-in-biology
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"
pre-commit install
```

## Workflow

1. **Fork** the repository on GitHub
2. **Create a branch** from `develop`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Write your code** following the style guide below
4. **Write or update tests** in `api/tests/`
5. **Run tests locally**:
   ```bash
   make test
   ```
6. **Lint your code**:
   ```bash
   make lint
   make format
   ```
7. **Commit** with a descriptive message using [Conventional Commits](https://www.conventionalcommits.org/)
8. **Push** your branch and open a Pull Request against `develop`

## Code Style

- Python code follows [PEP 8](https://pep8.org/) with a 100-character line length
- Formatting enforced by **black** and **ruff**
- Type annotations required for all public functions (enforced by **mypy**)
- Docstrings in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

## Pull Request Guidelines

- PRs should be focused — one feature or fix per PR
- All CI checks must pass before merging
- Request review from at least one maintainer
- Link related issues using `Closes #issue-number`

## Commit Message Format

```
type(scope): short description

Optional longer description

Closes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`

## Reporting Security Vulnerabilities

Please **do not** open public issues for security vulnerabilities. Email **security@dascient.com** instead.

## License

By contributing, you agree that your contributions will be licensed under the same [CC BY-NC-SA 4.0 License](../LICENSE) as the project.
