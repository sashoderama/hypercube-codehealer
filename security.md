# 🔐 Security Policy – Hypercube CodeHealer

## 📬 Reporting Vulnerabilities

Please report security issues to: **security@hypercube.ai**

We will triage and respond within 48 hours. Please include:
- The affected file or function
- Steps to reproduce
- Potential impact

## ⚠️ Usage Advisory

This project **executes untrusted code** in a sandboxed inference engine.

### Safe Deployment Guidelines:

- Always run inside a **container** (e.g., Docker)
- Use **`--security-opt seccomp=seccomp.json`**
- Never run as root
- Disable network access if healing external inputs

## 🔒 Threat Surface

CodeHealer uses:
- AST analysis to detect unsafe operations
- Redis for rate limiting (default: 10 requests/day)
- No dynamic `eval()` or reflection is used
- Telemetry is disabled by default in commercial deployments

## 🛑 Legal

We are **not liable** for improper usage of this code in production or misuse against live infrastructure. Use ethically and with caution.
