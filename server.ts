import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { spawn } from "child_process";
import fs from "fs";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API endpoint for RSA solving
  app.post("/api/solve", (req, res) => {
    const { n, bit_length } = req.body;

    if (typeof n !== "string" || !/^[1-9][0-9]{0,616}$/.test(n)) {
      return res.status(400).json({ error: "N must be a positive decimal string of at most 617 digits" });
    }

    const bits = bit_length ?? 16;
    if (bits !== 16 && bits !== 2048) {
      return res.status(400).json({ error: "Unsupported bit_length" });
    }
    console.log(`Solving for N (${bits}-bit): ${n}`);

    // Pass the decimal string as an argument without invoking a shell.
    const child = spawn(process.env.PYTHON_BIN ?? "python3", ["solver_api.py", n, String(bits)]);
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", chunk => { stdout += chunk; });
    child.stderr.on("data", chunk => { stderr += chunk; });
    child.on("error", error => {
      console.error("Solver launch error:", error);
      if (!res.headersSent) res.status(500).json({ error: "Solver unavailable" });
    });
    child.on("close", code => {
      if (res.headersSent) return;
      if (code !== 0) {
        console.error("Solver error:", stderr);
        return res.status(500).json({ error: "Solver failed" });
      }
      try {
        const result = JSON.parse(stdout);
        if (result.error) {
          return res.status(500).json({ error: result.error });
        }
        res.json(result);
      } catch (e) {
        console.error("Solver returned invalid JSON:", e);
        res.status(500).json({ error: "Failed to parse solver output" });
      }
    });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
