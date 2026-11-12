/**
 * scripts/take_screenshots.mjs
 * Captures pixel-perfect screenshots of PulmoVision Streamlit pages via Chrome DevTools Protocol.
 * Uses native Node 22 WebSockets and fetch - zero external dependencies.
 */

import { spawn } from 'child_process';
import { writeFileSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';

const BRAVE_BIN = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser';
const PORT = 9222;

const PAGES = [
  { name: '01_overview_dark.png', url: 'http://localhost:8501/?theme=dark&page=Research+Overview' },
  { name: '02_overview_light.png', url: 'http://localhost:8501/?theme=light&page=Research+Overview' },
  { name: '03_dataset_explorer.png', url: 'http://localhost:8501/?theme=dark&page=Dataset+Explorer' },
  { name: '04_model_laboratory.png', url: 'http://localhost:8501/?theme=dark&page=Model+Laboratory' },
  { name: '05_experiment_analytics.png', url: 'http://localhost:8501/?theme=dark&page=Experiment+Analytics' },
  { name: '06_ct_workstation.png', url: 'http://localhost:8501/?theme=dark&page=CT+Workstation' },
  { name: '07_explainability.png', url: 'http://localhost:8501/?theme=dark&page=Explainability' },
  { name: '08_research_documentation.png', url: 'http://localhost:8501/?theme=dark&page=Research+Documentation' }
];

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function main() {
  console.log('Launching headless browser with CDP on port', PORT);
  const browserProc = spawn(BRAVE_BIN, [
    `--remote-debugging-port=${PORT}`,
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--window-size=1440,1100',
    'about:blank'
  ]);

  // Give browser a moment to initialize
  await sleep(3000);

  try {
    const listRes = await fetch(`http://localhost:${PORT}/json/list`);
    const pages = await listRes.json();
    const wsUrl = pages[0].webSocketDebuggerUrl;
    console.log('Connected to CDP at:', wsUrl);

    const ws = new WebSocket(wsUrl);

    let id = 1;
    const callbacks = new Map();

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && callbacks.has(msg.id)) {
        callbacks.get(msg.id)(msg.result);
        callbacks.delete(msg.id);
      }
    };

    await new Promise((res) => (ws.onopen = res));

    function send(method, params = {}) {
      return new Promise((res) => {
        const curId = id++;
        callbacks.set(curId, res);
        ws.send(JSON.stringify({ id: curId, method, params }));
      });
    }

    await send('Page.enable');
    await send('Emulation.setDeviceMetricsOverride', {
      width: 1440,
      height: 1100,
      deviceScaleFactor: 1,
      mobile: false
    });

    const outDir = resolve('reports/figures');
    mkdirSync(outDir, { recursive: true });

    for (const page of PAGES) {
      console.log(`Navigating to ${page.name} (${page.url})...`);
      await send('Page.navigate', { url: page.url });
      
      // Wait for Streamlit to connect, render components and Altair charts
      await sleep(3500);

      const shotRes = await send('Page.captureScreenshot', {
        format: 'png',
        captureBeyondViewport: false
      });

      if (shotRes && shotRes.data) {
        const buffer = Buffer.from(shotRes.data, 'base64');
        const filePath = resolve(outDir, page.name);
        writeFileSync(filePath, buffer);
        console.log(`Saved ${page.name} (${buffer.length} bytes)`);
      } else {
        console.error(`Failed to capture ${page.name}`);
      }
    }

    ws.close();
  } finally {
    browserProc.kill();
  }
}

main().catch((err) => {
  console.error('Screenshot capture failed:', err);
  process.exit(1);
});

