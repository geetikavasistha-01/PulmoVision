/**
 * scripts/test_live_navigation.mjs
 * Connects to live Streamlit server on http://localhost:8501 via Brave CDP,
 * tests clicking each sidebar navigation destination, and logs DOM headers and URL.
 */

import { spawn } from 'child_process';
import { resolve } from 'path';

const BRAVE_BIN = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser';
const PORT = 9333; // Use unique CDP port

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function main() {
  console.log('Launching Brave Browser on port', PORT);
  const browserProc = spawn(BRAVE_BIN, [
    `--remote-debugging-port=${PORT}`,
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--window-size=1440,1100',
    'about:blank'
  ]);

  await sleep(2500);

  try {
    const listRes = await fetch(`http://localhost:${PORT}/json/list`);
    const pages = await listRes.json();
    const wsUrl = pages[0].webSocketDebuggerUrl;
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
    await send('Runtime.enable');

    console.log('Navigating to http://localhost:8501 ...');
    await send('Page.navigate', { url: 'http://localhost:8501' });
    await sleep(4000);

    // Evaluate initial page title
    const evalTitle = async () => {
      const res = await send('Runtime.evaluate', {
        expression: `(() => {
          const h1 = document.querySelector('h1, h2');
          const header = h1 ? h1.innerText : 'NO HEADER';
          const labels = Array.from(document.querySelectorAll('[data-testid="stRadio"] label')).map(l => l.innerText.trim());
          const checked = document.querySelector('[data-testid="stRadio"] input:checked');
          const checkedLabel = checked ? checked.closest('label')?.innerText.trim() : 'NONE';
          return JSON.stringify({ url: window.location.href, header, checkedLabel, labels });
        })()`
      });
      return JSON.parse(res.result.value);
    };

    console.log('Initial page state:', await evalTitle());

    // Click a radio option by text
    const clickOption = async (targetText) => {
      console.log(`\nAttempting to click radio: "${targetText}"...`);
      const clickRes = await send('Runtime.evaluate', {
        expression: `(() => {
          const labels = Array.from(document.querySelectorAll('[data-testid="stRadio"] label'));
          const target = labels.find(l => l.innerText.includes('${targetText}'));
          if (target) {
            target.click();
            return 'CLICKED';
          }
          return 'NOT_FOUND';
        })()`
      });
      console.log('Click result:', clickRes.result.value);
      await sleep(3000);
      const afterState = await evalTitle();
      console.log('State after click:', afterState);
      return afterState;
    };

    await clickOption('Dataset Explorer');
    await clickOption('Model Laboratory');
    await clickOption('Experiment Analytics');
    await clickOption('CT Workstation');
    await clickOption('Explainability');
    await clickOption('Research Documentation');
    await clickOption('Research Overview');

    ws.close();
  } catch (err) {
    console.error('Error during test:', err);
  } finally {
    browserProc.kill();
  }
}

main();
