/**
 * scripts/test_live_card_nav.mjs
 * Tests clicking an Overview page card ("Open Dataset Explorer") in live browser,
 * verifies navigation to Dataset Explorer, and then clicks sidebar back to Research Overview.
 */

import { spawn } from 'child_process';

const BRAVE_BIN = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser';
const PORT = 9334;

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function main() {
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

    await send('Page.navigate', { url: 'http://localhost:8501' });
    await sleep(4000);

    const evalTitle = async () => {
      const res = await send('Runtime.evaluate', {
        expression: `(() => {
          const h1 = document.querySelector('h1, h2');
          const header = h1 ? h1.innerText : 'NO HEADER';
          const checked = document.querySelector('[data-testid="stRadio"] input:checked');
          const checkedLabel = checked ? checked.closest('label')?.innerText.trim() : 'NONE';
          return JSON.stringify({ url: window.location.href, header, checkedLabel });
        })()`
      });
      return JSON.parse(res.result.value);
    };

    console.log('1. Initial page:', await evalTitle());

    // Click "Open Dataset Explorer" button
    console.log('\n2. Clicking "Open Dataset Explorer" card button...');
    await send('Runtime.evaluate', {
      expression: `(() => {
        const buttons = Array.from(document.querySelectorAll('button'));
        const btn = buttons.find(b => b.innerText.includes('Open Dataset Explorer'));
        if (btn) btn.click();
      })()`
    });

    await sleep(3500);
    const cardNavState = await evalTitle();
    console.log('State after card click:', cardNavState);

    // Now click sidebar back to Research Overview
    console.log('\n3. Clicking sidebar "Research Overview"...');
    await send('Runtime.evaluate', {
      expression: `(() => {
        const labels = Array.from(document.querySelectorAll('[data-testid="stRadio"] label'));
        const target = labels.find(l => l.innerText.includes('Research Overview'));
        if (target) target.click();
      })()`
    });

    await sleep(3500);
    const backNavState = await evalTitle();
    console.log('State after returning to Overview:', backNavState);

    ws.close();
  } catch (err) {
    console.error('Error:', err);
  } finally {
    browserProc.kill();
  }
}

main();
