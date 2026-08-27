let currentMode = 'hybrid';
let activeSources = [];

// Elements
const btnModeHybrid = document.getElementById('btnModeHybrid');
const btnModeDense = document.getElementById('btnModeDense');
const btnDocActions = document.getElementById('btnDocActions');
const docMenu = document.getElementById('docMenu');
const fileInput = document.getElementById('fileInput');
const btnViewDocs = document.getElementById('btnViewDocs');
const btnConfirmClear = document.getElementById('btnConfirmClear');
const confirmModal = document.getElementById('confirmModal');
const btnCancelClear = document.getElementById('btnCancelClear');
const btnExecuteClear = document.getElementById('btnExecuteClear');
const docListModal = document.getElementById('docListModal');
const btnCloseDocList = document.getElementById('btnCloseDocList');
const docListUl = document.getElementById('docListUl');

const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const chatContainer = document.getElementById('chatContainer');
const citationDrawer = document.getElementById('citationDrawer');
const closeDrawer = document.getElementById('closeDrawer');
const drawerBody = document.getElementById('drawerBody');

const telemetryLatency = document.getElementById('telemetryLatency');
const telemetryMode = document.getElementById('telemetryMode');
const telemetryScore = document.getElementById('telemetryScore');
const telemetryChunks = document.getElementById('telemetryChunks');

// Toggle Modes (Requirement 6)
btnModeHybrid.addEventListener('click', () => {
  currentMode = 'hybrid';
  btnModeHybrid.classList.add('active');
  btnModeDense.classList.remove('active');
  telemetryMode.innerText = 'Hybrid RRF';
});

btnModeDense.addEventListener('click', () => {
  currentMode = 'dense';
  btnModeDense.classList.add('active');
  btnModeHybrid.classList.remove('active');
  telemetryMode.innerText = 'Dense Only';
});

// Dropdown Toggles (Requirement 5)
btnDocActions.addEventListener('click', (e) => {
  e.stopPropagation();
  docMenu.classList.toggle('hidden');
});
document.addEventListener('click', () => docMenu.classList.add('hidden'));

// Upload Trigger
fileInput.addEventListener('change', async () => {
  if (!fileInput.files.length) return;
  const formData = new FormData();
  for (const file of fileInput.files) formData.append('files', file);

  const bubble = appendMessage('bot-message', 'System', 'Uploading and indexing embeddings into Qdrant...');
  
  try {
    const res = await fetch('/api/v1/upload', { method: 'POST', body: formData });
    const data = await res.json();
    bubble.querySelector('.message-body').innerText = `✅ Successfully indexed ${data.processed_files} document(s) (${data.total_chunks} total chunks).`;
    refreshDocStats();
  } catch (err) {
    bubble.querySelector('.message-body').innerText = '❌ Failed to process documents.';
  }
});

// Clear Collection Confirmation (Requirement 5)
btnConfirmClear.addEventListener('click', () => confirmModal.classList.remove('hidden'));
btnCancelClear.addEventListener('click', () => confirmModal.classList.add('hidden'));
btnExecuteClear.addEventListener('click', async () => {
  confirmModal.classList.add('hidden');
  try {
    await fetch('/api/v1/clear', { method: 'DELETE' });
    appendMessage('bot-message', 'System', '🧹 Knowledge base and vector store have been completely reset.');
    refreshDocStats();
  } catch (err) {
    appendMessage('bot-message', 'System', '❌ Failed to reset collection.');
  }
});

// View Loaded Documents
btnViewDocs.addEventListener('click', async () => {
  try {
    const res = await fetch('/api/v1/documents');
    const data = await res.json();
    docListUl.innerHTML = '';
    if (!data.documents.length) {
      docListUl.innerHTML = '<li>No documents uploaded yet.</li>';
    } else {
      data.documents.forEach(doc => {
        const li = document.createElement('li');
        li.innerText = `📄 ${doc}`;
        docListUl.appendChild(li);
      });
    }
    docListModal.classList.remove('hidden');
  } catch (err) {}
});
btnCloseDocList.addEventListener('click', () => docListModal.classList.add('hidden'));

// Query Execution & Streaming
sendBtn.addEventListener('click', executeQuery);
queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') executeQuery();
});

async function executeQuery() {
  const query = queryInput.value.trim();
  if (!query) return;

  appendMessage('user-message', 'User', query);
  queryInput.value = '';

  const botBubble = appendMessage('bot-message', 'NEXUS AI', '');
  const bodyElem = botBubble.querySelector('.message-body');

  try {
    const response = await fetch('/api/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, mode: currentMode })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop();

      for (const frame of frames) {
        if (frame.startsWith('event: metadata')) {
          const data = JSON.parse(frame.replace('event: metadata\ndata: ', ''));
          activeSources = data.sources;
          
          // Update Telemetry Cards (Requirement 4)
          telemetryLatency.innerText = `${data.telemetry.latency_ms} ms`;
          telemetryScore.innerText = `${data.telemetry.top_score}`;
          
          renderInlineCitations(botBubble, activeSources);
        } else if (frame.startsWith('event: token')) {
          const { token } = JSON.parse(frame.replace('event: token\ndata: ', ''));
          bodyElem.innerText += token;
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }
      }
    }
  } catch (err) {
    bodyElem.innerText = '❌ Connection interrupted or model timeout.';
  }
}

function appendMessage(typeClass, author, text) {
  const div = document.createElement('div');
  div.className = `message-bubble ${typeClass}`;
  div.innerHTML = `
    <div class="message-tag">${author}</div>
    <div class="message-body">${text}</div>
  `;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return div;
}

// Inline Citations (Requirement 3)
function renderInlineCitations(bubbleElement, sources) {
  if (!sources || !sources.length) return;
  const list = document.createElement('div');
  list.className = 'citation-badge-list';

  sources.forEach((s, idx) => {
    const tag = document.createElement('button');
    tag.className = 'cite-tag';
    tag.innerText = `📎 ${s.source} (p.${s.page})`;
    tag.addEventListener('click', () => openSourceDrawer(idx));
    list.appendChild(tag);
  });

  bubbleElement.appendChild(list);
}

// Side Drawer Inspector
function openSourceDrawer(focusIndex = 0) {
  drawerBody.innerHTML = '';
  activeSources.forEach((src, idx) => {
    const card = document.createElement('div');
    card.className = 'drawer-card';
    if (idx === focusIndex) card.style.borderColor = 'var(--teal-primary)';
    card.innerHTML = `
      <div class="drawer-card-top">
        <span>#${idx + 1} ${src.source} | Page ${src.page}</span>
        <span class="drawer-card-score">Score: ${src.score}</span>
      </div>
      <p>"${src.content}"</p>
    `;
    drawerBody.appendChild(card);
  });
  citationDrawer.classList.remove('hidden');
}

closeDrawer.addEventListener('click', () => citationDrawer.classList.add('hidden'));

async function refreshDocStats() {
  try {
    const res = await fetch('/api/v1/documents');
    const data = await res.json();
    telemetryChunks.innerText = data.total_chunks;
  } catch (err) {}
}
refreshDocStats();