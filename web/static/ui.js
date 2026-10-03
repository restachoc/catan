// Game UI: websocket session, side panels, and mapping legal action ids to board clicks/buttons.

const $ = (id) => document.getElementById(id);
const renderer = new BoardRenderer($("board"));
let A = null;            // action offsets from /api/meta
let NAMES = [];
let ws = null;
let state = null;
let names = [];          // display names per seat
let picker = null;       // active resource picker {kind, sel}
let replay = null;       // {frames, log, i, timer}

const YOP_PAIRS = [];
for (let a = 0; a < 5; a++) for (let b = a; b < 5; b++) YOP_PAIRS.push([a, b]);
const tradeId = (give, get) => A.TRADE + give * 4 + (get > give ? get - 1 : get);
// Player offers: kind 0 = 1:1, 1 = give 2 get 1, 2 = give 1 get 2 (mirrors offer_terms in actions.rs).
const OFFER_KINDS = [[1, 1], [2, 1], [1, 2]];
const offerId = (kind, give, get) => A.OFFER + kind * 20 + give * 4 + (get > give ? get - 1 : get);
const offerText = (o) => {
  const side = (v) => v.map((c, r) => (c ? `${c} ${RES_NAMES[r]}` : "")).filter(Boolean).join(" + ");
  return `${side(o.give)} for ${side(o.get)}`;
};
const DEV_NAMES = ["Knight", "Victory Pt", "Road Build", "Year of Plenty", "Monopoly"];

async function init() {
  const meta = await (await fetch("/api/meta")).json();
  A = meta.actions; NAMES = meta.action_names;
  $("opt-bot").innerHTML = meta.bots.map((b) => `<option ${b === "heuristic" ? "selected" : ""}>${b}</option>`).join("");
  renderer.onPick = send;
  $("btn-new").onclick = newGame;
  $("opt-speed").oninput = () => ws && ws.readyState === 1 && ws.send(JSON.stringify({ type: "speed", delay: botDelay() }));
  $("opt-replay").onchange = (e) => e.target.value && loadReplay(e.target.value);
  $("opt-replay").onfocus = refreshReplays;
  await refreshReplays();
  const q = new URLSearchParams(location.search);
  if (q.get("replay")) {
    await loadReplay(q.get("replay"));
    replayGo(Number(q.get("step") || 0));
  } else newGame();
}

const botDelay = () => Number($("opt-speed").value) / 1000;

// ------------------------------------------------------------------ live game

function newGame() {
  stopReplay();
  if (ws) ws.close();
  $("log").innerHTML = "";
  $("banner").hidden = true;
  const n = Number($("opt-players").value);
  const bot = $("opt-bot").value;
  names = ["You", ...Array.from({ length: n - 1 }, (_, i) => `${bot} ${i + 1}`)];
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => ws.send(JSON.stringify({
    type: "new", n_players: n, bot, seed: $("opt-seed").value || null, delay: botDelay(),
  }));
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "board") { renderer.setBoard(msg.board); $("opt-seed").placeholder = `random (${msg.seed})`; }
    else if (msg.type === "state") { if (msg.log) addLog(msg.log); show(msg.state, true); }
    else if (msg.type === "error") console.warn(msg.message);
  };
}

function send(action) {
  if (replay || !ws) return;
  picker = null;
  ws.send(JSON.stringify({ type: "action", action }));
}

// ------------------------------------------------------------------ rendering

function show(st, interactive) {
  state = st;
  const legal = new Set(interactive ? st.legal : []);
  renderer.setState(st, boardTargets(legal));
  renderPlayers(st);
  renderHand(st, legal);
  renderControls(st, legal);
  renderPrompt(st, legal);
  if (st.winner >= 0 || st.phase === "GameOver") {
    const b = $("banner");
    b.hidden = false;
    b.textContent = st.winner < 0 ? "Draw (turn limit)" : st.winner === 0 && !replay ? "🎉 You win!" : `${names[st.winner]} wins`;
  }
}

function boardTargets(legal) {
  const t = [];
  for (const a of legal) {
    if (a >= A.SETTLE && a < A.CITY) t.push({ kind: "vertex", id: a - A.SETTLE, action: a });
    else if (a >= A.CITY && a < A.ROAD) t.push({ kind: "vertex", id: a - A.CITY, action: a });
    else if (a >= A.ROAD && a < A.BUY_DEV) t.push({ kind: "edge", id: a - A.ROAD, action: a });
    else if (a >= A.MOVE_ROBBER && a < A.STEAL) t.push({ kind: "hex", id: a - A.MOVE_ROBBER, action: a });
  }
  return t;
}

function renderPlayers(st) {
  $("players").innerHTML = st.players.map((p, i) => `
    <div class="player ${i === st.actor && st.phase !== "GameOver" ? "active" : ""}">
      <span class="dot" style="background:${PLAYER_COLORS[i]}"></span>
      <span class="name">${names[i] ?? `P${i}`}
        ${st.longest_road === i ? '<span class="badge">Longest road</span>' : ""}
        ${st.largest_army === i ? '<span class="badge">Largest army</span>' : ""}</span>
      <span class="vp">${p.vp ?? p.public_vp}${p.vp != null && p.vp !== p.public_vp ? `<small title="incl. hidden VP cards">*</small>` : ""} VP</span>
      <div class="stats">
        <span>🂠 ${p.cards} cards</span><span>✦ ${p.dev_cards} dev</span>
        <span>⚔ ${p.knights}</span><span>🛣 ${p.road_len}</span>
      </div>
    </div>`).join("")
    + `<div class="stats" style="color:var(--muted);font-size:12px;margin-top:6px">
         Turn ${st.turn} · Bank ${st.bank.join("/")} · Dev deck ${st.dev_deck}
         ${st.last_roll[0] ? ` · Last roll <b>${st.last_roll[0] + st.last_roll[1]}</b>` : ""}</div>`;
}

function renderHand(st, legal) {
  const viewer = replay ? st.actor : 0;
  $("hand-title").textContent = replay ? `Hand · ${names[viewer]}` : "Your hand";
  const me = st.players[viewer];
  const hand = me.hand || [0, 0, 0, 0, 0];
  $("hand").innerHTML = hand.map((c, r) =>
    `<div class="card ${c ? "" : "zero"}" style="background:${RES_COLORS[r]}">${c}<small>${RES_NAMES[r]}</small></div>`).join("");
  const dev = me.dev_hand || [0, 0, 0, 0, 0], nw = me.dev_new || [0, 0, 0, 0, 0];
  $("devcards").innerHTML = dev.map((c, k) => c ? `<div class="card dev ${nw[k] ? "new" : ""}" title="${nw[k] ? `${nw[k]} bought this turn` : ""}">${c}<small>${DEV_NAMES[k]}</small></div>` : "").join("");
}

function button(label, action, legal, extra = "") {
  return `<button data-a="${action}" ${legal.has(action) ? "" : "disabled"} ${extra}>${label}</button>`;
}

function renderControls(st, legal) {
  const c = $("controls");
  const has = (lo, hi) => [...legal].some((a) => a >= lo && a < hi);
  let html = "";
  if (!replay && (st.phase === "Roll" || st.phase === "Main")) {
    html += button("🎲 Roll", A.ROLL, legal);
    html += button("Buy dev card", A.BUY_DEV, legal);
    html += button("Knight", A.PLAY_KNIGHT, legal);
    html += button("Road building", A.PLAY_ROAD_BUILDING, legal);
    html += `<button data-pick="monopoly" ${has(A.PLAY_MONOPOLY, A.PLAY_YOP) ? "" : "disabled"}>Monopoly</button>`;
    html += `<button data-pick="yop" ${has(A.PLAY_YOP, A.MOVE_ROBBER) ? "" : "disabled"}>Year of plenty</button>`;
    html += `<button data-pick="trade" ${has(A.TRADE, A.OFFER) ? "" : "disabled"}>Bank trade</button>`;
    html += button("Offer trade", A.PROPOSE_TRADE, legal);
    html += button("End turn ⏎", A.END_TURN, legal, 'class="primary"');
  }
  c.innerHTML = html;
  c.querySelectorAll("button[data-a]").forEach((b) => (b.onclick = () => send(Number(b.dataset.a))));
  c.querySelectorAll("button[data-pick]").forEach((b) => (b.onclick = () => { picker = { kind: b.dataset.pick, sel: [] }; renderPicker(st, legal); }));
  if (["Discard", "Steal", "OfferTerms", "TradeRespond", "TradeChoose"].includes(st.phase)) {
    if (picker?.kind !== st.phase.toLowerCase()) picker = { kind: st.phase.toLowerCase(), sel: [] };
  } else if (picker && !["monopoly", "yop", "trade"].includes(picker.kind)) picker = null;
  renderPicker(st, legal);
}

function resButtons(filter, onClick, sel = []) {
  return RES_NAMES.slice(0, 5).map((n, r) => {
    const ok = filter(r);
    return `<button class="res-btn ${sel.includes(r) ? "sel" : ""}" data-r="${r}" ${ok ? "" : "disabled"} style="background:${RES_COLORS[r]}">${n}</button>`;
  }).join("");
}

function renderPicker(st, legal) {
  const el = $("picker");
  if (!picker || !legal.size) { el.innerHTML = ""; return; }
  const bind = (fn) => el.querySelectorAll(".res-btn").forEach((b) => (b.onclick = () => fn(Number(b.dataset.r), b)));
  if (picker.kind === "discard") {
    el.innerHTML = `<div class="row">Discard ${st.discard_need[st.actor]} more:</div><div class="row">${resButtons((r) => legal.has(A.DISCARD + r))}</div>`;
    bind((r) => send(A.DISCARD + r));
  } else if (picker.kind === "steal") {
    const n = st.n_players;
    el.innerHTML = `<div class="row">Steal from:</div><div class="row">` +
      [1, 2, 3].filter((k) => legal.has(A.STEAL + k)).map((k) => {
        const p = (st.actor + k) % n;
        return `<button data-a="${A.STEAL + k}" style="border-color:${PLAYER_COLORS[p]}">${names[p]} (${st.players[p].cards})</button>`;
      }).join("") + "</div>";
    el.querySelectorAll("button[data-a]").forEach((b) => (b.onclick = () => send(Number(b.dataset.a))));
  } else if (picker.kind === "monopoly") {
    el.innerHTML = `<div class="row">Monopoly on:</div><div class="row">${resButtons((r) => legal.has(A.PLAY_MONOPOLY + r))}</div>`;
    bind((r) => send(A.PLAY_MONOPOLY + r));
  } else if (picker.kind === "yop") {
    el.innerHTML = `<div class="row">Take two resources (${picker.sel.map((r) => RES_NAMES[r]).join(" + ") || "pick"}):</div><div class="row">${resButtons(() => true, null, picker.sel)}</div>`;
    bind((r) => {
      picker.sel.push(r);
      if (picker.sel.length === 2) {
        const [a, b] = picker.sel.sort();
        const idx = YOP_PAIRS.findIndex(([x, y]) => x === a && y === b);
        if (legal.has(A.PLAY_YOP + idx)) return send(A.PLAY_YOP + idx);
        picker.sel = [];
      }
      renderPicker(st, legal);
    });
  } else if (picker.kind === "trade") {
    const ratios = st.trade_ratios || [4, 4, 4, 4, 4];
    const give = picker.sel[0];
    el.innerHTML = `<div class="row"><span class="label">Give</span>${resButtons((r) => [0, 1, 2, 3, 4].some((g) => g !== r && legal.has(tradeId(r, g))), null, give != null ? [give] : [])}</div>
      <div class="row" style="color:var(--muted);font-size:12px">Ratios: ${ratios.map((x, r) => `${RES_NAMES[r]} ${x}:1`).join(" · ")}</div>
      <div class="row" id="get-row"><span class="label">Get</span>${give == null ? "<i style='color:var(--muted)'>choose what to give</i>" : resButtons((r) => r !== give && legal.has(tradeId(give, r)))}</div>`;
    el.querySelectorAll(".row:first-child .res-btn").forEach((b) => (b.onclick = () => { picker.sel = [Number(b.dataset.r)]; renderPicker(st, legal); }));
    el.querySelectorAll("#get-row .res-btn").forEach((b) => (b.onclick = () => send(tradeId(give, Number(b.dataset.r)))));
  } else if (picker.kind === "offerterms") {
    const kind = picker.offerKind ?? 0;
    const [gn, rn] = OFFER_KINDS[kind];
    const give = picker.sel[0];
    el.innerHTML = `<div class="row" id="kind-row"><span class="label">Terms</span>${OFFER_KINDS.map(([a, b], k) =>
        `<button data-k="${k}" class="${k === kind ? "sel" : ""}">${a}:${b}</button>`).join("")}</div>
      <div class="row" id="give-row"><span class="label">Give ${gn}</span>${resButtons((r) => [0, 1, 2, 3, 4].some((g) => g !== r && legal.has(offerId(kind, r, g))), null, give != null ? [give] : [])}</div>
      <div class="row" id="get-row"><span class="label">Get ${rn}</span>${give == null ? "<i style='color:var(--muted)'>choose what to give</i>" : resButtons((r) => r !== give && legal.has(offerId(kind, give, r)))}</div>
      <div class="row">${button("Cancel", A.CANCEL_OFFER, legal)}</div>
      <div class="row" style="color:var(--muted);font-size:12px">Offered to everyone; you pick among those who accept. ${3 - (st.offers_made ?? 0)} more offer(s) this turn.</div>`;
    el.querySelectorAll("button[data-a]").forEach((b) => (b.onclick = () => send(Number(b.dataset.a))));
    el.querySelectorAll("#kind-row button").forEach((b) => (b.onclick = () => { picker.offerKind = Number(b.dataset.k); picker.sel = []; renderPicker(st, legal); }));
    el.querySelectorAll("#give-row .res-btn").forEach((b) => (b.onclick = () => { picker.sel = [Number(b.dataset.r)]; renderPicker(st, legal); }));
    el.querySelectorAll("#get-row .res-btn").forEach((b) => (b.onclick = () => send(offerId(kind, give, Number(b.dataset.r)))));
  } else if (picker.kind === "traderespond" && st.offer) {
    el.innerHTML = `<div class="row">${names[st.offer.from]} offers ${offerText(st.offer)}</div><div class="row">` +
      button("Accept", A.ACCEPT_OFFER, legal, 'class="primary"') + button("Decline", A.DECLINE_OFFER, legal) + "</div>";
    el.querySelectorAll("button[data-a]").forEach((b) => (b.onclick = () => send(Number(b.dataset.a))));
  } else if (picker.kind === "tradechoose" && st.offer) {
    const n = st.n_players;
    el.innerHTML = `<div class="row">Trade ${offerText(st.offer)} with:</div><div class="row">` +
      [1, 2, 3].filter((k) => legal.has(A.CHOOSE_PARTNER + k)).map((k) => {
        const p = (st.actor + k) % n;
        return `<button data-a="${A.CHOOSE_PARTNER + k}" style="border-color:${PLAYER_COLORS[p]}">${names[p]}</button>`;
      }).join("") + button("Cancel", A.CANCEL_OFFER, legal) + "</div>";
    el.querySelectorAll("button[data-a]").forEach((b) => (b.onclick = () => send(Number(b.dataset.a))));
  }
}

function renderPrompt(st, legal) {
  const p = $("prompt");
  if (replay || st.phase === "GameOver") { p.textContent = ""; return; }
  if (!legal.size) {
    p.textContent = st.offer ? `${names[st.offer.from]} offers ${offerText(st.offer)}: ${names[st.actor]} is answering…` : `${names[st.actor]} is thinking…`;
    return;
  }
  const msg = {
    SetupSettlement: "Place a settlement", SetupRoad: "Place a road next to it",
    Roll: "Roll the dice (or play a knight)", Main: "Build, trade, or end your turn",
    Discard: `Discard ${st.discard_need[st.actor]} card(s)`, MoveRobber: "Move the robber",
    Steal: "Choose who to steal from", RoadBuilding: `Place ${st.free_roads} free road(s)`,
    OfferTerms: "Choose what to offer", TradeRespond: "Accept or decline the offer", TradeChoose: "Pick a trading partner (or cancel)",
  }[st.phase];
  p.textContent = msg || "";
}

function addLog(entry, cls = "") {
  const li = document.createElement("li");
  li.className = cls;
  li.innerHTML = `<span class="who" style="color:${PLAYER_COLORS[entry.player] === "#f5f2ea" ? "#9a9588" : PLAYER_COLORS[entry.player]}">${names[entry.player] ?? `P${entry.player}`}</span> ${entry.text.replace(/_/g, " ")}`;
  $("log").appendChild(li);
  $("log").scrollTop = $("log").scrollHeight;
  return li;
}

document.addEventListener("keydown", (e) => {
  if (replay) {
    if (e.key === "ArrowRight") replayGo(replay.i + 1);
    if (e.key === "ArrowLeft") replayGo(replay.i - 1);
    return;
  }
  if (!state || e.target.tagName === "INPUT") return;
  const legal = new Set(state.legal);
  if (e.key === "Enter" && legal.has(A.END_TURN)) send(A.END_TURN);
  if ((e.key === " " || e.key === "r") && legal.has(A.ROLL)) { e.preventDefault(); send(A.ROLL); }
  if (e.key === "Escape") { picker = null; renderPicker(state, legal); }
});

// ------------------------------------------------------------------ replays

async function refreshReplays() {
  const list = await (await fetch("/api/replays")).json();
  const cur = $("opt-replay").value;
  $("opt-replay").innerHTML = `<option value="">—</option>` + list.map((n) => `<option ${n === cur ? "selected" : ""}>${n}</option>`).join("");
}

async function loadReplay(name) {
  if (ws) { ws.close(); ws = null; }
  const rep = await (await fetch(`/api/replays/${encodeURIComponent(name)}`)).json();
  stopReplay();
  const n = rep.frames[0].n_players;
  names = rep.players.length ? rep.players.map((p, i) => `${p} (${i})`) : Array.from({ length: n }, (_, i) => `P${i}`);
  replay = { ...rep, i: 0, timer: null };
  $("banner").hidden = true;
  $("log").innerHTML = "";
  replay.items = rep.log.map((e) => { const li = addLog(e); li.style.display = "none"; return li; });
  renderer.setBoard(rep.board);
  const bar = $("replay-bar");
  bar.hidden = false;
  $("replay-slider").max = rep.frames.length - 1;
  $("replay-slider").oninput = (e) => replayGo(Number(e.target.value));
  bar.querySelectorAll("button[data-step]").forEach((b) => (b.onclick = () => replayGo(replay.i + Number(b.dataset.step))));
  $("replay-play").onclick = () => {
    if (replay.timer) { clearInterval(replay.timer); replay.timer = null; $("replay-play").textContent = "▶ play"; return; }
    $("replay-play").textContent = "⏸ pause";
    replay.timer = setInterval(() => replay.i >= replay.frames.length - 1 ? $("replay-play").click() : replayGo(replay.i + 1), Math.max(30, botDelay() * 1000));
  };
  replayGo(0);
}

function replayGo(i) {
  if (!replay) return;
  i = Math.max(0, Math.min(replay.frames.length - 1, i));
  replay.i = i;
  replay.items.forEach((li, k) => { li.style.display = k < i ? "" : "none"; li.classList.toggle("cur", k === i - 1); });
  $("log").scrollTop = $("log").scrollHeight;
  $("replay-slider").value = i;
  $("replay-pos").textContent = `${i} / ${replay.frames.length - 1}`;
  $("banner").hidden = true;
  show(replay.frames[i], false);
}

function stopReplay() {
  if (replay && replay.timer) clearInterval(replay.timer);
  replay = null;
  $("replay-bar").hidden = true;
  $("opt-replay").value = "";
}

init();
