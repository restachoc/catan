// Canvas board renderer. The static layer (sea, tiles, numbers, ports) is drawn once per board
// into an offscreen canvas; each state update only blits it and draws pieces + highlights.

const RES_COLORS = ["#3d8b3d", "#c4643a", "#a8d86b", "#f0c93f", "#9aa3b2", "#e6d5a5"];
const RES_NAMES = ["wood", "brick", "wool", "grain", "ore", "desert"];
const RES_ICONS = ["🌲", "🧱", "🐑", "🌾", "⛰️", ""];
const PLAYER_COLORS = ["#d64541", "#2e6bd6", "#ef8a1c", "#f5f2ea"];
const SQ3 = Math.sqrt(3);

class BoardRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.board = null;
    this.state = null;
    this.targets = [];       // [{kind, id, action, x, y}] clickable highlights
    this.hover = null;
    this.onPick = null;      // callback(action)
    this.staticLayer = document.createElement("canvas");

    new ResizeObserver(() => this.resize()).observe(canvas);
    canvas.addEventListener("mousemove", (e) => {
      const t = this.hit(e);
      if (t !== this.hover) { this.hover = t; this.draw(); }
      canvas.style.cursor = t ? "pointer" : "default";
    });
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.draw(); });
    canvas.addEventListener("click", (e) => {
      const t = this.hit(e);
      if (t && this.onPick) this.onPick(t.action);
    });
  }

  // ------------------------------------------------------------ geometry

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.w = r.width; this.h = r.height; this.dpr = dpr;
    for (const c of [this.canvas, this.staticLayer]) {
      c.width = Math.round(r.width * dpr);
      c.height = Math.round(r.height * dpr);
    }
    // Board spans ~±5.4 hex sizes horizontally and ~±5.3 vertically including ports.
    this.size = Math.min(r.width / 11.2, r.height / 11.0);
    this.cx = r.width / 2; this.cy = r.height / 2;
    this.renderStatic();
    this.draw();
  }

  // Lattice coords -> screen px.
  pt([X, Y]) {
    return [this.cx + X * SQ3 / 2 * this.size, this.cy + Y / 2 * this.size];
  }
  vxy(v) { return this.pt(this.board.vertex_xy[v]); }
  hxy(h) { return this.pt(this.board.hex_center[h]); }
  exy(e) {
    const [a, b] = this.board.edge_vertices[e];
    const [ax, ay] = this.vxy(a), [bx, by] = this.vxy(b);
    return [(ax + bx) / 2, (ay + by) / 2];
  }

  // ------------------------------------------------------------ static layer

  setBoard(board) {
    this.board = board;
    this.renderStatic();
    this.draw();
  }

  renderStatic() {
    if (!this.board || !this.size) return;
    const c = this.staticLayer.getContext("2d");
    const s = this.size;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, this.w, this.h);

    // Island shadow / beach outline.
    c.save();
    c.shadowColor = "rgba(0,0,0,.35)"; c.shadowBlur = s * 0.4;
    for (let h = 0; h < 19; h++) this.hexPath(c, h, 1.06), c.fillStyle = "#e9dcb4", c.fill();
    c.restore();

    // Ports.
    const b = this.board;
    for (let i = 0; i < b.port_edges.length; i++) {
      const [va, vb] = b.edge_vertices[b.port_edges[i]];
      const [ax, ay] = this.vxy(va), [bx, by] = this.vxy(vb);
      const mx = (ax + bx) / 2, my = (ay + by) / 2;
      const dx = mx - this.cx, dy = my - this.cy, dl = Math.hypot(dx, dy);
      const px = mx + dx / dl * s * 0.62, py = my + dy / dl * s * 0.62;
      c.strokeStyle = "#8a6a45"; c.lineWidth = s * 0.07; c.lineCap = "round";
      for (const [x, y] of [[ax, ay], [bx, by]]) {
        c.beginPath(); c.moveTo(x + (px - x) * 0.25, y + (py - y) * 0.25); c.lineTo(px, py); c.stroke();
      }
      const t = b.port_types[i];
      c.beginPath(); c.arc(px, py, s * 0.3, 0, Math.PI * 2);
      c.fillStyle = t === 5 ? "#fffaf0" : RES_COLORS[t]; c.fill();
      c.lineWidth = s * 0.04; c.strokeStyle = "#5b4630"; c.stroke();
      c.fillStyle = "#222"; c.textAlign = "center"; c.textBaseline = "middle";
      c.font = `700 ${s * 0.2}px system-ui`;
      c.fillText(t === 5 ? "3:1" : "2:1", px, py);
    }

    // Tiles.
    for (let h = 0; h < 19; h++) {
      const res = b.hex_res[h];
      this.hexPath(c, h, 0.96);
      const [x, y] = this.hxy(h);
      const g = c.createRadialGradient(x, y - s * 0.3, s * 0.1, x, y, s);
      g.addColorStop(0, shade(RES_COLORS[res], 18));
      g.addColorStop(1, shade(RES_COLORS[res], -12));
      c.fillStyle = g; c.fill();
      c.lineWidth = s * 0.03; c.strokeStyle = "rgba(0,0,0,.25)"; c.stroke();
      c.font = `${s * 0.34}px system-ui`; c.textAlign = "center"; c.textBaseline = "middle";
      c.globalAlpha = 0.55; c.fillText(RES_ICONS[res], x, y - s * 0.52); c.globalAlpha = 1;

      const n = b.hex_num[h];
      if (n) {
        c.beginPath(); c.arc(x, y + s * 0.08, s * 0.3, 0, Math.PI * 2);
        c.fillStyle = "#fbf5e6"; c.fill();
        c.lineWidth = s * 0.02; c.strokeStyle = "rgba(0,0,0,.3)"; c.stroke();
        const red = n === 6 || n === 8;
        c.fillStyle = red ? "#c0262d" : "#2b2a27";
        c.font = `${red ? 800 : 700} ${s * (red ? 0.3 : 0.26)}px system-ui`;
        c.fillText(String(n), x, y + s * 0.04);
        const pips = 6 - Math.abs(7 - n);
        for (let k = 0; k < pips; k++) {
          c.beginPath();
          c.arc(x + (k - (pips - 1) / 2) * s * 0.055, y + s * 0.26, s * 0.02, 0, Math.PI * 2);
          c.fill();
        }
      }
    }
  }

  hexPath(c, h, scale) {
    const [x, y] = this.hxy(h);
    c.beginPath();
    for (let k = 0; k < 6; k++) {
      const [vx, vy] = this.vxy(this.board.hex_vertices[h][k]);
      const px = x + (vx - x) * scale, py = y + (vy - y) * scale;
      k ? c.lineTo(px, py) : c.moveTo(px, py);
    }
    c.closePath();
  }

  // ------------------------------------------------------------ dynamic layer

  setState(state, targets = []) {
    this.state = state;
    this.targets = targets.map((t) => {
      const [x, y] = t.kind === "vertex" ? this.vxy(t.id) : t.kind === "edge" ? this.exy(t.id) : this.hxy(t.id);
      return { ...t, x, y };
    });
    this.hover = null;
    this.draw();
  }

  // Recompute target positions after a resize.
  refreshTargets() {
    for (const t of this.targets) {
      [t.x, t.y] = t.kind === "vertex" ? this.vxy(t.id) : t.kind === "edge" ? this.exy(t.id) : this.hxy(t.id);
    }
  }

  draw() {
    const c = this.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.board || !this.size) return;
    c.drawImage(this.staticLayer, 0, 0);
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.refreshTargets();
    const s = this.size, st = this.state;
    if (!st) return;

    // Robber.
    {
      const [x, y] = this.hxy(st.robber);
      const rx = x - s * 0.42, ry = y + s * 0.05;
      c.fillStyle = "rgba(30,30,34,.92)";
      c.beginPath(); c.arc(rx, ry - s * 0.2, s * 0.1, 0, Math.PI * 2); c.fill();
      c.beginPath();
      c.moveTo(rx - s * 0.13, ry + s * 0.2); c.quadraticCurveTo(rx - s * 0.12, ry - s * 0.12, rx, ry - s * 0.12);
      c.quadraticCurveTo(rx + s * 0.12, ry - s * 0.12, rx + s * 0.13, ry + s * 0.2); c.closePath(); c.fill();
    }

    // Roads.
    c.lineCap = "round";
    st.players.forEach((p, i) => {
      for (const e of p.roads) {
        const [a, b] = this.board.edge_vertices[e];
        const [ax, ay] = this.vxy(a), [bx, by] = this.vxy(b);
        const x1 = ax + (bx - ax) * 0.16, y1 = ay + (by - ay) * 0.16;
        const x2 = bx + (ax - bx) * 0.16, y2 = by + (ay - by) * 0.16;
        c.strokeStyle = "rgba(0,0,0,.55)"; c.lineWidth = s * 0.17;
        c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
        c.strokeStyle = PLAYER_COLORS[i]; c.lineWidth = s * 0.11;
        c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
      }
    });

    // Buildings.
    st.players.forEach((p, i) => {
      for (const v of p.settlements) this.house(c, ...this.vxy(v), s * 0.17, PLAYER_COLORS[i], false);
      for (const v of p.cities) this.house(c, ...this.vxy(v), s * 0.17, PLAYER_COLORS[i], true);
    });

    // Highlights.
    for (const t of this.targets) {
      const hov = t === this.hover;
      c.beginPath();
      const r = t.kind === "hex" ? s * 0.36 : t.kind === "edge" ? s * 0.11 : s * 0.13;
      c.arc(t.x, t.y, hov ? r * 1.35 : r, 0, Math.PI * 2);
      c.fillStyle = hov ? "rgba(255,255,255,.95)" : "rgba(255,255,255,.55)";
      c.fill();
      c.lineWidth = 2; c.strokeStyle = hov ? "#111" : "rgba(0,0,0,.45)"; c.stroke();
    }
  }

  house(c, x, y, r, color, city) {
    c.beginPath();
    if (city) {
      c.moveTo(x - r * 1.3, y + r); c.lineTo(x - r * 1.3, y - r * 0.2); c.lineTo(x - r * 0.6, y - r * 0.9);
      c.lineTo(x + r * 0.1, y - r * 0.2); c.lineTo(x + r * 0.1, y - r * 0.1);
      c.lineTo(x + r * 1.3, y - r * 0.1); c.lineTo(x + r * 1.3, y + r);
    } else {
      c.moveTo(x - r, y + r * 0.9); c.lineTo(x - r, y - r * 0.2); c.lineTo(x, y - r * 1.1);
      c.lineTo(x + r, y - r * 0.2); c.lineTo(x + r, y + r * 0.9);
    }
    c.closePath();
    c.fillStyle = color; c.fill();
    c.lineWidth = Math.max(1.5, r * 0.18); c.strokeStyle = "rgba(0,0,0,.7)"; c.stroke();
  }

  hit(e) {
    if (!this.targets.length) return null;
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = Infinity;
    for (const t of this.targets) {
      const d = Math.hypot(t.x - x, t.y - y);
      const lim = (t.kind === "hex" ? 0.55 : 0.28) * this.size;
      if (d < lim && d < bd) { bd = d; best = t; }
    }
    return best;
  }
}

function shade(hex, pct) {
  const n = parseInt(hex.slice(1), 16);
  const f = (v) => Math.max(0, Math.min(255, Math.round(v + (pct / 100) * 255)));
  const r = f(n >> 16), g = f((n >> 8) & 255), b = f(n & 255);
  return `rgb(${r},${g},${b})`;
}
