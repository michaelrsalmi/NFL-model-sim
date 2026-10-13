const COLORS = {
  ARI:"#97233F",ATL:"#A71930",BAL:"#241773",BUF:"#00338D",CAR:"#0085CA",
  CHI:"#0B162A",CIN:"#FB4F14",CLE:"#E31837",DAL:"#003594",DEN:"#FB4F14",
  DET:"#0076B6",GB:"#203731",HOU:"#03202F",IND:"#002C5F",JAX:"#006778",
  KC:"#E31837",LV:"#5a5a5a",LAC:"#0080C6",LAR:"#003594",MIA:"#008E97",
  MIN:"#4F2683",NE:"#002244",NO:"#D3BC8D",NYG:"#0B2265",NYJ:"#125740",
  PHI:"#004C54",PIT:"#FFB612",SEA:"#002244",SF:"#AA0000",TB:"#D50A0A",
  TEN:"#0C2340",WAS:"#5A1414"
};
const TEAMS = Object.keys(COLORS).sort();
let CUR_WEEK = 5;

const $ = id => document.getElementById(id);
const dot = t => `<span class="tdot" style="background:${COLORS[t]};color:${COLORS[t]}"></span>`;
const pct = p => (p*100).toFixed(1)+"%";
const odds = o => o==null ? "--" : (o<0 ? ""+o : "+"+o);
const fdate = d => { try { return new Date(d).toLocaleString("en-US",{weekday:"short",month:"short",day:"numeric",hour:"numeric",minute:"2-digit"}); } catch(e){ return d; } };

function probbar(a, pa, h, ph) {
  return `<div class="probbar">
    <div class="seg" style="width:${(pa*100).toFixed(1)}%;background:${COLORS[a]}">${pct(pa)}</div>
    <div class="seg" style="width:${(ph*100).toFixed(1)}%;background:${COLORS[h]}">${pct(ph)}</div>
  </div>
  <div class="problabels"><span>${dot(a)} ${a}</span><span>${h} ${dot(h)}</span></div>`;
}

async function loadWeek(w) {
  CUR_WEEK = w;
  document.querySelectorAll("#weeks button").forEach(b =>
    b.classList.toggle("active", +b.dataset.w === w));
  $("cards").innerHTML = `<div class="spinner">Running 10,000 simulations per game…</div>`;
  const r = await fetch(`/api/slate?week=${w}`).then(r => r.json());
  $("cards").innerHTML = "";
  r.games.forEach((g, i) => {
    const el = document.createElement("div");
    el.className = "card";
    let extra = "";
    if (g.done) {
      const hit = (g.winner === g.home && g.p_home > .5) || (g.winner === g.away && g.p_away > .5);
      extra = `<div class="final ${hit ? "hit" : "miss"}">Final ${g.final} — model ${hit ? "✓" : "✗"}</div>`;
    }
    el.innerHTML = `
      <div class="date">${fdate(g.date)}${g.done ? " · Final" : ""}</div>
      <div class="matchup"><span class="team">${dot(g.away)} ${g.away}</span>
        <span class="proj">${g.proj_away.toFixed(1)}</span></div>
      <div class="matchup"><span class="team">${dot(g.home)} ${g.home}</span>
        <span class="proj">${g.proj_home.toFixed(1)}</span></div>
      ${probbar(g.away, g.p_away, g.home, g.p_home)}
      <div class="chips">
        <span class="chip">spread <b>${g.fair_spread > 0 ? "+" : ""}${g.fair_spread.toFixed(1)}</b></span>
        <span class="chip">total <b>${g.fair_total.toFixed(1)}</b></span>
      </div>${extra}`;
    el.onclick = () => openGame(g.away, g.home);
    $("cards").appendChild(el);
  });
}

function tdTable(rows, key) {
  return `<table class="td"><tr><th>Player</th><th>Team</th><th>Prob</th><th>Fair odds</th></tr>` +
    rows.map(p => `<tr><td>${p.name}</td><td>${p.team}</td><td>${(p.prob*100).toFixed(key==="first"?2:1)}%</td><td class="odds">${odds(p.odds)}</td></tr>`).join("") +
    `</table>`;
}

function detailHTML(r) {
  return `<div class="detail">
    <h2 style="letter-spacing:2px">${r.away} @ ${r.home}</h2>
    <div class="scoreline">${r.away} ${r.proj_away.toFixed(1)} — ${r.proj_home.toFixed(1)} ${r.home}</div>
    <div class="big-prob">
      <div class="seg" style="width:${(r.p_away*100).toFixed(1)}%;background:${COLORS[r.away]}">${r.away} ${pct(r.p_away)}</div>
      <div class="seg" style="width:${(r.p_home*100).toFixed(1)}%;background:${COLORS[r.home]}">${r.home} ${pct(r.p_home)}</div>
    </div>
    <div class="statgrid">
      <div class="stat"><div class="k">Fair spread</div><div class="v">${r.fair_spread>0?"+":""}${r.fair_spread.toFixed(1)}</div></div>
      <div class="stat"><div class="k">Fair total</div><div class="v">${r.fair_total.toFixed(1)}</div></div>
      <div class="stat"><div class="k">Elo ${r.home}</div><div class="v">${r.elo_home.toFixed(0)}</div></div>
      <div class="stat"><div class="k">Elo ${r.away}</div><div class="v">${r.elo_away.toFixed(0)}</div></div>
    </div>
    <div class="tabs">
      <button class="active" data-t="any">Anytime TD</button>
      <button data-t="first">First TD</button>
    </div>
    <div id="td-body">${tdTable(r.anytime, "any")}</div>
  </div>`;
}

async function openGame(away, home) {
  $("modal-body").innerHTML = `<div class="spinner">Simulating ${away} @ ${home}…</div>`;
  $("modal").classList.remove("hidden");
  const r = await fetch(`/api/predict?away=${away}&home=${home}`).then(r => r.json());
  $("modal-body").innerHTML = detailHTML(r);
  $("modal-body").querySelectorAll(".tabs button").forEach(b => {
    b.onclick = () => {
      $("modal-body").querySelectorAll(".tabs button").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      $("modal-body").querySelector("#td-body").innerHTML =
        b.dataset.t === "any" ? tdTable(r.anytime, "any") : tdTable(r.first_td, "first");
    };
  });
}
$("modal-close").onclick = () => $("modal").classList.add("hidden");
$("modal").onclick = e => { if (e.target.id === "modal") $("modal").classList.add("hidden"); };

async function runLab() {
  const a = $("lab-away").value, h = $("lab-home").value;
  if (a === h) { $("lab-out").innerHTML = `<p class="dim">Pick two different teams.</p>`; return; }
  $("lab-run").disabled = true;
  $("lab-out").innerHTML = `<div class="spinner">Running 20,000 simulations…</div>`;
  try {
    const r = await fetch(`/api/predict?away=${a}&home=${h}`).then(r => r.json());
    $("lab-out").innerHTML = detailHTML(r);
    $("lab-out").querySelectorAll(".tabs button").forEach(b => {
      b.onclick = () => {
        $("lab-out").querySelectorAll(".tabs button").forEach(x => x.classList.remove("active"));
        b.classList.add("active");
        $("lab-out").querySelector("#td-body").innerHTML =
          b.dataset.t === "any" ? tdTable(r.anytime, "any") : tdTable(r.first_td, "first");
      };
    });
  } finally { $("lab-run").disabled = false; }
}

(function init() {
  const nav = $("weeks");
  for (let w = 1; w <= 18; w++) {
    const b = document.createElement("button");
    b.textContent = "Week " + w;
    b.dataset.w = w;
    b.onclick = () => loadWeek(w);
    nav.appendChild(b);
  }
  TEAMS.forEach(t => {
    $("lab-away").appendChild(new Option(t, t));
    $("lab-home").appendChild(new Option(t, t));
  });
  $("lab-away").value = "BUF"; $("lab-home").value = "LAR";
  $("lab-run").onclick = runLab;
  loadWeek(CUR_WEEK);
})();
