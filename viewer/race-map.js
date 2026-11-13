export function createRaceMap(element, track) {
  const canvas = element.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const list = element.querySelector('ol');
  const points = track.centerline;
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const left = Math.min(...xs), right = Math.max(...xs);
  const bottom = Math.min(...ys), top = Math.max(...ys);
  const scale = Math.min(184 / (right - left), 122 / (top - bottom));
  const x = v => 100 + (v - (left + right) / 2) * scale;
  const y = v => 69 - (v - (top + bottom) / 2) * scale;
  const route = new Path2D();
  points.forEach(([a,b], i) => i ? route.lineTo(x(a), y(b)) : route.moveTo(x(a), y(b)));
  route.closePath();
  ctx.scale(2, 2);
  let lastLegend = '';
  return (game, states) => {
    element.hidden = game.phase !== 'racing';
    if (element.hidden) return;
    const drivers = game.players.filter(p => !p.waiting);
    const color = p => '#' + p.color.toString(16).padStart(6,'0');
    const legend = drivers.map(p => [p.rank, p.id === game.player_id ? 'Du' : p.name,
      p.finish_time != null ? 'Ziel' : 'R' + Math.min(p.lap + 1, game.race_laps), color(p)]);
    const key = JSON.stringify(legend);
    if (key !== lastLegend) {
      lastLegend = key;
      list.replaceChildren(...legend.map(([rank,name,lap,tint]) => {
        const row = document.createElement('li');
        for (const label of [rank,name,lap]) {
          const cell = document.createElement('span');
          cell.textContent = label;
          row.append(cell);
        }
        row.style.setProperty('--kart-color', tint);
        return row;
      }));
    }
    ctx.clearRect(0,0,200,138);
    ctx.lineJoin = 'round';
    ctx.lineWidth = 4; ctx.strokeStyle = '#819294'; ctx.stroke(route);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
    ctx.beginPath();
    for (const [i,edge] of [track.left_edge,track.right_edge].entries()) {
      ctx[i ? 'lineTo' : 'moveTo'](x(edge[0][0]),y(edge[0][1]));
    }
    ctx.stroke();
    const field = [...states].sort((a,b) => Number(a.id === game.player_id) - Number(b.id === game.player_id));
    for (const kart of field) {
      const driver = drivers.find(p => p.id === kart.id);
      if (!driver || driver.finish_time != null) continue;
      const own = kart.id === game.player_id;
      ctx.beginPath(); ctx.arc(x(kart.x),y(kart.y),own ? 6 : 5,0,Math.PI*2);
      ctx.fillStyle = color(driver); ctx.fill();
      ctx.lineWidth = own ? 2 : 1; ctx.strokeStyle = own ? '#fff' : '#162124'; ctx.stroke();
      ctx.fillStyle = '#101416'; ctx.font = 'bold 9px sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(driver.rank,x(kart.x),y(kart.y)+0.5);
    }
  };
}
