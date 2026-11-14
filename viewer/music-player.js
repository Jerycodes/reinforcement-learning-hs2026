// Set this to a cleared, locally hosted audio file, e.g. './audio/theme.ogg'.
// Record the rights and attribution in audio/CREDITS.txt before distribution.
export const soundtrackUrl = null;

export function musicPlayer(ctx, bus, url = soundtrackUrl) {
  if (!url) return () => {};
  const levels = {menu: .85, lobby: .85, countdown: .10, racing: .30, finished: .65};
  const gain = ctx.createGain();
  gain.gain.value = 0;
  gain.connect(bus);
  let source, loading, level = 0;
  return function setPhase(phase) {
    level = levels[phase] ?? 0;
    gain.gain.setTargetAtTime(level, ctx.currentTime, .55);
    if (!level || source || loading) return;
    loading = fetch(url).then(response => {
      if (!response.ok) throw Error(response.statusText);
      return response.arrayBuffer();
    }).then(bytes => ctx.decodeAudioData(bytes)).then(buffer => {
      source = ctx.createBufferSource();
      source.buffer = buffer;
      source.loop = true;
      source.connect(gain);
      gain.gain.setValueAtTime(0, ctx.currentTime);
      gain.gain.setTargetAtTime(level, ctx.currentTime, .55);
      source.start();
    }).catch(error => console.warn('Musik konnte nicht geladen werden.', error));
  };
}
