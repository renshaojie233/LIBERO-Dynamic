'use strict';
// Tabs use one active tab stop and support keyboard navigation.
function bindTabs(tabs, activate) {
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(index));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault(); activate(next); tabs[next].focus();
    });
  });
}
function selectTab(tabs, index) {
  tabs.forEach((tab, i) => {
    tab.setAttribute('aria-selected', String(i === index));
    tab.tabIndex = i === index ? 0 : -1;
  });
}
// Initialize media before any task source is loaded. WebKit needs both the
// muted attribute and property, and may defer offscreen autoplay until visible.
const videos = [...document.querySelectorAll('video')];
function startVideo(video) {
  if (document.hidden || video.error || !video.paused) return;
  video.muted = true;
  video.defaultMuted = true;
  const request = video.play();
  if (request) request.catch(error => {
    if (error.name !== 'AbortError') video.dataset.playbackError = error.name;
  });
}
videos.forEach(video => {
  video.muted = true;
  video.defaultMuted = true;
  video.autoplay = true;
  video.playsInline = true;
  video.setAttribute('muted', '');
  video.setAttribute('playsinline', '');
  video.setAttribute('webkit-playsinline', '');
  video.addEventListener('canplay', () => startVideo(video));
  video.addEventListener('playing', () => { delete video.dataset.playbackError; });
});
const playbackObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => { if (entry.isIntersecting) startVideo(entry.target); });
}, {threshold: 0.01});
videos.forEach(video => playbackObserver.observe(video));
window.addEventListener('pageshow', () => videos.forEach(startVideo));
// If the browser requires a gesture, any normal page interaction retries media.
document.addEventListener('pointerup', event => {
  if (!event.target.closest('video')) videos.forEach(startVideo);
});
document.addEventListener('keydown', event => {
  if (!event.target.closest('video')) videos.forEach(startVideo);
});
const categories = JSON.parse(document.querySelector('#task-data').textContent);
const categoryTabs = [...document.querySelectorAll('.category-card')];
const taskSelect = document.querySelector('#task-select');
const taskVideo = document.querySelector('#task-video');
let categoryIndex = 0;
let taskIndex = 0;
function showTask(index) {
  const tasks = categories[categoryIndex].tasks;
  taskIndex = (index + tasks.length) % tasks.length;
  const task = tasks[taskIndex];
  taskVideo.pause();
  taskVideo.src = task.src;
  taskVideo.poster = task.poster;
  taskVideo.setAttribute('aria-label', task.name);
  taskVideo.load();
  startVideo(taskVideo);
  taskSelect.value = String(taskIndex);
  document.querySelector('#task-name').textContent = task.name;
  document.querySelector('#task-kicker').textContent = task.kicker;
  document.querySelector('#task-position').textContent = `${taskIndex + 1} / ${tasks.length}`;
}
function showCategory(index) {
  categoryIndex = index;
  selectTab(categoryTabs, index);
  document.querySelector('#task-panel').setAttribute('aria-labelledby', categoryTabs[index].id);
  taskSelect.replaceChildren(...categories[index].tasks.map((task, i) => new Option(`${String(i + 1).padStart(2, '0')} · ${task.name}`, String(i))));
  showTask(0);
}
bindTabs(categoryTabs, showCategory);
taskSelect.addEventListener('change', () => showTask(Number(taskSelect.value)));
document.querySelector('#previous-task').addEventListener('click', () => showTask(taskIndex - 1));
document.querySelector('#next-task').addEventListener('click', () => showTask(taskIndex + 1));
showCategory(0);
videos.forEach(startVideo);
document.querySelectorAll('.real-playlist').forEach(video => {
  const playlist = JSON.parse(video.dataset.playlist);
  const card = video.closest('.real-card');
  const select = card.querySelector('.trial-select');
  const status = card.querySelector('.playlist-status');
  let index = 0;
  const show = (next, resume) => {
    index = next;
    video.src = playlist[index];
    // The poster follows the selected trial; it never shows a different recording.
    video.poster = `assets/site/posters/${playlist[index].split('/').pop().replace('.mp4', '.jpg')}`;
    status.textContent = `Trial ${index + 1} / ${playlist.length}`;
    if (select) select.value = String(index);
    video.load();
    if (resume) startVideo(video);
  };
  if (select) select.addEventListener('change', () => show(Number(select.value), true));
  video.addEventListener('ended', () => show((index + 1) % playlist.length, true));
});
document.addEventListener('visibilitychange', () => {
  videos.forEach(video => {
    if (document.hidden) video.pause();
    else startVideo(video);
  });
});
