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
const resultTabs = [...document.querySelectorAll('.result-tabs button')];
const resultPanels = [...document.querySelectorAll('.result-card')];
bindTabs(resultTabs, index => {
  selectTab(resultTabs, index);
  resultPanels.forEach((panel, i) => { panel.hidden = i !== index; });
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
// Playback is opt-in; only one clip runs at a time.
const videos = [...document.querySelectorAll('video')];
videos.forEach(video => video.addEventListener('play', () => {
  videos.forEach(other => { if (other !== video) other.pause(); });
}));
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
    if (resume) video.play().catch(() => {});
  };
  if (select) select.addEventListener('change', () => show(Number(select.value), false));
  video.addEventListener('ended', () => show((index + 1) % playlist.length, true));
});
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => { if (!entry.isIntersecting) entry.target.pause(); });
}, {threshold: 0});
videos.forEach(video => observer.observe(video));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) videos.forEach(video => video.pause());
});
