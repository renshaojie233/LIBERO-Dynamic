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
const videos = [...document.querySelectorAll('video:not(#task-dialog-video)')];
function startVideo(video) {
  if (document.hidden || video.error || !video.paused || video.dataset.userPaused === 'true') return;
  video.muted = true;
  video.defaultMuted = true;
  const request = video.play();
  if (request) request.catch(error => {
    if (error.name !== 'AbortError') video.dataset.playbackError = error.name;
  });
}
const heroVideo = document.querySelector('#dataset-showcase-video');
const heroToggle = document.querySelector('#hero-playback-toggle');
if (heroVideo && heroToggle) {
  const updateHeroToggle = () => {
    const paused = heroVideo.paused;
    heroToggle.textContent = paused ? 'Play ▷' : 'Pause Ⅱ';
    heroToggle.setAttribute('aria-label', paused ? 'Play background video' : 'Pause background video');
  };
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    heroVideo.dataset.userPaused = 'true';
    heroVideo.removeAttribute('autoplay');
    heroVideo.pause();
  }
  heroToggle.addEventListener('click', () => {
    if (heroVideo.paused) {
      delete heroVideo.dataset.userPaused;
      startVideo(heroVideo);
    } else {
      heroVideo.dataset.userPaused = 'true';
      heroVideo.pause();
    }
    updateHeroToggle();
  });
  heroVideo.addEventListener('play', updateHeroToggle);
  heroVideo.addEventListener('pause', updateHeroToggle);
  updateHeroToggle();
}
videos.forEach(video => {
  video.muted = true;
  video.defaultMuted = true;
  video.autoplay = video.dataset.userPaused !== 'true';
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
const taskPanel = document.querySelector('#task-panel');
const galleryToggle = document.querySelector('#gallery-playback');
const taskDialog = document.querySelector('#task-dialog');
const dialogVideo = document.querySelector('#task-dialog-video');
let galleryPaused = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let galleryVideos = [];
function playGalleryVideo(video) {
  if (!galleryPaused && !taskDialog.open && video.dataset.visible === 'true') startVideo(video);
}
const galleryObserver = new IntersectionObserver(entries => {
  entries.forEach(({target: video, isIntersecting}) => {
    video.dataset.visible = String(isIntersecting);
    if (isIntersecting) {
      if (!video.getAttribute('src')) { video.src = video.dataset.src; video.load(); }
      playGalleryVideo(video);
    } else video.pause();
  });
}, {threshold: 0.1});
function updateGalleryToggle() {
  galleryToggle.textContent = galleryPaused ? 'Play demos' : 'Pause demos';
  galleryToggle.setAttribute('aria-pressed', String(galleryPaused));
}
galleryToggle.addEventListener('click', () => {
  galleryPaused = !galleryPaused;
  galleryVideos.forEach(video => galleryPaused ? video.pause() : playGalleryVideo(video));
  updateGalleryToggle();
});
function openTask(task) {
  document.querySelector('#task-dialog-title').textContent = task.name;
  document.querySelector('#task-dialog-kicker').textContent = task.kicker;
  dialogVideo.poster = task.poster;
  dialogVideo.src = task.src;
  dialogVideo.setAttribute('aria-label', task.name);
  taskDialog.showModal();
  galleryVideos.forEach(video => video.pause());
  dialogVideo.load();
  startVideo(dialogVideo);
}
document.querySelector('#task-dialog-close').addEventListener('click', () => taskDialog.close());
taskDialog.addEventListener('click', event => { if (event.target === taskDialog) taskDialog.close(); });
taskDialog.addEventListener('close', () => {
  dialogVideo.pause();
  dialogVideo.removeAttribute('src');
  dialogVideo.load();
  galleryVideos.forEach(playGalleryVideo);
});
function showCategory(index) {
  galleryVideos.forEach(video => {
    galleryObserver.unobserve(video);
    video.pause(); video.removeAttribute('src'); video.load();
  });
  galleryVideos = [];
  selectTab(categoryTabs, index);
  taskPanel.setAttribute('aria-labelledby', categoryTabs[index].id);
  taskPanel.replaceChildren(...categories[index].tasks.map(task => {
    const card = document.createElement('article');
    card.className = 'task-card';
    const preview = document.createElement('button');
    preview.type = 'button'; preview.className = 'task-preview';
    preview.setAttribute('aria-label', `Expand demo: ${task.name}`);
    const video = document.createElement('video');
    video.dataset.src = task.src; video.poster = task.poster;
    video.loop = true; video.muted = true; video.defaultMuted = true; video.playsInline = true;
    video.preload = 'none'; video.setAttribute('muted', ''); video.setAttribute('playsinline', '');
    video.setAttribute('aria-hidden', 'true');
    video.addEventListener('canplay', () => playGalleryVideo(video));
    const expand = document.createElement('span');
    expand.className = 'task-expand'; expand.textContent = '↗'; expand.setAttribute('aria-hidden', 'true');
    preview.append(video, expand);
    preview.addEventListener('click', () => openTask(task));
    const details = document.createElement('div'); details.className = 'task-card-details';
    const meta = document.createElement('div'); meta.className = 'task-card-meta';
    const [id, motion] = task.kicker.split(' · ');
    const number = document.createElement('span'); number.textContent = id;
    const badge = document.createElement('span'); badge.textContent = motion;
    badge.className = 'motion-badge'; badge.dataset.motion = motion.toLowerCase();
    meta.append(number, badge);
    const title = document.createElement('h3'); title.textContent = task.name;
    details.append(meta, title); card.append(preview, details);
    galleryVideos.push(video); galleryObserver.observe(video);
    return card;
  }));
}
bindTabs(categoryTabs, showCategory);
showCategory(0);
updateGalleryToggle();
document.addEventListener('visibilitychange', () => {
  galleryVideos.forEach(video => document.hidden ? video.pause() : playGalleryVideo(video));
  if (document.hidden) dialogVideo.pause();
  else if (taskDialog.open) startVideo(dialogVideo);
});
document.addEventListener('pointerup', () => galleryVideos.forEach(playGalleryVideo));
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
