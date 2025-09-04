// Timestamp formatting utility for blog/article cards
// Formats elements: <span class="timestamp" data-article-id><span class="timestamp-text" data-initial></span></span>

(function () {
  'use strict';

  var DEFAULT_LOCALE = 'en-US';
  var DEFAULT_TIMEZONE = 'Asia/Nicosia';

  function isIsoLike(value) {
    return typeof value === 'string' && /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(value);
  }

  function parseDate(value) {
    if (!value) return null;
    try {
      // Normalize Z issue if present
      var normalized = value.replace('Z', '+00:00');
      var date = new Date(normalized);
      return isNaN(date.getTime()) ? null : date;
    } catch (_) {
      return null;
    }
  }

  function formatTimestamp(date) {
    if (!date) return 'Invalid date';
    var options = {
      year: 'numeric',
      month: 'long',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
      timeZone: DEFAULT_TIMEZONE
    };
    return date.toLocaleString(DEFAULT_LOCALE, options);
  }

  function humanizeDiff(ms) {
    var sec = Math.floor(ms / 1000);
    if (sec < 60) return sec + 's ago';
    var min = Math.floor(sec / 60);
    if (min < 60) return min + 'm ago';
    var hr = Math.floor(min / 60);
    if (hr < 24) return hr + 'h ago';
    var d = Math.floor(hr / 24);
    return d + 'd ago';
  }

  function updateTimestampElement(wrapper) {
    if (!wrapper) return;
    var textEl = wrapper.querySelector('.timestamp-text');
    if (!textEl) return;

    var initial = textEl.getAttribute('data-initial') || textEl.textContent || '';
    var date = parseDate(initial);
    if (!date) return;

    textEl.textContent = formatTimestamp(date);
    try {
      textEl.title = humanizeDiff(Date.now() - date.getTime());
    } catch (_) {}
  }

  function updateAllTimestamps() {
    var wrappers = document.querySelectorAll('.timestamp');
    for (var i = 0; i < wrappers.length; i++) {
      updateTimestampElement(wrappers[i]);
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    updateAllTimestamps();
    // Refresh relative tooltip roughly every minute
    setInterval(updateAllTimestamps, 60 * 1000);
  });

  // Expose small API for other scripts if needed
  window.TimestampUtils = {
    updateAll: updateAllTimestamps,
    format: function (isoString) { return formatTimestamp(parseDate(isoString)); }
  };
})();


