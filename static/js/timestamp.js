
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.timestamp-text').forEach(element => {
        const timestamp = element.getAttribute('data-initial');
        element.textContent = new Date(timestamp).toLocaleString('en-US', {
            month: 'long', day: 'numeric', year: 'numeric', hour: 'numeric', minute: 'numeric', second: 'numeric'
        });
    });
});
