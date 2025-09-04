// Password reset helpers and wiring for forgot_password.html

async function requestPasswordReset(email) {
  if (!email) throw new Error('Email requis');
  const res = await fetch('/api/auth/request-reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: String(email).trim().toLowerCase() })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Erreur envoi du code');
  return data;
}

async function resetPassword(email, code, newPassword) {
  if (!email || !code || !newPassword) throw new Error('Champs requis');
  if (!/^\d{6}$/.test(code)) throw new Error('Code invalide (6 chiffres)');
  if (newPassword.length < 8) throw new Error('Mot de passe trop court');
  const res = await fetch('/api/auth/reset-password', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: String(email).trim().toLowerCase(),
      code: String(code).trim(),
      new_password: newPassword
    })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Erreur réinitialisation');
  return data;
}

// Auto-wire if the page contains the expected forms/fields
document.addEventListener('DOMContentLoaded', () => {
  const emailInput = document.getElementById('email');
  const requestForm = document.getElementById('request-form');
  const requestMsg = document.getElementById('request-msg');
  const resetForm = document.getElementById('reset-form');
  const resetMsg = document.getElementById('reset-msg');

  if (requestForm && emailInput && requestMsg) {
    requestForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      requestMsg.textContent = '';
      try {
        const data = await requestPasswordReset(emailInput.value);
        requestMsg.textContent = data.message || 'Code envoyé (si le compte existe)';
      } catch (err) {
        requestMsg.textContent = err.message || 'Erreur réseau';
      }
    });
  }

  if (resetForm && emailInput && resetMsg) {
    resetForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      resetMsg.textContent = '';
      const codeEl = document.getElementById('code');
      const passEl = document.getElementById('new_password');
      try {
        const data = await resetPassword(emailInput.value, codeEl.value, passEl.value);
        resetMsg.textContent = data.message || 'Mot de passe mis à jour';
      } catch (err) {
        resetMsg.textContent = err.message || 'Erreur réseau';
      }
    });
  }
});


