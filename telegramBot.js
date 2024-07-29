const TelegramBot = require('node-telegram-bot-api');
const axios = require('axios');

// Telegram bot tokeniniz
const token = '';

// Bot oluşturuluyor
const bot = new TelegramBot(token, { polling: true });

// '/echo' komutunu işleyen fonksiyon
bot.onText(/\/echo (.+)/, (msg, match) => {
  const chatId = msg.chat.id;
  const resp = match[1]; // Alınan "whatever" kısmı
  bot.sendMessage(chatId, resp); // Geri gönderme
});

// Herhangi bir mesajı dinleyen fonksiyon
bot.on('message', async (msg) => {
  const chatId = msg.chat.id;
  const text = msg.text;

  // Telegram API'ye post request yaparak Flask API'sine mesajı gönderme
  try {
    const response = await axios.post('https://4e9c-195-155-170-199.ngrok-free.app/predict', { question: text });
    const answer = response.data.answer;
    bot.sendMessage(chatId, answer);
  } catch (error) {
    console.error('Error:', error);
    bot.sendMessage(chatId, 'Bir hata oluştu, lütfen daha sonra tekrar deneyiniz.');
  }
});

console.log('Bot çalışıyor...');
