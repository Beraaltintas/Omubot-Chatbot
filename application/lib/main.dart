import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'OMUBOT',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const QAHomePage(),
    );
  }
}

class QAHomePage extends StatefulWidget {
  const QAHomePage({Key? key}) : super(key: key);

  @override
  QAHomePageState createState() => QAHomePageState();
}

class QAHomePageState extends State<QAHomePage> {
  final TextEditingController _questionController = TextEditingController();
  final List<Map<String, String>> _messages = [];
  bool _isLoading = false;
  late stt.SpeechToText _speech;
  late FlutterTts _flutterTts;

  @override
  void initState() {
    super.initState();
    _speech = stt.SpeechToText();
    _flutterTts = FlutterTts();
    _initializeTts();
  }

  Future<void> _initializeTts() async {
    await _flutterTts.setLanguage("tr-TR");
    await _flutterTts.setPitch(1.0);

    // Sesleri kontrol etmek için mevcut sesleri alın
    var voices = await _flutterTts.getVoices;
    // Türkçe kadın sesi varsa seçin
    var femaleVoice = voices.firstWhere(
        (voice) => voice["locale"] == "tr-TR" && voice["gender"] == "female",
        orElse: () => null);
    if (femaleVoice != null) {
      await _flutterTts.setVoice(femaleVoice["name"]);
    } else {
      print("Kadın sesi bulunamadı, varsayılan ses kullanılacak.");
    }
  }

  Future<void> _getAnswer(String question) async {
    setState(() {
      _isLoading = true;
    });

    final String apiUrl = 'https://4e9c-195-155-170-199.ngrok-free.app/predict';

    try {
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'question': question,
        }),
      );

      if (!mounted) return;

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _messages.add({
            'role': 'user',
            'text': question
          }); // Kullanıcının sorduğu soruyu _messages listesine ekle
          _messages.add({'role': 'bot', 'text': data['answer']});
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Failed to get answer')),
        );
      }
    } catch (error) {
      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('An error occurred: $error')),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Widget _buildMessage(Map<String, String> message) {
    final bool isUser = message['role'] == 'user';
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
            maxWidth: MediaQuery.of(context).size.width *
                0.7), // Maksimum genişliği sınırla
        padding: const EdgeInsets.all(10),
        margin: const EdgeInsets.symmetric(vertical: 5, horizontal: 10),
        decoration: BoxDecoration(
          color: isUser ? Colors.blue[100] : Colors.grey[300],
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(10),
            topRight: Radius.circular(10),
            bottomLeft: isUser ? Radius.circular(10) : Radius.circular(0),
            bottomRight: isUser ? Radius.circular(0) : Radius.circular(10),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Expanded(
              child: Text(
                message['text']!,
                style: const TextStyle(fontSize: 16),
              ),
            ),
            if (!isUser) // Sadece bot mesajları için hoparlör simgesi ekleyin
              IconButton(
                icon: const Icon(Icons.volume_up),
                onPressed: () => _speak(message['text']!),
              ),
          ],
        ),
      ),
    );
  }

  Future<void> _speak(String text) async {
    await _flutterTts.speak(text);
  }

  Future<void> _listen() async {
    if (!_speech.isListening) {
      bool available = await _speech.initialize(
        onStatus: (status) {
          print('Speech recognition status: $status');
        },
        onError: (errorNotification) {
          print('Speech recognition error: $errorNotification');
        },
      );

      if (available) {
        await _speech.listen(
          onResult: (result) {
            setState(() {
              _questionController.text = result.recognizedWords;
            });
            // Cümle tamamlandığında işleme sokmak için kontrol
            if (!result.finalResult) {
              // Cümle tamamlanmamışsa bekleyin veya işlem yapın
            } else {
              _getAnswer(_questionController.text);
              _questionController.clear();
            }
          },
        );
      } else {
        print('Speech recognition not available');
      }
    } else {
      _speech.stop();
    }
  }

  void _handleSubmitted(String value) {
    if (value.isNotEmpty) {
      _getAnswer(value);
      _questionController
          .clear(); // Soru gönderildikten sonra metin kutusunu temizle
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('OMUBOT'),
        backgroundColor: const Color.fromARGB(255, 179, 28, 53),
      ),
      body: Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage(
                "assets/omu_logo.png"), // OMÜ logosunun bulunduğu yer
            fit: BoxFit.cover,
            colorFilter: ColorFilter.mode(
                Colors.white.withOpacity(0.1), BlendMode.dstATop),
          ),
        ),
        child: Column(
          children: <Widget>[
            Expanded(
              child: ListView.builder(
                reverse: true,
                itemCount: _messages.length,
                itemBuilder: (context, index) {
                  return _buildMessage(_messages[_messages.length - 1 - index]);
                },
              ),
            ),
            if (_isLoading)
              const Padding(
                padding: EdgeInsets.all(8.0),
                child: CircularProgressIndicator(),
              ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Row(
                children: <Widget>[
                  Expanded(
                    child: TextField(
                      controller: _questionController,
                      onSubmitted: _handleSubmitted,
                      decoration: const InputDecoration(
                        hintText: 'Enter a question...',
                      ),
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.mic),
                    onPressed: _listen,
                  ),
                  IconButton(
                    icon: const Icon(Icons.send),
                    onPressed: () {
                      if (_questionController.text.isNotEmpty) {
                        _handleSubmitted(_questionController.text);
                      }
                    },
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
