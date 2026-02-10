import { useCallback, useState, useRef, useEffect } from 'react';

export const useSpeech = () => {
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [isSupported, setIsSupported] = useState(false);
    const utteranceRef = useRef(null);

    useEffect(() => {
        setIsSupported('speechSynthesis' in window);
    }, []);

    const speak = useCallback(async (text) => {
        setIsSpeaking(true);
        try {
            // Try backend TTS first
            if (isSupported) { // Using isSupported to check if we should even try? No, backend is separate.
                // Actually, let's try backend first, fallback to browser.
            }

            // Note: Implementing hybrid approach. 
            // 1. Try backend
            // 2. If backend fails, use browser TTS

            try {
                const { speakText } = await import('@/lib/api');
                const audioBlob = await speakText(text);
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);

                audio.onended = () => {
                    setIsSpeaking(false);
                    URL.revokeObjectURL(audioUrl);
                };
                audio.onerror = () => {
                    throw new Error("Audio playback failed");
                };

                await audio.play();
                return; // Success
            } catch (err) {
                console.warn("Backend TTS failed, falling back to browser TTS", err);
            }

            // Fallback to browser TTS
            if (!isSupported) return;

            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utteranceRef.current = utterance;

            const voices = window.speechSynthesis.getVoices();
            const tamilVoice = voices.find(voice =>
                voice.lang.startsWith('ta') ||
                voice.name.toLowerCase().includes('tamil')
            );

            if (tamilVoice) {
                utterance.voice = tamilVoice;
            }

            utterance.lang = 'ta-IN';
            utterance.rate = 0.8;
            utterance.pitch = 1;
            utterance.volume = 1;

            utterance.onstart = () => setIsSpeaking(true);
            utterance.onend = () => setIsSpeaking(false);
            utterance.onerror = () => setIsSpeaking(false);

            window.speechSynthesis.speak(utterance);
        } catch (e) {
            console.error("Speech error", e);
            setIsSpeaking(false);
        }
    }, [isSupported]);

    const stop = useCallback(() => {
        if (!isSupported) return;
        window.speechSynthesis.cancel();
        setIsSpeaking(false);
    }, [isSupported]);

    // Load voices when available
    useEffect(() => {
        if (!isSupported) return;

        const loadVoices = () => {
            window.speechSynthesis.getVoices();
        };

        loadVoices();
        window.speechSynthesis.onvoiceschanged = loadVoices;

        return () => {
            window.speechSynthesis.onvoiceschanged = null;
        };
    }, [isSupported]);

    return {
        speak,
        stop,
        isSpeaking,
        isSupported
    };
};
