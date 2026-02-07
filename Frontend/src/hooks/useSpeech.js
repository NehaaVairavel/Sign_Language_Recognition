import { useCallback, useState, useRef, useEffect } from 'react';

export const useSpeech = () => {
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [isSupported, setIsSupported] = useState(false);
    const utteranceRef = useRef(null);

    useEffect(() => {
        setIsSupported('speechSynthesis' in window);
    }, []);

    const speak = useCallback((text) => {
        if (!isSupported) return;

        // Cancel any ongoing speech
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utteranceRef.current = utterance;

        // Try to find a Tamil voice, fallback to default
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
