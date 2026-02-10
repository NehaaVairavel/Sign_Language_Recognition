import { useState, useCallback, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Volume2, Camera, Check, X, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { CameraFeed } from '@/components/CameraFeed';
import { SignAvatar } from '@/components/SignAvatar';
import { FeedbackDisplay } from '@/components/FeedbackDisplay';
import { TAMIL_LETTERS } from '@/lib/tamilLetters';
import { useSpeech } from '@/hooks/useSpeech';
import { useWebcam } from '@/hooks/useWebcam';
import { useSignRecognition } from '@/hooks/useSignRecognition';
import { cn } from '@/lib/utils';

export const LearnMode = () => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [learningStep, setLearningStep] = useState('observe');
    const [isCorrect, setIsCorrect] = useState(null);

    const { speak, isSpeaking } = useSpeech();
    const { videoRef, isStreaming, error, startCamera, stopCamera } = useWebcam();
    const {
        detectedLetter,
        startDetection,
        stopDetection,
        isHandPresent
    } = useSignRecognition(videoRef);

    const currentLetter = TAMIL_LETTERS[currentIndex];

    // Manage detection state based on mode
    useEffect(() => {
        if (isStreaming && learningStep === 'practice') {
            startDetection();
        } else {
            stopDetection();
        }
    }, [isStreaming, learningStep, startDetection, stopDetection]);

    // Auto-verify when a letter is detected
    useEffect(() => {
        if (learningStep === 'practice' && detectedLetter) {
            if (detectedLetter.id === currentLetter.id) {
                setIsCorrect(true);
                setLearningStep('feedback');
                speak("Correct! " + currentLetter.letter);
                speak(currentLetter.letter);
            }
        }
    }, [detectedLetter, learningStep, currentLetter, speak]);

    const goToNext = () => {
        const nextIndex = (currentIndex + 1) % TAMIL_LETTERS.length;
        setCurrentIndex(nextIndex);
        resetLearningState();
        speak(TAMIL_LETTERS[nextIndex].letter);
    };

    const goToPrevious = () => {
        const prevIndex = (currentIndex - 1 + TAMIL_LETTERS.length) % TAMIL_LETTERS.length;
        setCurrentIndex(prevIndex);
        resetLearningState();
        speak(TAMIL_LETTERS[prevIndex].letter);
    };

    const resetLearningState = () => {
        setLearningStep('observe');
        setIsCorrect(null);
    };

    const handleSpeak = () => {
        speak(currentLetter.letter);
    };

    const handleLetterSelect = (index) => {
        setCurrentIndex(index);
        resetLearningState();
        speak(TAMIL_LETTERS[index].letter);
    };

    const handleStartPractice = () => {
        setLearningStep('practice');
        if (!isStreaming) {
            startCamera();
        }
    };

    // Manual verify (kept for fallback/testing if needed, but UI can be simplified)
    const handleVerifySign = useCallback((selectedLetter) => {
        const correct = selectedLetter.id === currentLetter.id;
        setIsCorrect(correct);
        setLearningStep('feedback');

        if (correct) {
            speak(currentLetter.letter);
        }
    }, [currentLetter, speak]);

    const handleTryAgain = () => {
        setLearningStep('practice');
        setIsCorrect(null);
    };

    const getCameraStatus = () => {
        if (!isStreaming) return 'idle';
        if (isCorrect === true) return 'success';
        if (isCorrect === false) return 'error';
        if (learningStep === 'practice') return 'detecting'; // Shows "Detecting..." overlay
        return 'idle';
    };

    return (
        <div className="container mx-auto px-4 py-6">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">
                        Learn Tamil Signs
                    </h2>
                    <p className="text-muted-foreground">
                        {learningStep === 'observe' && "Observe the letter and sign gesture, then practice"}
                        {learningStep === 'practice' && "Now perform the sign with your hands"}
                        {learningStep === 'feedback' && (isCorrect ? "Great job! Move to the next letter" : "Let's try again")}
                    </p>
                </div>

                {/* Progress indicator */}
                <div className="flex items-center justify-center gap-2 mb-6">
                    <span className="text-sm text-muted-foreground">
                        Letter {currentIndex + 1} of {TAMIL_LETTERS.length}
                    </span>
                    <div className="h-2 w-32 bg-muted rounded-full overflow-hidden">
                        <div
                            className="h-full gradient-primary transition-all duration-300"
                            style={{ width: `${((currentIndex + 1) / TAMIL_LETTERS.length) * 100}%` }}
                        />
                    </div>
                </div>

                {/* Main Learning Area */}
                <div className="grid lg:grid-cols-2 gap-6 mb-6">
                    {/* Left: Tamil Letter & Sign Avatar */}
                    <div className="space-y-4">
                        {/* Tamil Letter Display */}
                        <div className="elevated-card p-6 text-center">
                            <div className="tamil-display text-primary mb-2 animate-bounce-in" key={currentLetter.id}>
                                {currentLetter.letter}
                            </div>
                            <p className="text-xl font-semibold text-foreground mb-1">
                                "{currentLetter.transliteration}"
                            </p>
                            <p className="text-sm text-muted-foreground mb-4">
                                {currentLetter.description}
                            </p>
                            <Button
                                onClick={handleSpeak}
                                variant={isSpeaking ? "secondary" : "outline"}
                                size="sm"
                            >
                                <Volume2 className="w-4 h-4" />
                                {isSpeaking ? 'Speaking...' : 'Hear Pronunciation'}
                            </Button>
                        </div>

                        {/* Sign Avatar - Always visible */}
                        <SignAvatar letter={currentLetter} size="lg" />
                    </div>

                    {/* Right: Practice Area */}
                    <div className="space-y-4">
                        {learningStep === 'observe' && (
                            <div className="elevated-card p-8 flex flex-col items-center justify-center min-h-[350px]">
                                <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mb-6">
                                    <Camera className="w-10 h-10 text-primary" />
                                </div>
                                <h3 className="text-xl font-semibold text-foreground mb-2 text-center">
                                    Ready to Practice?
                                </h3>
                                <p className="text-muted-foreground text-center mb-6 max-w-sm">
                                    Study the sign gesture shown on the left, then practice performing it with your camera.
                                </p>
                                <Button onClick={handleStartPractice} variant="hero" size="lg">
                                    <Camera className="w-5 h-5" />
                                    Start Practice
                                </Button>
                            </div>
                        )}

                        {(learningStep === 'practice' || learningStep === 'feedback') && (
                            <>
                                <CameraFeed
                                    ref={videoRef}
                                    isStreaming={isStreaming}
                                    error={error}
                                    onStart={startCamera}
                                    onStop={stopCamera}
                                    status={getCameraStatus()}
                                />

                                {learningStep === 'feedback' && (
                                    <FeedbackDisplay
                                        isCorrect={isCorrect}
                                        message={isCorrect
                                            ? `Excellent! You correctly signed "${currentLetter.transliteration}"!`
                                            : `That wasn't quite right. Look at the avatar and try again.`}
                                        onTryAgain={!isCorrect ? handleTryAgain : undefined}
                                    />
                                )}

                                {learningStep === 'practice' && (
                                    <div className="glass-card p-4">
                                        <p className="text-sm text-center text-muted-foreground mb-3">
                                            <strong>Practice Mode:</strong> Show the sign to the camera.
                                            {detectedLetter && (
                                                <span className="block mt-1 font-semibold text-primary">
                                                    Detected: {detectedLetter.letter} ({detectedLetter.transliteration})
                                                </span>
                                            )}
                                        </p>

                                        {/* Optional: Keep manual buttons for debugging/fallback if detection isn't working */}
                                        <div className="text-xs text-center text-muted-foreground mt-4">
                                            Having trouble?
                                            <button
                                                onClick={() => handleVerifySign(currentLetter)}
                                                className="ml-1 underline hover:text-primary"
                                            >
                                                Skip/Manual Verify
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>

                {/* Navigation & Action Buttons */}
                <div className="flex items-center justify-center gap-4 mb-8">
                    <Button
                        onClick={goToPrevious}
                        variant="outline"
                        size="lg"
                    >
                        <ChevronLeft className="w-5 h-5" />
                        Previous
                    </Button>

                    {learningStep === 'feedback' && isCorrect && (
                        <Button onClick={goToNext} variant="success" size="lg">
                            Next Letter
                            <ChevronRight className="w-5 h-5" />
                        </Button>
                    )}

                    {learningStep !== 'observe' && (
                        <Button onClick={resetLearningState} variant="outline" size="lg">
                            <RotateCcw className="w-4 h-4" />
                            Start Over
                        </Button>
                    )}

                    {learningStep === 'observe' && (
                        <Button
                            onClick={goToNext}
                            variant="default"
                            size="lg"
                        >
                            Skip
                            <ChevronRight className="w-5 h-5" />
                        </Button>
                    )}
                </div>

                {/* Letter selector */}
                <div className="grid grid-cols-4 sm:grid-cols-6 gap-2">
                    {TAMIL_LETTERS.map((letter, index) => (
                        <button
                            key={letter.id}
                            onClick={() => handleLetterSelect(index)}
                            className={cn(
                                "p-3 rounded-xl border-2 transition-all duration-200",
                                "hover:border-primary hover:bg-primary/5",
                                index === currentIndex
                                    ? "border-primary bg-primary/10"
                                    : "border-border bg-card"
                            )}
                        >
                            <div className="font-tamil text-2xl font-bold text-primary">
                                {letter.letter}
                            </div>
                            <div className="text-xs text-muted-foreground">
                                {letter.transliteration}
                            </div>
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
};
