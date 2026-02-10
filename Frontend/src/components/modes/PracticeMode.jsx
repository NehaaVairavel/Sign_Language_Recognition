import { useState, useCallback } from 'react';
import { RefreshCw, Play } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { CameraFeed } from '@/components/CameraFeed';
import { TamilLetterDisplay } from '@/components/TamilLetterDisplay';
import { LetterGrid } from '@/components/LetterGrid';
import { FeedbackDisplay } from '@/components/FeedbackDisplay';
import { ScoreDisplay } from '@/components/ScoreDisplay';
import { useWebcam } from '@/hooks/useWebcam';
import { useSignRecognition } from '@/hooks/useSignRecognition';
import { useSpeech } from '@/hooks/useSpeech';
import { getRandomLetter } from '@/lib/tamilLetters';

export const PracticeMode = () => {
    const { videoRef, isStreaming, error, startCamera, stopCamera } = useWebcam();
    const { simulateDetection } = useSignRecognition(videoRef);
    const { speak } = useSpeech();

    const [targetLetter, setTargetLetter] = useState(null);
    const [isCorrect, setIsCorrect] = useState(null);
    const [score, setScore] = useState({ correct: 0, total: 0 });
    const [isPracticing, setIsPracticing] = useState(false);

    const startPractice = useCallback(() => {
        const letter = getRandomLetter();
        setTargetLetter(letter);
        setIsCorrect(null);
        setIsPracticing(true);
        speak(letter.letter);
    }, [speak]);

    const handleLetterClick = useCallback((letter) => {
        if (!targetLetter || !isPracticing) return;

        const correct = letter.id === targetLetter.id;
        setIsCorrect(correct);
        setScore(prev => ({
            correct: prev.correct + (correct ? 1 : 0),
            total: prev.total + 1
        }));

        if (correct) {
            speak(letter.letter);
        }
    }, [targetLetter, isPracticing, speak]);

    const handleTryAgain = () => {
        setIsCorrect(null);
        if (targetLetter) {
            speak(targetLetter.letter);
        }
    };

    const handleNextLetter = () => {
        startPractice();
    };

    const resetScore = () => {
        setScore({ correct: 0, total: 0 });
        setTargetLetter(null);
        setIsCorrect(null);
        setIsPracticing(false);
    };

    const getCameraStatus = () => {
        if (!isStreaming) return 'idle';
        if (isCorrect === true) return 'success';
        if (isCorrect === false) return 'error';
        return 'detecting';
    };

    return (
        <div className="container mx-auto px-4 py-6">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">
                        Practice Mode
                    </h2>
                    <p className="text-muted-foreground">
                        Practice signing the displayed Tamil letters
                    </p>
                </div>

                {/* Score Display */}
                {score.total > 0 && (
                    <div className="mb-6">
                        <ScoreDisplay
                            correct={score.correct}
                            total={score.total}
                        />
                    </div>
                )}

                <div className="grid lg:grid-cols-2 gap-6 mb-6">
                    {/* Target Letter */}
                    <div className="space-y-4">
                        {targetLetter ? (
                            <>
                                <div className="text-center mb-2">
                                    <span className="text-sm font-medium text-muted-foreground">
                                        Show the sign for:
                                    </span>
                                </div>
                                <TamilLetterDisplay
                                    letter={targetLetter}
                                    onSpeak={() => speak(targetLetter.letter)}
                                    showDetails={false}
                                    size="lg"
                                />
                            </>
                        ) : (
                            <div className="elevated-card p-8 flex flex-col items-center justify-center min-h-[300px]">
                                <div className="text-center mb-6">
                                    <h3 className="text-xl font-semibold text-foreground mb-2">
                                        Ready to Practice?
                                    </h3>
                                    <p className="text-muted-foreground">
                                        Start practicing to improve your sign language skills
                                    </p>
                                </div>
                                <Button onClick={startPractice} variant="hero" size="xl">
                                    <Play className="w-6 h-6" />
                                    Start Practice
                                </Button>
                            </div>
                        )}
                    </div>

                    {/* Feedback Section */}
                    <div className="space-y-4">
                        <CameraFeed
                            ref={videoRef}
                            isStreaming={isStreaming}
                            error={error}
                            onStart={startCamera}
                            onStop={stopCamera}
                            status={getCameraStatus()}
                        />

                        {isCorrect !== null && (
                            <FeedbackDisplay
                                isCorrect={isCorrect}
                                message={isCorrect
                                    ? "Great job! You got it right!"
                                    : `The correct sign is for "${targetLetter?.letter}"`}
                                onTryAgain={!isCorrect ? handleTryAgain : undefined}
                            />
                        )}
                    </div>
                </div>

                {/* Action Buttons */}
                {isPracticing && (
                    <div className="flex justify-center gap-4 mb-6">
                        {isCorrect === true && (
                            <Button onClick={handleNextLetter} variant="success" size="lg">
                                Next Letter
                            </Button>
                        )}
                        <Button onClick={resetScore} variant="outline" size="lg">
                            <RefreshCw className="w-4 h-4" />
                            Reset Score
                        </Button>
                    </div>
                )}

                {/* Letter Grid for Demo */}
                {isPracticing && (
                    <div>
                        <h3 className="text-lg font-semibold text-foreground mb-4 text-center">
                            Select your answer:
                        </h3>
                        <LetterGrid
                            onLetterClick={handleLetterClick}
                            selectedLetterId={isCorrect ? targetLetter?.id : undefined}
                        />
                    </div>
                )}
            </div>
        </div>
    );
};
