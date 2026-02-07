import { useState, useCallback } from 'react';
import { Play, RotateCcw, Trophy, Clock } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { LetterGrid } from '@/components/LetterGrid';
import { Timer } from '@/components/Timer';
import { FeedbackDisplay } from '@/components/FeedbackDisplay';
import { useSpeech } from '@/hooks/useSpeech';
import { getRandomLetters } from '@/lib/tamilLetters';
import { cn } from '@/lib/utils';

const QUIZ_LENGTH = 5;
const TIME_PER_QUESTION = 15;

export const QuizMode = () => {
    const { speak } = useSpeech();

    const [quizLetters, setQuizLetters] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [results, setResults] = useState([]);
    const [isQuizActive, setIsQuizActive] = useState(false);
    const [isQuizComplete, setIsQuizComplete] = useState(false);
    const [currentFeedback, setCurrentFeedback] = useState(null);
    const [questionStartTime, setQuestionStartTime] = useState(0);
    const [timerKey, setTimerKey] = useState(0);

    const startQuiz = useCallback(() => {
        const letters = getRandomLetters(QUIZ_LENGTH);
        setQuizLetters(letters);
        setCurrentIndex(0);
        setResults([]);
        setIsQuizActive(true);
        setIsQuizComplete(false);
        setCurrentFeedback(null);
        setQuestionStartTime(Date.now());
        setTimerKey(prev => prev + 1);

        // Speak first letter
        speak(letters[0].letter);
    }, [speak]);

    const currentLetter = quizLetters[currentIndex];

    const moveToNext = useCallback(() => {
        if (currentIndex < quizLetters.length - 1) {
            const nextIndex = currentIndex + 1;
            setCurrentIndex(nextIndex);
            setCurrentFeedback(null);
            setQuestionStartTime(Date.now());
            setTimerKey(prev => prev + 1);
            speak(quizLetters[nextIndex].letter);
        } else {
            setIsQuizActive(false);
            setIsQuizComplete(true);
        }
    }, [currentIndex, quizLetters, speak]);

    const handleAnswer = useCallback((selectedLetter) => {
        if (!currentLetter || currentFeedback !== null) return;

        const isCorrect = selectedLetter.id === currentLetter.id;
        const timeSpent = (Date.now() - questionStartTime) / 1000;

        setCurrentFeedback(isCorrect);
        setResults(prev => [...prev, {
            letter: currentLetter,
            isCorrect,
            timeSpent
        }]);

        if (isCorrect) {
            speak(selectedLetter.letter);
        }

        // Auto-advance after delay
        setTimeout(moveToNext, 1500);
    }, [currentLetter, currentFeedback, questionStartTime, moveToNext, speak]);

    const handleTimeUp = useCallback(() => {
        if (currentFeedback !== null) return;

        const timeSpent = TIME_PER_QUESTION;
        setCurrentFeedback(false);
        setResults(prev => [...prev, {
            letter: currentLetter,
            isCorrect: false,
            timeSpent
        }]);

        setTimeout(moveToNext, 1500);
    }, [currentLetter, currentFeedback, moveToNext]);

    const correctCount = results.filter(r => r.isCorrect).length;
    const percentage = results.length > 0
        ? Math.round((correctCount / results.length) * 100)
        : 0;

    return (
        <div className="container mx-auto px-4 py-6">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">
                        Quiz Mode
                    </h2>
                    <p className="text-muted-foreground">
                        Test your Tamil sign language knowledge
                    </p>
                </div>

                {/* Quiz Start Screen */}
                {!isQuizActive && !isQuizComplete && (
                    <div className="elevated-card p-8 sm:p-12 text-center">
                        <div className="w-20 h-20 rounded-full gradient-primary mx-auto mb-6 flex items-center justify-center shadow-glow">
                            <Trophy className="w-10 h-10 text-primary-foreground" />
                        </div>
                        <h3 className="text-2xl font-bold text-foreground mb-3">
                            Ready for the Quiz?
                        </h3>
                        <p className="text-muted-foreground mb-2 max-w-md mx-auto">
                            You'll be shown {QUIZ_LENGTH} Tamil letters.
                            Identify the correct letter for each sign within {TIME_PER_QUESTION} seconds.
                        </p>
                        <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground mb-6">
                            <Clock className="w-4 h-4" />
                            <span>{TIME_PER_QUESTION} seconds per question</span>
                        </div>
                        <Button onClick={startQuiz} variant="hero" size="xl">
                            <Play className="w-6 h-6" />
                            Start Quiz
                        </Button>
                    </div>
                )}

                {/* Active Quiz */}
                {isQuizActive && currentLetter && (
                    <div className="space-y-6">
                        {/* Progress & Timer */}
                        <div className="grid sm:grid-cols-2 gap-4">
                            <div className="elevated-card p-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm text-muted-foreground">Question</span>
                                    <span className="text-2xl font-bold text-foreground">
                                        {currentIndex + 1}/{QUIZ_LENGTH}
                                    </span>
                                </div>
                                <div className="mt-2 h-2 bg-muted rounded-full overflow-hidden">
                                    <div
                                        className="h-full gradient-primary transition-all duration-300"
                                        style={{ width: `${((currentIndex + 1) / QUIZ_LENGTH) * 100}%` }}
                                    />
                                </div>
                            </div>

                            <div className="elevated-card p-4">
                                <Timer
                                    key={timerKey}
                                    duration={TIME_PER_QUESTION}
                                    isRunning={isQuizActive && currentFeedback === null}
                                    onTimeUp={handleTimeUp}
                                />
                            </div>
                        </div>

                        {/* Current Letter Display */}
                        <div className="elevated-card p-8 text-center">
                            <p className="text-sm text-muted-foreground mb-4">
                                Which letter is this?
                            </p>
                            <div className="tamil-display text-primary mb-4">
                                {currentLetter.letter}
                            </div>
                            <Button
                                onClick={() => speak(currentLetter.letter)}
                                variant="outline"
                                size="sm"
                            >
                                Hear Again
                            </Button>
                        </div>

                        {/* Feedback */}
                        {currentFeedback !== null && (
                            <FeedbackDisplay
                                isCorrect={currentFeedback}
                                message={currentFeedback
                                    ? "Correct! Moving to next..."
                                    : `The answer was "${currentLetter.transliteration}"`}
                            />
                        )}

                        {/* Answer Grid */}
                        <div>
                            <h3 className="text-lg font-semibold text-foreground mb-4 text-center">
                                Select your answer:
                            </h3>
                            <LetterGrid
                                onLetterClick={handleAnswer}
                                disabledLetterIds={currentFeedback !== null ? [] : []}
                            />
                        </div>
                    </div>
                )}

                {/* Quiz Results */}
                {isQuizComplete && (
                    <div className="space-y-6">
                        {/* Score Card */}
                        <div className="elevated-card p-8 text-center">
                            <div className={cn(
                                "w-24 h-24 rounded-full mx-auto mb-6 flex items-center justify-center",
                                percentage >= 80 ? "gradient-success" :
                                    percentage >= 50 ? "gradient-primary" :
                                        "bg-destructive"
                            )}>
                                <Trophy className="w-12 h-12 text-white" />
                            </div>

                            <h3 className="text-3xl font-bold text-foreground mb-2">
                                {percentage}%
                            </h3>
                            <p className="text-xl text-muted-foreground mb-4">
                                {correctCount} out of {results.length} correct
                            </p>

                            <p className={cn(
                                "text-lg font-medium",
                                percentage >= 80 ? "text-success" :
                                    percentage >= 50 ? "text-primary" :
                                        "text-destructive"
                            )}>
                                {percentage >= 80 ? "Excellent! 🎉" :
                                    percentage >= 50 ? "Good job! Keep practicing!" :
                                        "Keep learning! You'll get better!"}
                            </p>
                        </div>

                        {/* Results breakdown */}
                        <div className="elevated-card p-6">
                            <h4 className="font-semibold text-foreground mb-4">Results Breakdown</h4>
                            <div className="space-y-3">
                                {results.map((result, index) => (
                                    <div
                                        key={index}
                                        className={cn(
                                            "flex items-center justify-between p-3 rounded-xl",
                                            result.isCorrect ? "bg-success/10" : "bg-destructive/10"
                                        )}
                                    >
                                        <div className="flex items-center gap-3">
                                            <span className="font-tamil text-2xl font-bold text-primary">
                                                {result.letter.letter}
                                            </span>
                                            <span className="text-foreground">
                                                {result.letter.transliteration}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <span className="text-sm text-muted-foreground">
                                                {result.timeSpent.toFixed(1)}s
                                            </span>
                                            <span className={cn(
                                                "text-sm font-medium",
                                                result.isCorrect ? "text-success" : "text-destructive"
                                            )}>
                                                {result.isCorrect ? "✓ Correct" : "✗ Incorrect"}
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Restart button */}
                        <div className="flex justify-center">
                            <Button onClick={startQuiz} variant="hero" size="lg">
                                <RotateCcw className="w-5 h-5" />
                                Try Again
                            </Button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
