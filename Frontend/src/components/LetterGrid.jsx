import { TAMIL_LETTERS } from '@/lib/tamilLetters';
import { cn } from '@/lib/utils';

export const LetterGrid = ({
    onLetterClick,
    selectedLetterId,
    disabledLetterIds = [],
    showHints = false,
    className
}) => {
    return (
        <div className={cn("grid grid-cols-3 sm:grid-cols-4 gap-3 sm:gap-4", className)}>
            {TAMIL_LETTERS.map((letter) => {
                const isSelected = selectedLetterId === letter.id;
                const isDisabled = disabledLetterIds.includes(letter.id);

                return (
                    <button
                        key={letter.id}
                        onClick={() => onLetterClick?.(letter)}
                        disabled={isDisabled}
                        className={cn(
                            "letter-card group",
                            isSelected && "selected",
                            isDisabled && "opacity-50 cursor-not-allowed",
                            !isDisabled && !isSelected && "hover:scale-105"
                        )}
                    >
                        <div className="font-tamil text-3xl sm:text-4xl font-bold text-primary mb-1 group-hover:scale-110 transition-transform">
                            {letter.letter}
                        </div>
                        <div className="text-sm font-medium text-foreground">
                            {letter.transliteration}
                        </div>
                        {showHints && (
                            <div className="text-xs text-muted-foreground mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                {letter.signHint}
                            </div>
                        )}
                    </button>
                );
            })}
        </div>
    );
};
