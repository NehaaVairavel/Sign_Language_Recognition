// 12 Basic Tamil Letters for Sign Language Recognition

export const TAMIL_LETTERS = [
    {
        id: 1,
        letter: "அ",
        transliteration: "a",
        description: "First vowel - 'a' as in 'about'",
        signHint: "Open palm facing forward"
    },
    {
        id: 2,
        letter: "ஆ",
        transliteration: "aa",
        description: "Long vowel - 'aa' as in 'father'",
        signHint: "Open palm with fingers spread"
    },
    {
        id: 3,
        letter: "இ",
        transliteration: "i",
        description: "Short vowel - 'i' as in 'bit'",
        signHint: "Index finger pointing up"
    },
    {
        id: 4,
        letter: "ஈ",
        transliteration: "ee",
        description: "Long vowel - 'ee' as in 'see'",
        signHint: "Index and middle finger together"
    },
    {
        id: 5,
        letter: "உ",
        transliteration: "u",
        description: "Short vowel - 'u' as in 'put'",
        signHint: "Closed fist with thumb up"
    },
    {
        id: 6,
        letter: "ஊ",
        transliteration: "oo",
        description: "Long vowel - 'oo' as in 'moon'",
        signHint: "Closed fist with thumb extended"
    },
    {
        id: 7,
        letter: "எ",
        transliteration: "e",
        description: "Short vowel - 'e' as in 'bed'",
        signHint: "Three fingers extended"
    },
    {
        id: 8,
        letter: "ஏ",
        transliteration: "ae",
        description: "Long vowel - 'ae' as in 'bay'",
        signHint: "Three fingers with thumb out"
    },
    {
        id: 9,
        letter: "ஐ",
        transliteration: "ai",
        description: "Diphthong - 'ai' as in 'kite'",
        signHint: "Peace sign with palm facing out"
    },
    {
        id: 10,
        letter: "ஒ",
        transliteration: "o",
        description: "Short vowel - 'o' as in 'hot'",
        signHint: "Circle with fingers"
    },
    {
        id: 11,
        letter: "ஓ",
        transliteration: "oa",
        description: "Long vowel - 'oa' as in 'boat'",
        signHint: "Circle with extended thumb"
    },
    {
        id: 12,
        letter: "ஔ",
        transliteration: "au",
        description: "Diphthong - 'au' as in 'cow'",
        signHint: "Cupped hand shape"
    }
];

export const getLetterById = (id) => {
    return TAMIL_LETTERS.find(letter => letter.id === id);
};

export const getRandomLetter = () => {
    const randomIndex = Math.floor(Math.random() * TAMIL_LETTERS.length);
    return TAMIL_LETTERS[randomIndex];
};

export const getRandomLetters = (count) => {
    const shuffled = [...TAMIL_LETTERS].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, Math.min(count, TAMIL_LETTERS.length));
};
