// Sign language gesture images for each Tamil letter
// These images show how to perform each sign

// IDs 1-8 are PNGs
import sign1 from '@/assets/signs/1.png';
import sign2 from '@/assets/signs/2.png';
import sign3 from '@/assets/signs/3.png';
import sign4 from '@/assets/signs/4.png';
import sign5 from '@/assets/signs/5.png';
import sign6 from '@/assets/signs/6.png';
import sign7 from '@/assets/signs/7.png';
import sign8 from '@/assets/signs/8.png';

// IDs 9-12 are JPEGs
import sign9 from '@/assets/signs/9.jpeg';
import sign10 from '@/assets/signs/10.jpeg';
import sign11 from '@/assets/signs/11.jpeg';
import sign12 from '@/assets/signs/12.jpeg';

export const SIGN_IMAGES = {
    1: sign1,    // அ - a
    2: sign2,    // ஆ - aa
    3: sign3,    // இ - i
    4: sign4,    // ஈ - ee
    5: sign5,    // உ - u
    6: sign6,    // ஊ - oo
    7: sign7,    // எ - e
    8: sign8,    // ஏ - ae
    9: sign9,    // ஐ - ai
    10: sign10,  // ஒ - o
    11: sign11,  // ஓ - oa
    12: sign12,  // ஔ - au
};

export const getSignImage = (letterId) => {
    return SIGN_IMAGES[letterId];
};
