// Sign language gesture images for each Tamil letter
// These images show how to perform each sign

import signA from '@/assets/signs/sign-a.png';
import signAA from '@/assets/signs/sign-aa.png';
import signI from '@/assets/signs/sign-i.png';
import signEE from '@/assets/signs/sign-ee.png';
import signU from '@/assets/signs/sign-u.png';
import signOO from '@/assets/signs/sign-oo.png';
import signE from '@/assets/signs/sign-ae.png'; // Reusing ae for e
import signAE from '@/assets/signs/sign-ae.png';
import signAI from '@/assets/signs/sign-ai.png';
import signO from '@/assets/signs/sign-o.png';
import signOA from '@/assets/signs/sign-au.png'; // Reusing au for oa
import signAU from '@/assets/signs/sign-au.png';

export const SIGN_IMAGES = {
    1: signA,    // அ - a
    2: signAA,   // ஆ - aa
    3: signI,    // இ - i
    4: signEE,   // ஈ - ee
    5: signU,    // உ - u
    6: signOO,   // ஊ - oo
    7: signE,    // எ - e
    8: signAE,   // ஏ - ae
    9: signAI,   // ஐ - ai
    10: signO,   // ஒ - o
    11: signOA,  // ஓ - oa
    12: signAU,  // ஔ - au
};

export const getSignImage = (letterId) => {
    return SIGN_IMAGES[letterId];
};
