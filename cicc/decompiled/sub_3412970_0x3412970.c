// Function: sub_3412970
// Address: 0x3412970
//
unsigned __int8 *__fastcall sub_3412970(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned int *a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9)
{
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __int128 v12; // [rsp-10h] [rbp-40h]
  _OWORD v13[3]; // [rsp+0h] [rbp-30h] BYREF

  v9 = _mm_loadu_si128((const __m128i *)&a8);
  *((_QWORD *)&v12 + 1) = 3;
  *(_QWORD *)&v12 = v13;
  v10 = _mm_loadu_si128((const __m128i *)&a9);
  v13[0] = _mm_loadu_si128((const __m128i *)&a7);
  v13[1] = v9;
  v13[2] = v10;
  return sub_3411630(a1, a2, a3, a4, a5, a6, v12);
}
