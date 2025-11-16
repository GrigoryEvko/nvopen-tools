// Function: sub_34129B0
// Address: 0x34129b0
//
unsigned __int8 *__fastcall sub_34129B0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned int *a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int128 a10)
{
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  __int128 v14; // [rsp-10h] [rbp-50h]
  _OWORD v15[4]; // [rsp+0h] [rbp-40h] BYREF

  v10 = _mm_loadu_si128((const __m128i *)&a8);
  *((_QWORD *)&v14 + 1) = 4;
  v11 = _mm_loadu_si128((const __m128i *)&a9);
  v12 = _mm_loadu_si128((const __m128i *)&a10);
  *(_QWORD *)&v14 = v15;
  v15[0] = _mm_loadu_si128((const __m128i *)&a7);
  v15[1] = v10;
  v15[2] = v11;
  v15[3] = v12;
  return sub_3411630(a1, a2, a3, a4, a5, a6, v14);
}
