// Function: sub_3411F20
// Address: 0x3411f20
//
unsigned __int8 *__fastcall sub_3411F20(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned int *a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  __m128i v8; // xmm1
  __int128 v10; // [rsp-10h] [rbp-30h]
  _OWORD v11[2]; // [rsp+0h] [rbp-20h] BYREF

  v8 = _mm_loadu_si128((const __m128i *)&a8);
  *((_QWORD *)&v10 + 1) = 2;
  *(_QWORD *)&v10 = v11;
  v11[0] = _mm_loadu_si128((const __m128i *)&a7);
  v11[1] = v8;
  return sub_3411630(a1, a2, a3, a4, a5, a6, v10);
}
