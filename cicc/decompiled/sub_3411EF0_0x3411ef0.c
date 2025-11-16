// Function: sub_3411EF0
// Address: 0x3411ef0
//
unsigned __int8 *__fastcall sub_3411EF0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned int *a4,
        __int64 a5,
        __int64 a6,
        __int128 a7)
{
  __int128 v8; // [rsp-10h] [rbp-20h]
  __m128i v9; // [rsp+0h] [rbp-10h] BYREF

  *((_QWORD *)&v8 + 1) = 1;
  *(_QWORD *)&v8 = &v9;
  v9 = _mm_loadu_si128((const __m128i *)&a7);
  return sub_3411630(a1, a2, a3, a4, a5, a6, v8);
}
