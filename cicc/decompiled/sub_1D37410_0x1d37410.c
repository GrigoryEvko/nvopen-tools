// Function: sub_1D37410
// Address: 0x1d37410
//
__int64 *__fastcall sub_1D37410(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        const void ***a4,
        int a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9,
        __int128 a10)
{
  __int128 v11; // [rsp-10h] [rbp-20h]
  __int128 v12; // [rsp+0h] [rbp-10h] BYREF

  *((_QWORD *)&v11 + 1) = 1;
  *(_QWORD *)&v11 = &v12;
  return sub_1D36D80(a1, a2, a3, a4, a5, *(double *)_mm_loadu_si128((const __m128i *)&a10).m128i_i64, a8, a9, a6, v11);
}
