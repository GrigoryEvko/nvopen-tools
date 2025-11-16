// Function: sub_1D373B0
// Address: 0x1d373b0
//
__int64 *__fastcall sub_1D373B0(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        unsigned __int8 *a4,
        __int64 a5,
        double a6,
        double a7,
        __m128i a8,
        __int64 a9,
        __int128 a10)
{
  const void ***v11; // rax
  __int128 si128; // xmm0
  int v13; // edx
  __int64 v14; // r9
  __m128i v16[3]; // [rsp+0h] [rbp-30h] BYREF

  v16[0] = _mm_loadu_si128((const __m128i *)&a10);
  v11 = (const void ***)sub_1D25C30((__int64)a1, a4, a5);
  si128 = (__int128)_mm_load_si128(v16);
  return sub_1D36D80(a1, a2, a3, v11, v13, *(double *)&si128, a7, a8, v14, si128);
}
