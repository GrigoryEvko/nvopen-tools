// Function: sub_340F900
// Address: 0x340f900
//
__int64 __fastcall sub_340F900(
        _QWORD *a1,
        unsigned int a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9)
{
  __int64 v9; // rax
  __int64 v10; // r9
  __m128i v11; // xmm0
  __int128 v12; // xmm1
  __int128 v13; // xmm2

  v9 = a1[128];
  v10 = 0;
  v11 = _mm_loadu_si128((const __m128i *)&a7);
  v12 = (__int128)_mm_loadu_si128((const __m128i *)&a8);
  v13 = (__int128)_mm_loadu_si128((const __m128i *)&a9);
  if ( v9 )
    v10 = *(unsigned int *)(v9 + 8);
  return sub_340EC60(a1, a2, a3, a4, a5, v10, v11.m128i_i64[0], v11.m128i_i64[1], v12, v13);
}
