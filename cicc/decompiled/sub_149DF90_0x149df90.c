// Function: sub_149DF90
// Address: 0x149df90
//
__int64 __fastcall sub_149DF90(
        const __m128i *a1,
        __m128i *a2,
        unsigned __int64 a3,
        __int64 (__fastcall *a4)(__m128i *),
        __int64 a5,
        __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // r14
  __int64 i; // rbx
  __m128i *j; // rbx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int128 v15; // xmm2
  __int128 v16; // xmm3

  result = (char *)a2 - (char *)a1;
  v8 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)a1) >> 3);
  if ( (char *)a2 - (char *)a1 > 40 )
  {
    for ( i = (v8 - 2) / 2; ; --i )
    {
      result = sub_149DD50(
                 (__int64)a1,
                 i,
                 v8,
                 a4,
                 a5,
                 a6,
                 *(_OWORD *)&_mm_loadu_si128((const __m128i *)((char *)a1 + 40 * i)),
                 *(_OWORD *)&_mm_loadu_si128((const __m128i *)((char *)a1 + 40 * i + 16)),
                 a1[2].m128i_i64[5 * i]);
      if ( !i )
        break;
    }
  }
  for ( j = a2; a3 > (unsigned __int64)j; result = sub_149DD50((__int64)a1, 0, v8, a4, v12, v13, v15, v16, v14) )
  {
    while ( 1 )
    {
      result = ((__int64 (__fastcall *)(__m128i *, const __m128i *))a4)(j, a1);
      if ( (_BYTE)result )
        break;
      j = (__m128i *)((char *)j + 40);
      if ( a3 <= (unsigned __int64)j )
        return result;
    }
    v14 = j[2].m128i_i64[0];
    v15 = (__int128)_mm_loadu_si128(j);
    v16 = (__int128)_mm_loadu_si128(j + 1);
    *j = _mm_loadu_si128(a1);
    j = (__m128i *)((char *)j + 40);
    *(__m128i *)((char *)j - 24) = _mm_loadu_si128(a1 + 1);
    j[-1].m128i_i32[2] = a1[2].m128i_i32[0];
  }
  return result;
}
