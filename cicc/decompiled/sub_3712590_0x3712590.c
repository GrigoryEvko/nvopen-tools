// Function: sub_3712590
// Address: 0x3712590
//
__int64 __fastcall sub_3712590(
        const __m128i *a1,
        __m128i *a2,
        unsigned __int64 a3,
        unsigned __int8 (__fastcall *a4)(__m128i *, unsigned __int64 *),
        __int64 a5,
        __int64 a6)
{
  __int64 result; // rax
  __m128i *v8; // rbx
  __int64 v9; // r15
  __int64 i; // r14
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __int128 v14; // xmm2
  __int128 v15; // xmm3

  result = (char *)a2 - (char *)a1;
  v8 = a2;
  if ( (char *)a2 - (char *)a1 > 40 )
  {
    v9 = 0xCCCCCCCCCCCCCCCDLL * (result >> 3);
    for ( i = (v9 - 2) / 2; ; --i )
    {
      result = sub_3712350(
                 (__int64)a1,
                 i,
                 v9,
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
  if ( (unsigned __int64)a2 < a3 )
  {
    do
    {
      while ( 1 )
      {
        result = ((__int64 (__fastcall *)(__m128i *, const __m128i *))a4)(v8, a1);
        if ( (_BYTE)result )
          break;
        v8 = (__m128i *)((char *)v8 + 40);
        if ( a3 <= (unsigned __int64)v8 )
          return result;
      }
      v13 = v8[2].m128i_i64[0];
      v14 = (__int128)_mm_loadu_si128(v8);
      v15 = (__int128)_mm_loadu_si128(v8 + 1);
      *v8 = _mm_loadu_si128(a1);
      v8 = (__m128i *)((char *)v8 + 40);
      *(__m128i *)((char *)v8 - 24) = _mm_loadu_si128(a1 + 1);
      v8[-1].m128i_i16[4] = a1[2].m128i_i16[0];
      result = sub_3712350(
                 (__int64)a1,
                 0,
                 0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)a1) >> 3),
                 a4,
                 v11,
                 v12,
                 v14,
                 v15,
                 v13);
    }
    while ( a3 > (unsigned __int64)v8 );
  }
  return result;
}
