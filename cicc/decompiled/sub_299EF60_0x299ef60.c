// Function: sub_299EF60
// Address: 0x299ef60
//
void __fastcall sub_299EF60(
        const __m128i *a1,
        const __m128i *a2,
        const __m128i *a3,
        const __m128i *a4,
        __m128i *a5,
        unsigned __int8 (__fastcall *a6)(const __m128i *, const __m128i *))
{
  const __m128i *v9; // r15
  const __m128i *v10; // r12
  __int8 *v11; // rdx

  if ( a1 == a2 )
  {
    v11 = (__int8 *)a4;
    if ( a3 == a4 )
      return;
LABEL_12:
    memmove((char *)a5 - (v11 - (__int8 *)a3), a3, v11 - (__int8 *)a3);
    return;
  }
  if ( a3 != a4 )
  {
    v9 = (const __m128i *)((char *)a2 - 56);
    v10 = (const __m128i *)((char *)a4 - 56);
    while ( 1 )
    {
      while ( 1 )
      {
        a5 = (__m128i *)((char *)a5 - 56);
        if ( a6(v10, v9) )
          break;
        *a5 = _mm_loadu_si128(v10);
        a5[1] = _mm_loadu_si128(v10 + 1);
        a5[2] = _mm_loadu_si128(v10 + 2);
        a5[3].m128i_i64[0] = v10[3].m128i_i64[0];
        if ( v10 == a3 )
          return;
        v10 = (const __m128i *)((char *)v10 - 56);
      }
      *a5 = _mm_loadu_si128(v9);
      a5[1] = _mm_loadu_si128(v9 + 1);
      a5[2] = _mm_loadu_si128(v9 + 2);
      a5[3].m128i_i64[0] = v9[3].m128i_i64[0];
      if ( v9 == a1 )
        break;
      v9 = (const __m128i *)((char *)v9 - 56);
    }
    v11 = &v10[3].m128i_i8[8];
    if ( a3 != (const __m128i *)&v10[3].m128i_u64[1] )
      goto LABEL_12;
  }
}
