// Function: sub_5CF790
// Address: 0x5cf790
//
const __m128i *__fastcall sub_5CF790(__int64 a1)
{
  const __m128i *v1; // r12
  __int64 v2; // r13
  __m128i **v3; // rbx
  __m128i *v5; // rax
  __int64 v6; // [rsp+8h] [rbp-28h] BYREF

  v1 = *(const __m128i **)(a1 + 64);
  v6 = 0;
  if ( v1 )
  {
    v2 = 0x400000000008101LL;
    v3 = (__m128i **)&v6;
    do
    {
      if ( (unsigned __int8)(v1->m128i_i8[8] - 6) <= 0x3Au && _bittest64(&v2, (unsigned int)v1->m128i_u8[8] - 6) )
      {
        v5 = (__m128i *)sub_727670();
        *v3 = v5;
        *v5 = _mm_loadu_si128(v1);
        v5[1] = _mm_loadu_si128(v1 + 1);
        v5[2] = _mm_loadu_si128(v1 + 2);
        v5[3] = _mm_loadu_si128(v1 + 3);
        v5[4] = _mm_loadu_si128(v1 + 4);
        (*v3)->m128i_i64[0] = 0;
        (*v3)->m128i_i64[0] = 0;
        (*v3)[3].m128i_i64[0] = 0;
        (*v3)->m128i_i8[10] = 0;
        v3 = (__m128i **)*v3;
      }
      v1 = (const __m128i *)v1->m128i_i64[0];
    }
    while ( v1 );
    return (const __m128i *)v6;
  }
  return v1;
}
