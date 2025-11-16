// Function: sub_161CBB0
// Address: 0x161cbb0
//
__int64 __fastcall sub_161CBB0(__m128i *a1)
{
  unsigned __int64 v1; // rcx
  __int64 v2; // rsi
  __m128i *v3; // rax
  __int64 v4; // rdx
  __m128i v6; // [rsp+0h] [rbp-20h]

  v1 = a1[1].m128i_u64[0];
  v2 = a1->m128i_i64[0];
  v6 = _mm_loadu_si128(a1);
  if ( v1 < a1[-1].m128i_i64[1] )
  {
    v3 = (__m128i *)((char *)a1 - 24);
    do
    {
      v4 = v3->m128i_i64[0];
      a1 = v3;
      v3 = (__m128i *)((char *)v3 - 24);
      v3[3].m128i_i64[0] = v4;
      v3[3].m128i_i64[1] = v3[2].m128i_i64[0];
      v3[4].m128i_i64[0] = v3[2].m128i_i64[1];
    }
    while ( v1 < v3[1].m128i_i64[0] );
  }
  a1->m128i_i64[0] = v2;
  a1[1].m128i_i64[0] = v1;
  a1->m128i_i64[1] = v6.m128i_i64[1];
  return v6.m128i_i64[1];
}
