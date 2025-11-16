// Function: sub_89EE30
// Address: 0x89ee30
//
__int64 __fastcall sub_89EE30(const __m128i *a1)
{
  const __m128i *v1; // rbx
  __m128i **v2; // r12
  __int64 v3; // r13
  __m128i *v4; // rax
  __int64 v6; // [rsp+8h] [rbp-28h] BYREF

  v6 = 0;
  if ( !a1 )
    return 0;
  v1 = a1;
  v2 = (__m128i **)&v6;
  do
  {
    v3 = v1->m128i_i64[1];
    v4 = (__m128i *)sub_880AD0(v3);
    *v2 = v4;
    *v4 = _mm_loadu_si128(v1);
    v4[1] = _mm_loadu_si128(v1 + 1);
    v4[2] = _mm_loadu_si128(v1 + 2);
    v4[3] = _mm_loadu_si128(v1 + 3);
    v4[4] = _mm_loadu_si128(v1 + 4);
    v4[5] = _mm_loadu_si128(v1 + 5);
    v4[6] = _mm_loadu_si128(v1 + 6);
    v4[7] = _mm_loadu_si128(v1 + 7);
    v4[8].m128i_i64[0] = v1[8].m128i_i64[0];
    (*v2)->m128i_i64[0] = 0;
    (*v2)->m128i_i64[1] = v3;
    v1 = (const __m128i *)v1->m128i_i64[0];
    v2 = (__m128i **)*v2;
  }
  while ( v1 );
  return v6;
}
