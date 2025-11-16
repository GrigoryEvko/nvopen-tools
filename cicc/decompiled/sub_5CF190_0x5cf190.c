// Function: sub_5CF190
// Address: 0x5cf190
//
__int64 __fastcall sub_5CF190(const __m128i *a1)
{
  const __m128i *v1; // rbx
  __m128i **v2; // r12
  __m128i *v3; // rax
  __int64 v5; // [rsp+8h] [rbp-18h] BYREF

  v5 = 0;
  if ( !a1 )
    return 0;
  v1 = a1;
  v2 = (__m128i **)&v5;
  do
  {
    v3 = (__m128i *)sub_727670();
    *v2 = v3;
    *v3 = _mm_loadu_si128(v1);
    v3[1] = _mm_loadu_si128(v1 + 1);
    v3[2] = _mm_loadu_si128(v1 + 2);
    v3[3] = _mm_loadu_si128(v1 + 3);
    v3[4] = _mm_loadu_si128(v1 + 4);
    (*v2)->m128i_i64[0] = 0;
    v1 = (const __m128i *)v1->m128i_i64[0];
    v2 = (__m128i **)*v2;
  }
  while ( v1 );
  return v5;
}
