// Function: sub_6E15C0
// Address: 0x6e15c0
//
__int64 __fastcall sub_6E15C0(const __m128i *a1)
{
  const __m128i *v1; // rbx
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // r12
  __m128i v5; // xmm5
  __int64 v6; // rax
  __int64 v7; // rdx
  __m128i v8; // xmm2

  if ( !a1 )
    return 0;
  v1 = a1;
  v2 = sub_6DED10(a1[1].m128i_i64[0], a1[2].m128i_i64);
  v3 = *(_QWORD *)(v2 + 40);
  *(__m128i *)v2 = _mm_loadu_si128(a1);
  v4 = v2;
  *(__m128i *)(v2 + 16) = _mm_loadu_si128(a1 + 1);
  v5 = _mm_loadu_si128(a1 + 2);
  *(_QWORD *)(v2 + 48) = 0;
  *(__m128i *)(v2 + 32) = v5;
  *(_QWORD *)(v2 + 40) = v3;
  while ( 1 )
  {
    v1 = (const __m128i *)v1[3].m128i_i64[0];
    if ( !v1 )
      break;
    v6 = sub_6DED10(v1[1].m128i_i64[0], v1[2].m128i_i64);
    v7 = *(_QWORD *)(v6 + 40);
    *(__m128i *)v6 = _mm_loadu_si128(v1);
    *(__m128i *)(v6 + 16) = _mm_loadu_si128(v1 + 1);
    v8 = _mm_loadu_si128(v1 + 2);
    *(_QWORD *)(v6 + 48) = 0;
    *(__m128i *)(v6 + 32) = v8;
    *(_QWORD *)(v6 + 40) = v7;
    *(_QWORD *)(v4 + 48) = v6;
    v4 = v6;
  }
  return v2;
}
