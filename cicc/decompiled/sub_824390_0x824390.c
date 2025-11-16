// Function: sub_824390
// Address: 0x824390
//
__int64 __fastcall sub_824390(__m128i *a1)
{
  __int64 v1; // rax
  const __m128i *v2; // rbx
  const char *v3; // r14
  const char *v4; // r13
  __int64 v5; // r12
  int v6; // r15d

  v1 = a1[2].m128i_i64[0];
  v2 = (const __m128i *)qword_4F07320[0];
  if ( !qword_4F07320[0] )
    return 0;
  v3 = *(const char **)(v1 + 8);
  v4 = *(const char **)(v1 + 24);
  while ( 1 )
  {
    v5 = v2[2].m128i_i64[0];
    v6 = strcmp(*(const char **)(v5 + 8), v3);
    if ( v4 )
    {
      if ( !strcmp(*(const char **)(v5 + 24), v4) )
        break;
    }
    if ( !v6 )
      break;
    v2 = (const __m128i *)v2->m128i_i64[0];
    if ( !v2 )
      return 0;
  }
  sub_684AE0(0xC01u, (__m128i *)a1[1].m128i_i32, (__int64)v3);
  *a1 = _mm_loadu_si128(v2);
  a1[1] = _mm_loadu_si128(v2 + 1);
  a1[2] = _mm_loadu_si128(v2 + 2);
  return 1;
}
