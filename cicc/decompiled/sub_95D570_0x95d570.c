// Function: sub_95D570
// Address: 0x95d570
//
__m128i *__fastcall sub_95D570(__m128i *a1, const char *a2, __int64 a3)
{
  size_t v4; // rax
  __m128i *v5; // rax
  __int64 v6; // rcx

  v4 = strlen(a2);
  v5 = (__m128i *)sub_2241130(a3, 0, 0, a2, v4);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
  {
    a1[1] = _mm_loadu_si128(v5 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v5->m128i_i64[0];
    a1[1].m128i_i64[0] = v5[1].m128i_i64[0];
  }
  v6 = v5->m128i_i64[1];
  v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
  v5->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v6;
  v5[1].m128i_i8[0] = 0;
  return a1;
}
