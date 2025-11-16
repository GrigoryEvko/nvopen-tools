// Function: sub_94F930
// Address: 0x94f930
//
__m128i *__fastcall sub_94F930(__m128i *a1, __int64 a2, const char *a3)
{
  size_t v4; // rdx
  __int64 v5; // rcx
  __m128i *v6; // rax
  __int64 v7; // rcx

  v4 = strlen(a3);
  if ( v4 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a2 + 8) )
    sub_4262D8((__int64)"basic_string::append");
  v6 = (__m128i *)sub_2241490(a2, a3, v4, v5);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    a1[1] = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v6->m128i_i64[0];
    a1[1].m128i_i64[0] = v6[1].m128i_i64[0];
  }
  v7 = v6->m128i_i64[1];
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v6->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v7;
  v6[1].m128i_i8[0] = 0;
  return a1;
}
