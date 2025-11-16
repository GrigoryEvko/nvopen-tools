// Function: sub_8FC460
// Address: 0x8fc460
//
__m128i *__fastcall sub_8FC460(__m128i *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __m128i *v4; // rax
  __int64 v5; // rcx

  v4 = (__m128i *)sub_2241490(a2, a3, a4, a4);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v4->m128i_i64[0] == &v4[1] )
  {
    a1[1] = _mm_loadu_si128(v4 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v4->m128i_i64[0];
    a1[1].m128i_i64[0] = v4[1].m128i_i64[0];
  }
  v5 = v4->m128i_i64[1];
  v4->m128i_i64[0] = (__int64)v4[1].m128i_i64;
  v4->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v5;
  v4[1].m128i_i8[0] = 0;
  return a1;
}
