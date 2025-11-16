// Function: sub_2677B10
// Address: 0x2677b10
//
__m128i *__fastcall sub_2677B10(__m128i *a1, __int64 a2, unsigned __int64 *a3)
{
  __m128i *v3; // rax
  __int64 v4; // rcx

  v3 = (__m128i *)sub_2241130(a3, 0, 0, *(_BYTE **)a2, *(_QWORD *)(a2 + 8));
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v3->m128i_i64[0] == &v3[1] )
  {
    a1[1] = _mm_loadu_si128(v3 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v3->m128i_i64[0];
    a1[1].m128i_i64[0] = v3[1].m128i_i64[0];
  }
  v4 = v3->m128i_i64[1];
  v3->m128i_i64[0] = (__int64)v3[1].m128i_i64;
  v3->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v4;
  v3[1].m128i_i8[0] = 0;
  return a1;
}
