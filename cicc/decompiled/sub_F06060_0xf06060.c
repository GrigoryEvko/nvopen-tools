// Function: sub_F06060
// Address: 0xf06060
//
void __fastcall sub_F06060(__m128i **a1, __m128i *a2)
{
  __m128i *v3; // rsi
  __int64 v4; // rcx

  v3 = a1[1];
  if ( v3 == a1[2] )
  {
    sub_8F99A0(a1, v3, a2);
  }
  else
  {
    if ( v3 )
    {
      v3->m128i_i64[0] = (__int64)v3[1].m128i_i64;
      if ( (__m128i *)a2->m128i_i64[0] == &a2[1] )
      {
        v3[1] = _mm_loadu_si128(a2 + 1);
      }
      else
      {
        v3->m128i_i64[0] = a2->m128i_i64[0];
        v3[1].m128i_i64[0] = a2[1].m128i_i64[0];
      }
      v4 = a2->m128i_i64[1];
      a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
      a2->m128i_i64[1] = 0;
      v3->m128i_i64[1] = v4;
      a2[1].m128i_i8[0] = 0;
      v3 = a1[1];
    }
    a1[1] = v3 + 2;
  }
}
