// Function: sub_2BE99F0
// Address: 0x2be99f0
//
void __fastcall sub_2BE99F0(unsigned __int64 *a1, __m128i *a2)
{
  __m128i *v3; // rsi
  __int64 v4; // rcx
  __m128i *v5; // rcx
  __int64 v6; // rcx

  v3 = (__m128i *)a1[1];
  if ( v3 == (__m128i *)a1[2] )
  {
    sub_2BE9660(a1, v3, a2);
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
      v3[2].m128i_i64[0] = (__int64)v3[3].m128i_i64;
      v5 = (__m128i *)a2[2].m128i_i64[0];
      if ( v5 == &a2[3] )
      {
        v3[3] = _mm_loadu_si128(a2 + 3);
      }
      else
      {
        v3[2].m128i_i64[0] = (__int64)v5;
        v3[3].m128i_i64[0] = a2[3].m128i_i64[0];
      }
      v6 = a2[2].m128i_i64[1];
      a2[2].m128i_i64[0] = (__int64)a2[3].m128i_i64;
      a2[2].m128i_i64[1] = 0;
      v3[2].m128i_i64[1] = v6;
      a2[3].m128i_i8[0] = 0;
      v3 = (__m128i *)a1[1];
    }
    a1[1] = (unsigned __int64)&v3[4];
  }
}
