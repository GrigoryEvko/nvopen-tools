// Function: sub_2BE03F0
// Address: 0x2be03f0
//
unsigned __int64 __fastcall sub_2BE03F0(unsigned __int64 *a1, const __m128i *a2)
{
  __m128i *v3; // rsi
  __m128i v4; // xmm0
  __m128i v5; // xmm2
  bool v6; // zf
  __int64 v7; // rax
  __int64 v8; // rcx
  __m128i v9; // xmm3
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // rsi

  v3 = (__m128i *)a1[8];
  if ( v3 == (__m128i *)a1[9] )
  {
    sub_2BE00E0(a1 + 7, v3, a2);
    v11 = a1[8];
  }
  else
  {
    if ( v3 )
    {
      v4 = _mm_loadu_si128(a2 + 1);
      v5 = _mm_loadu_si128(a2 + 2);
      v6 = a2->m128i_i32[0] == 11;
      *v3 = _mm_loadu_si128(a2);
      v3[1] = v4;
      v3[2] = v5;
      if ( v6 )
      {
        v3[2].m128i_i64[0] = 0;
        v7 = a2[2].m128i_i64[0];
        v8 = v3[2].m128i_i64[1];
        v9 = _mm_loadu_si128(a2 + 1);
        a2[2].m128i_i64[0] = 0;
        v3[2].m128i_i64[0] = v7;
        v10 = a2[2].m128i_i64[1];
        a2[2].m128i_i64[1] = v8;
        v3[2].m128i_i64[1] = v10;
        v3[1] = v9;
        a2[1] = v4;
      }
      v3 = (__m128i *)a1[8];
    }
    v11 = (unsigned __int64)&v3[3];
    a1[8] = v11;
  }
  v12 = v11 - a1[7];
  if ( (unsigned __int64)v12 > 0x493E00 )
    abort();
  return 0xAAAAAAAAAAAAAAABLL * (v12 >> 4) - 1;
}
