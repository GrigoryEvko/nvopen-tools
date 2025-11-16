// Function: sub_9C6370
// Address: 0x9c6370
//
__m128i *__fastcall sub_9C6370(__m128i *a1, const __m128i *a2, const __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // cl
  __m128i *result; // rax
  char v8; // di
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __int64 v11; // rdx
  __m128i v12; // xmm3
  __int64 v13; // rdx

  v6 = a2[2].m128i_i8[0];
  result = a1;
  if ( v6 && (v8 = a3[2].m128i_i8[0]) != 0 )
  {
    if ( v6 == 1 )
    {
      v9 = _mm_loadu_si128(a3);
      v10 = _mm_loadu_si128(a3 + 1);
      v11 = a3[2].m128i_i64[0];
      *result = v9;
      result[2].m128i_i64[0] = v11;
      result[1] = v10;
    }
    else if ( v8 == 1 )
    {
      v12 = _mm_loadu_si128(a2 + 1);
      v13 = a2[2].m128i_i64[0];
      *result = _mm_loadu_si128(a2);
      result[2].m128i_i64[0] = v13;
      result[1] = v12;
    }
    else
    {
      if ( a2[2].m128i_i8[1] == 1 )
      {
        a6 = a2->m128i_i64[1];
        a2 = (const __m128i *)a2->m128i_i64[0];
      }
      else
      {
        v6 = 2;
      }
      if ( a3[2].m128i_i8[1] == 1 )
      {
        a5 = a3->m128i_i64[1];
        a3 = (const __m128i *)a3->m128i_i64[0];
      }
      else
      {
        v8 = 2;
      }
      result->m128i_i64[0] = (__int64)a2;
      result->m128i_i64[1] = a6;
      result[1].m128i_i64[0] = (__int64)a3;
      result[1].m128i_i64[1] = a5;
      result[2].m128i_i8[0] = v6;
      result[2].m128i_i8[1] = v8;
    }
  }
  else
  {
    result[2].m128i_i16[0] = 256;
  }
  return result;
}
