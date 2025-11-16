// Function: sub_14EC200
// Address: 0x14ec200
//
__m128i *__fastcall sub_14EC200(__m128i *a1, const __m128i *a2, const __m128i *a3)
{
  char v3; // cl
  __m128i *result; // rax
  char v5; // di
  __m128i v6; // xmm0
  __int64 v7; // rdx
  __int64 v8; // rdx

  v3 = a2[1].m128i_i8[0];
  result = a1;
  if ( v3 && (v5 = a3[1].m128i_i8[0]) != 0 )
  {
    if ( v3 == 1 )
    {
      v6 = _mm_loadu_si128(a3);
      v7 = a3[1].m128i_i64[0];
      *result = v6;
      result[1].m128i_i64[0] = v7;
    }
    else if ( v5 == 1 )
    {
      v8 = a2[1].m128i_i64[0];
      *result = _mm_loadu_si128(a2);
      result[1].m128i_i64[0] = v8;
    }
    else
    {
      if ( a2[1].m128i_i8[1] == 1 )
        a2 = (const __m128i *)a2->m128i_i64[0];
      else
        v3 = 2;
      if ( a3[1].m128i_i8[1] == 1 )
        a3 = (const __m128i *)a3->m128i_i64[0];
      else
        v5 = 2;
      result->m128i_i64[0] = (__int64)a2;
      result->m128i_i64[1] = (__int64)a3;
      result[1].m128i_i8[0] = v3;
      result[1].m128i_i8[1] = v5;
    }
  }
  else
  {
    result[1].m128i_i16[0] = 256;
  }
  return result;
}
