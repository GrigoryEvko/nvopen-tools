// Function: sub_149D9A0
// Address: 0x149d9a0
//
__int64 __fastcall sub_149D9A0(
        __m128i *a1,
        __m128i *a2,
        __m128i *a3,
        __m128i *a4,
        unsigned __int8 (__fastcall *a5)(__m128i *, __m128i *))
{
  unsigned __int8 v8; // al
  __m128i v9; // xmm1
  __m128i v10; // xmm0
  bool v11; // zf
  __int64 result; // rax
  unsigned __int8 v13; // al

  if ( !a5(a2, a3) )
  {
    if ( a5(a2, a4) )
    {
      v9 = _mm_loadu_si128(a1);
      v10 = _mm_loadu_si128(a1 + 1);
      result = a1[2].m128i_i64[0];
      *a1 = _mm_loadu_si128(a2);
      a1[1] = _mm_loadu_si128(a2 + 1);
      goto LABEL_5;
    }
    v13 = a5(a3, a4);
    v9 = _mm_loadu_si128(a1);
    v10 = _mm_loadu_si128(a1 + 1);
    v11 = v13 == 0;
    result = a1[2].m128i_i64[0];
    if ( !v11 )
      goto LABEL_10;
    *a1 = _mm_loadu_si128(a3);
    a1[1] = _mm_loadu_si128(a3 + 1);
LABEL_9:
    a1[2].m128i_i32[0] = a3[2].m128i_i32[0];
    a3[2].m128i_i32[0] = result;
    a3[1] = v10;
    *a3 = v9;
    return result;
  }
  if ( a5(a3, a4) )
  {
    v9 = _mm_loadu_si128(a1);
    v10 = _mm_loadu_si128(a1 + 1);
    result = a1[2].m128i_i64[0];
    *a1 = _mm_loadu_si128(a3);
    a1[1] = _mm_loadu_si128(a3 + 1);
    goto LABEL_9;
  }
  v8 = a5(a2, a4);
  v9 = _mm_loadu_si128(a1);
  v10 = _mm_loadu_si128(a1 + 1);
  v11 = v8 == 0;
  result = a1[2].m128i_i64[0];
  if ( v11 )
  {
    *a1 = _mm_loadu_si128(a2);
    a1[1] = _mm_loadu_si128(a2 + 1);
LABEL_5:
    a1[2].m128i_i32[0] = a2[2].m128i_i32[0];
    a2[2].m128i_i32[0] = result;
    a2[1] = v10;
    *a2 = v9;
    return result;
  }
LABEL_10:
  *a1 = _mm_loadu_si128(a4);
  a1[1] = _mm_loadu_si128(a4 + 1);
  a1[2].m128i_i32[0] = a4[2].m128i_i32[0];
  a4[2].m128i_i32[0] = result;
  a4[1] = v10;
  *a4 = v9;
  return result;
}
