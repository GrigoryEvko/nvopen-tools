// Function: sub_2ABD910
// Address: 0x2abd910
//
const __m128i *__fastcall sub_2ABD910(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *result; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // r12
  __m128i *v10; // rdi
  const __m128i *v11; // rcx
  const __m128i *v12; // r8
  __m128i *v13; // rdx

  result = (const __m128i *)sub_C8CD80((__int64)a1, (__int64)(a1 + 4), a2, a4, a5, a6);
  v9 = *(_QWORD *)(a2 + 104) - *(_QWORD *)(a2 + 96);
  a1[12] = 0;
  a1[13] = 0;
  a1[14] = 0;
  if ( v9 )
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a1 + 4, v8);
    result = (const __m128i *)sub_22077B0(v9);
    v10 = (__m128i *)result;
  }
  else
  {
    v9 = 0;
    v10 = 0;
  }
  a1[12] = v10;
  a1[13] = v10;
  a1[14] = (char *)v10 + v9;
  v11 = *(const __m128i **)(a2 + 104);
  v12 = *(const __m128i **)(a2 + 96);
  if ( v11 != v12 )
  {
    v13 = v10;
    result = *(const __m128i **)(a2 + 96);
    do
    {
      if ( v13 )
      {
        *v13 = _mm_loadu_si128(result);
        v13[1].m128i_i64[0] = result[1].m128i_i64[0];
      }
      result = (const __m128i *)((char *)result + 24);
      v13 = (__m128i *)((char *)v13 + 24);
    }
    while ( v11 != result );
    v10 = (__m128i *)((char *)v10 + 8 * ((unsigned __int64)((char *)&v11[-2].m128i_u64[1] - (char *)v12) >> 3) + 24);
  }
  a1[13] = v10;
  return result;
}
