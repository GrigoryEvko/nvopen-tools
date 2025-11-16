// Function: sub_317F330
// Address: 0x317f330
//
__m128i *__fastcall sub_317F330(__int64 a1, __int64 a2, __int64 a3)
{
  const __m128i *v4; // rbx
  __m128i *result; // rax
  __m128i *v6; // r12
  const __m128i *v7; // [rsp+8h] [rbp-28h] BYREF

  v4 = (const __m128i *)sub_317E470(a2);
  result = (__m128i *)sub_317E470(a3);
  if ( v4 )
  {
    v6 = result;
    if ( result )
    {
      result = (__m128i *)sub_C1D5C0(result, v4, 1u);
      v6[3].m128i_i32[0] |= 2u;
      v4[3].m128i_i32[0] |= 8u;
      if ( (v4[3].m128i_i8[4] & 2) != 0 )
        v6[3].m128i_i32[1] |= 2u;
    }
    else
    {
      sub_317E630(a3, (__int64)v4);
      v7 = v4;
      result = (__m128i *)sub_317EE30((unsigned __int64 *)(a1 + 56), (unsigned __int64 *)&v7);
      result->m128i_i64[0] = a3;
      v4[3].m128i_i32[0] |= 2u;
    }
  }
  return result;
}
