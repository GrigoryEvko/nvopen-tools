// Function: sub_23DFA50
// Address: 0x23dfa50
//
__int64 __fastcall sub_23DFA50(__m128i *a1, const __m128i *a2, __int8 a3, __int8 a4, __int32 a5, __int32 a6)
{
  __m128i v6; // xmm0
  __int64 result; // rax

  v6 = _mm_loadu_si128(a2);
  result = a2[1].m128i_u32[0];
  a1[1].m128i_i8[4] = a3;
  a1[1].m128i_i8[5] = a4;
  a1[1].m128i_i32[0] = result;
  a1[1].m128i_i32[2] = a5;
  a1[1].m128i_i32[3] = a6;
  *a1 = v6;
  return result;
}
