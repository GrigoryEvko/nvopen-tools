// Function: sub_27A7D80
// Address: 0x27a7d80
//
__int64 __fastcall sub_27A7D80(const __m128i *a1, __int64 a2, __int64 a3)
{
  __int64 m128i_i64; // rbx
  int v4; // eax
  __int64 result; // rax
  __int64 v6[2]; // [rsp+0h] [rbp-40h] BYREF
  __m128i v7; // [rsp+10h] [rbp-30h] BYREF

  m128i_i64 = (__int64)a1[-1].m128i_i64;
  v6[0] = a2;
  v6[1] = a3;
  v7 = _mm_loadu_si128(a1);
  while ( (unsigned __int8)sub_27A2220(v6, v7.m128i_i32, m128i_i64) )
  {
    v4 = *(_DWORD *)m128i_i64;
    m128i_i64 -= 16;
    *(_DWORD *)(m128i_i64 + 32) = v4;
    *(_QWORD *)(m128i_i64 + 40) = *(_QWORD *)(m128i_i64 + 24);
  }
  *(_DWORD *)(m128i_i64 + 16) = v7.m128i_i32[0];
  result = v7.m128i_i64[1];
  *(_QWORD *)(m128i_i64 + 24) = v7.m128i_i64[1];
  return result;
}
