// Function: sub_CB0550
// Address: 0xcb0550
//
__int64 __fastcall sub_CB0550(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __m128i v7; // xmm0

  v6 = a1[2].m128i_u32[2];
  if ( v6 + 1 > (unsigned __int64)a1[2].m128i_u32[3] )
  {
    sub_C8D5F0((__int64)a1[2].m128i_i64, &a1[3], v6 + 1, 4u, a5, a6);
    v6 = a1[2].m128i_u32[2];
  }
  *(_DWORD *)(a1[2].m128i_i64[0] + 4 * v6) = 0;
  v7 = _mm_loadu_si128(a1 + 6);
  a1[6].m128i_i64[0] = (__int64)"\n";
  ++a1[2].m128i_i32[2];
  a1[6].m128i_i64[1] = 1;
  a1[7] = v7;
  return 0;
}
