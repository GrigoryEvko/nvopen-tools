// Function: sub_CB2220
// Address: 0xcb2220
//
__int64 __fastcall sub_CB2220(__m128i *a1)
{
  __int64 result; // rax

  if ( *(_DWORD *)(a1[2].m128i_i64[0] + 4LL * a1[2].m128i_u32[2] - 4) == 4 )
  {
    a1[6] = _mm_loadu_si128(a1 + 7);
    sub_CB20A0((__int64)a1, 0);
    sub_CB1B10((__int64)a1, "{}", 2u);
    a1[6].m128i_i64[1] = 1;
    a1[6].m128i_i64[0] = (__int64)"\n";
  }
  result = (unsigned int)(a1[2].m128i_i32[2] - 1);
  a1[2].m128i_i32[2] = result;
  return result;
}
