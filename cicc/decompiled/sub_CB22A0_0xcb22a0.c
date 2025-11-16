// Function: sub_CB22A0
// Address: 0xcb22a0
//
__int64 __fastcall sub_CB22A0(__m128i *a1)
{
  __int64 v1; // rcx
  __int64 result; // rax

  v1 = a1[2].m128i_u32[2];
  if ( *(_DWORD *)(a1[2].m128i_i64[0] + 4 * v1 - 4) )
  {
    result = (unsigned int)(v1 - 1);
  }
  else
  {
    a1[6] = _mm_loadu_si128(a1 + 7);
    sub_CB20A0((__int64)a1, 1);
    sub_CB1B10((__int64)a1, "[]", 2u);
    a1[6].m128i_i64[1] = 1;
    a1[6].m128i_i64[0] = (__int64)"\n";
    result = (unsigned int)(a1[2].m128i_i32[2] - 1);
  }
  a1[2].m128i_i32[2] = result;
  return result;
}
