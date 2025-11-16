// Function: sub_6F7150
// Address: 0x6f7150
//
__int64 __fastcall sub_6F7150(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  result = sub_6F6F40(a1, 0, a3, a4, a5, a6);
  if ( a1[8].m128i_i64[0] )
    *(_BYTE *)(result + 26) |= 4u;
  return result;
}
