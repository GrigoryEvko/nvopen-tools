// Function: sub_12F99A0
// Address: 0x12f99a0
//
__int64 __fastcall sub_12F99A0(unsigned __int64 *a1)
{
  unsigned __int64 v1; // rsi
  unsigned __int64 v2; // r8
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 result; // rax

  v2 = *a1;
  v1 = a1[1];
  v3 = v1 >> 63;
  v4 = v1 & 0xFFFFFFFFFFFFLL;
  v5 = HIWORD(v1) & 0x7FFF;
  if ( v5 == 0x7FFF )
  {
    result = (v3 << 63) + 0x7FF0000000000000LL;
    if ( v2 | v4 )
      return sub_12FAD30();
  }
  else
  {
    v6 = (v2 << 14 != 0) | (v2 >> 50) | (v4 << 14);
    if ( v6 | v5 )
      return sub_12FBCF0((unsigned __int8)v3, v5 - 15361, v6 | 0x4000000000000000LL);
    else
      return v3 << 63;
  }
  return result;
}
