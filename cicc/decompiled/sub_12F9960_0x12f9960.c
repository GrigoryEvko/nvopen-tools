// Function: sub_12F9960
// Address: 0x12f9960
//
__int64 __fastcall sub_12F9960(__int64 *a1)
{
  __int64 v1; // r8
  __int64 v2; // rdi
  unsigned __int64 v3; // rax
  __int64 v4; // rsi
  unsigned __int64 v5; // rdx
  __int64 result; // rax

  v1 = *a1;
  v2 = a1[1];
  v3 = (unsigned __int64)v2 >> 63;
  v4 = HIWORD(v2) & 0x7FFF;
  if ( v4 == 0x7FFF )
  {
    result = (unsigned int)(((_DWORD)v3 << 31) + 2139095040);
    if ( (v1 != 0) | v2 & 0xFFFFFFFFFFFFLL )
      return sub_12FAB80();
  }
  else
  {
    v5 = (((v1 != 0) | v2 & 0x3FFFF) != 0) | (((v1 != 0) | v2 & 0xFFFFFFFFFFFFuLL) >> 18);
    if ( v5 | v4 )
      return sub_12F9D30(v2 < 0, v4 - 16257, v5 | 0x40000000);
    else
      return (unsigned int)((_DWORD)v3 << 31);
  }
  return result;
}
