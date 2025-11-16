// Function: sub_12F9950
// Address: 0x12f9950
//
__int64 __fastcall sub_12F9950(__int64 *a1)
{
  unsigned __int64 v1; // rsi
  __int64 v2; // r9
  unsigned __int64 v3; // r8
  unsigned __int64 v4; // rdi
  __int64 v5; // rsi
  unsigned __int64 v6; // rdx
  __int64 result; // rax

  v1 = a1[1];
  v2 = *a1;
  v3 = v1;
  v4 = v1 >> 63;
  v5 = HIWORD(v1) & 0x7FFF;
  if ( v5 == 0x7FFF )
  {
    result = (v4 << 15) + 31744;
    if ( (v2 != 0) | v3 & 0xFFFFFFFFFFFFLL )
      return sub_12FAAB8();
  }
  else
  {
    v6 = (((v2 != 0) | v3 & 0x3FFFFFFFFLL) != 0) | (((v2 != 0) | v3 & 0xFFFFFFFFFFFFLL) >> 34);
    if ( v6 | v5 )
    {
      BYTE1(v6) |= 0x40u;
      return sub_12F9B80((unsigned __int8)v4, v5 - 16369, v6);
    }
    else
    {
      return (unsigned __int8)v4 << 15;
    }
  }
  return result;
}
