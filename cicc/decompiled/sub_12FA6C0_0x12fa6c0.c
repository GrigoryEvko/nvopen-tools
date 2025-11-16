// Function: sub_12FA6C0
// Address: 0x12fa6c0
//
__int64 __fastcall sub_12FA6C0(unsigned __int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // rax
  __int64 v4; // rdx
  _BYTE v5[40]; // [rsp+0h] [rbp-30h] BYREF

  v1 = a1 & 0xFFFFFFFFFFFFFLL;
  v2 = (a1 >> 52) & 0x7FF;
  if ( v2 == 2047 )
  {
    if ( v1 )
    {
      sub_12FB920(a1, v5);
      return sub_12FBB00(v5);
    }
  }
  else
  {
    if ( !v2 )
    {
      if ( !v1 )
        return v1;
      sub_12FBCB0(a1 & 0xFFFFFFFFFFFFFLL);
      v1 = v4;
    }
    v1 <<= 60;
  }
  return v1;
}
