// Function: sub_12FA540
// Address: 0x12fa540
//
__int64 __fastcall sub_12FA540(unsigned __int16 a1)
{
  __int64 v1; // r8
  __int16 v2; // dx
  unsigned __int64 v3; // rax
  __int64 v4; // rdi
  __int64 result; // rax
  _BYTE v6[40]; // [rsp+0h] [rbp-30h] BYREF

  v1 = a1;
  v2 = a1 & 0x3FF;
  v3 = ((unsigned __int64)a1 >> 10) & 0x1F;
  v4 = a1 & 0x3FF;
  if ( (_BYTE)v3 == 31 )
  {
    if ( v2 )
    {
      sub_12FB820(v1, v6);
      return sub_12FBB00(v6);
    }
    else
    {
      return 0;
    }
  }
  else
  {
    if ( (_BYTE)v3 )
      return 0;
    result = 0;
    if ( v2 )
    {
      sub_12FBC30(v4);
      return 0;
    }
  }
  return result;
}
