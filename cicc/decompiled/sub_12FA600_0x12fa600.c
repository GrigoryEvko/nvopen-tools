// Function: sub_12FA600
// Address: 0x12fa600
//
__int64 __fastcall sub_12FA600(unsigned int a1)
{
  __int64 v1; // r9
  __int64 result; // rax
  unsigned int v3; // edi
  _BYTE v4[40]; // [rsp+0h] [rbp-30h] BYREF

  v1 = a1;
  result = (unsigned __int8)((unsigned __int64)a1 >> 23);
  v3 = a1 & 0x7FFFFF;
  if ( result == 255 )
  {
    if ( v3 )
    {
      sub_12FB8A0(v1, v4);
      return sub_12FBB00(v4);
    }
    else
    {
      return 0;
    }
  }
  else
  {
    if ( result )
      return 0;
    if ( v3 )
    {
      sub_12FBC80(v3);
      return 0;
    }
  }
  return result;
}
