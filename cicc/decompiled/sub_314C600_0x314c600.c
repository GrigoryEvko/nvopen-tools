// Function: sub_314C600
// Address: 0x314c600
//
__int64 __fastcall sub_314C600(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 v3; // [rsp+Fh] [rbp-21h]
  __int64 v4; // [rsp+18h] [rbp-18h] BYREF

  v4 = 0;
  result = sub_CE7BB0(a1, "full_custom_abi", 0xFu, &v4);
  if ( a2 )
  {
    if ( (_BYTE)result )
    {
      v3 = result;
      sub_314BE80(v4, a2);
      return v3;
    }
  }
  return result;
}
