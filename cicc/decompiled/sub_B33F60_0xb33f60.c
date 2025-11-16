// Function: sub_B33F60
// Address: 0xb33f60
//
__int64 __fastcall sub_B33F60(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 result; // rax
  _BYTE v6[32]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v7; // [rsp+30h] [rbp-20h]

  result = sub_AD64C0(a2, a3, 0);
  if ( a4 )
  {
    v7 = 257;
    return sub_B33D80(a1, result, (__int64)v6);
  }
  return result;
}
