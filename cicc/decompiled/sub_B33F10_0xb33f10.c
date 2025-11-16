// Function: sub_B33F10
// Address: 0xb33f10
//
__int64 __fastcall sub_B33F10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  char v4; // [rsp+Ch] [rbp-44h]
  _BYTE v5[32]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v6; // [rsp+30h] [rbp-20h]

  v4 = BYTE4(a3);
  result = sub_AD64C0(a2, (unsigned int)a3, 0);
  if ( v4 )
  {
    v6 = 257;
    return sub_B33D80(a1, result, (__int64)v5);
  }
  return result;
}
