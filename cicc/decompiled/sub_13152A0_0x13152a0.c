// Function: sub_13152A0
// Address: 0x13152a0
//
int __fastcall sub_13152A0(__int64 a1, __int64 a2, unsigned __int64 *a3)
{
  int result; // eax
  _BYTE v4[17]; // [rsp+Fh] [rbp-11h] BYREF

  v4[0] = 0;
  result = sub_130B7F0(a1, a2 + 10648, a3, (__int64)v4);
  if ( v4[0] )
    return sub_1314D40(a1, a2);
  return result;
}
