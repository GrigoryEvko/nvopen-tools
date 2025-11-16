// Function: sub_130A0F0
// Address: 0x130a0f0
//
__int64 __fastcall sub_130A0F0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r13
  __int64 result; // rax
  _BYTE v4[17]; // [rsp+Fh] [rbp-11h] BYREF

  v2 = qword_50579C0[*a2 & 0xFFFLL];
  v4[0] = 0;
  result = sub_130B7F0(a1, v2 + 10648, a2, v4);
  if ( v4[0] )
    return sub_1314D40(a1, v2);
  return result;
}
