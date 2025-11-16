// Function: sub_130A160
// Address: 0x130a160
//
__int64 __fastcall sub_130A160(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r14
  __int64 result; // rax
  _BYTE v5[33]; // [rsp+Fh] [rbp-21h] BYREF

  v2 = qword_50579C0[*a2 & 0xFFFLL];
  sub_13094D0(a1, v2, (__int64)a2, 0);
  v5[0] = 0;
  result = sub_130B7F0(a1, v2 + 10648, a2, v5);
  if ( v5[0] )
    result = sub_1314D40(a1, v2);
  if ( a1 )
  {
    if ( --*(_DWORD *)(a1 + 152) < 0 )
    {
      result = sub_1309470((_DWORD *)(a1 + 152), (unsigned __int64 *)(a1 + 112));
      if ( (_BYTE)result )
        return sub_1315160(a1, v2, 0, 0);
    }
  }
  return result;
}
