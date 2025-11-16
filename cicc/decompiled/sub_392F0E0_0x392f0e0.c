// Function: sub_392F0E0
// Address: 0x392f0e0
//
__int64 __fastcall sub_392F0E0(_QWORD *a1, __int64 a2)
{
  _BYTE *v2; // rsi
  __int64 v3; // rax
  _QWORD *v4; // r13
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v8[0] = a2;
  v2 = (_BYTE *)a1[14];
  if ( v2 == (_BYTE *)a1[15] )
  {
    sub_392EF50((__int64)(a1 + 13), v2, v8);
    v3 = v8[0];
  }
  else
  {
    v3 = v8[0];
    if ( v2 )
    {
      *(_QWORD *)v2 = v8[0];
      v2 = (_BYTE *)a1[14];
    }
    a1[14] = v2 + 8;
  }
  v4 = *(_QWORD **)(v3 + 152);
  v5 = *(_QWORD *)(v3 + 160);
  v6 = sub_16D3930(v4, v5);
  sub_1680880((__int64)(a1 + 4), (__int64)v4, (v6 << 32) | (unsigned int)v5);
  return (__int64)(a1[14] - a1[13]) >> 3;
}
