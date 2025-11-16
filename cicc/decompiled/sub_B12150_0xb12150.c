// Function: sub_B12150
// Address: 0xb12150
//
__int64 __fastcall sub_B12150(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v9; // rsi
  __int64 v10; // rsi
  __int64 result; // rax
  _QWORD v13[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_B10CB0(v13, a5);
  v9 = v13[0];
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = v9;
  if ( v9 )
  {
    sub_B96E90(a1 + 24, v9, 1);
    v10 = v13[0];
    *(_BYTE *)(a1 + 32) = 0;
    if ( v10 )
      sub_B91220(v13);
  }
  else
  {
    *(_BYTE *)(a1 + 32) = 0;
  }
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 40) = a2;
  sub_B96F80(a1 + 40);
  *(_BYTE *)(a1 + 64) = a6;
  sub_B11FC0((_QWORD *)(a1 + 72), a3);
  result = sub_B11F20((_QWORD *)(a1 + 80), a4);
  *(_QWORD *)(a1 + 88) = 0;
  return result;
}
