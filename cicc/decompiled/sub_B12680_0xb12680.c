// Function: sub_B12680
// Address: 0xb12680
//
__int64 __fastcall sub_B12680(
        __int64 a1,
        char a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // rsi
  __int64 v12; // rsi
  _QWORD v17[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_B10CB0(v17, a9);
  v11 = v17[0];
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = v11;
  if ( v11 )
  {
    sub_B96E90(a1 + 24, v11, 1);
    v12 = v17[0];
    *(_BYTE *)(a1 + 32) = 0;
    if ( v12 )
      sub_B91220(v17);
  }
  else
  {
    *(_BYTE *)(a1 + 32) = 0;
  }
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 48) = a7;
  *(_QWORD *)(a1 + 56) = a6;
  sub_B96F80(a1 + 40);
  *(_BYTE *)(a1 + 64) = a2;
  sub_B11FE0((_QWORD *)(a1 + 72), a4);
  sub_B11F40((_QWORD *)(a1 + 80), a5);
  return sub_B11F40((_QWORD *)(a1 + 88), a8);
}
