// Function: sub_B12230
// Address: 0xb12230
//
__int64 __fastcall sub_B12230(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v10; // rsi
  __int64 v11; // rsi
  _QWORD v16[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_B10CB0(v16, a8);
  v10 = v16[0];
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = v10;
  if ( v10 )
  {
    sub_B96E90(a1 + 24, v10, 1);
    v11 = v16[0];
    *(_BYTE *)(a1 + 32) = 0;
    if ( v11 )
      sub_B91220(v16);
  }
  else
  {
    *(_BYTE *)(a1 + 32) = 0;
  }
  *(_QWORD *)(a1 + 40) = a2;
  *(_QWORD *)(a1 + 48) = a6;
  *(_QWORD *)(a1 + 56) = a5;
  sub_B96F80(a1 + 40);
  *(_BYTE *)(a1 + 64) = 2;
  sub_B11FC0((_QWORD *)(a1 + 72), a3);
  sub_B11F20((_QWORD *)(a1 + 80), a4);
  return sub_B11F20((_QWORD *)(a1 + 88), a7);
}
