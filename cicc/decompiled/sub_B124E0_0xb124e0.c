// Function: sub_B124E0
// Address: 0xb124e0
//
__int64 __fastcall sub_B124E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v5; // rsi
  _QWORD v7[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_B10CB0(v7, a3);
  v4 = v7[0];
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = v4;
  if ( v4 )
  {
    sub_B96E90(a1 + 24, v4, 1);
    v5 = v7[0];
    *(_BYTE *)(a1 + 32) = 1;
    if ( v5 )
      sub_B91220(v7);
  }
  else
  {
    *(_BYTE *)(a1 + 32) = 1;
  }
  return sub_B11F90((_QWORD *)(a1 + 40), a2);
}
