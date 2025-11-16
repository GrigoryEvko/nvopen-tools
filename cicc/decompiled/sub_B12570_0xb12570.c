// Function: sub_B12570
// Address: 0xb12570
//
__int64 __fastcall sub_B12570(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rsi
  __int64 v5; // rsi
  __int64 v6; // rsi
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *a3;
  v8[0] = v4;
  if ( !v4 )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_7;
  }
  sub_B96E90(v8, v4, 1);
  v5 = v8[0];
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = v5;
  if ( !v5 )
  {
LABEL_7:
    *(_BYTE *)(a1 + 32) = 1;
    return sub_B11F70((_QWORD *)(a1 + 40), a2);
  }
  sub_B96E90(a1 + 24, v5, 1);
  v6 = v8[0];
  *(_BYTE *)(a1 + 32) = 1;
  if ( v6 )
    sub_B91220(v8);
  return sub_B11F70((_QWORD *)(a1 + 40), a2);
}
