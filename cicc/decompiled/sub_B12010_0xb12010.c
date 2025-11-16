// Function: sub_B12010
// Address: 0xb12010
//
__int64 __fastcall sub_B12010(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rsi
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // rsi
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD *)(a2 + 24);
  v12[0] = v3;
  if ( !v3 )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_8;
  }
  sub_B96E90(v12, v3, 1);
  v4 = v12[0];
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = v4;
  if ( !v4 )
  {
LABEL_8:
    *(_BYTE *)(a1 + 32) = 0;
    goto LABEL_5;
  }
  sub_B96E90(a1 + 24, v4, 1);
  v5 = v12[0];
  *(_BYTE *)(a1 + 32) = 0;
  if ( v5 )
    sub_B91220(v12);
LABEL_5:
  v6 = *(_QWORD *)(a2 + 48);
  v7 = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 48) = v6;
  *(_QWORD *)(a1 + 56) = v7;
  sub_B96F80(a1 + 40);
  *(_BYTE *)(a1 + 64) = *(_BYTE *)(a2 + 64);
  v8 = sub_B12000(a2 + 72);
  sub_B11FC0((_QWORD *)(a1 + 72), v8);
  v9 = sub_B11F60(a2 + 80);
  result = sub_B11F20((_QWORD *)(a1 + 80), v9);
  v11 = *(_QWORD *)(a2 + 88);
  *(_QWORD *)(a1 + 88) = v11;
  if ( v11 )
    return sub_B96E90(a1 + 88, v11, 1);
  return result;
}
