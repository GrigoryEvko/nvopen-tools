// Function: sub_B13D50
// Address: 0xb13d50
//
__int64 __fastcall sub_B13D50(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rsi
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 result; // rax
  int v9; // edx
  __int64 v10; // r14
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rsi
  _QWORD v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD *)(a2 + 48);
  v15[0] = v3;
  if ( v3 )
  {
    sub_B96E90(v15, v3, 1);
    v4 = v15[0];
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = v4;
    if ( v4 )
    {
      sub_B96E90(a1 + 24, v4, 1);
      v5 = v15[0];
      *(_BYTE *)(a1 + 32) = 0;
      if ( v5 )
        sub_B91220(v15);
      goto LABEL_5;
    }
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
  }
  *(_BYTE *)(a1 + 32) = 0;
LABEL_5:
  v6 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 24LL);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 40) = v6;
  sub_B96F80(a1 + 40);
  sub_B11FC0((_QWORD *)(a1 + 72), *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL));
  sub_B11F20((_QWORD *)(a1 + 80), *(_QWORD *)(*(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL));
  v7 = *(_QWORD *)(a2 - 32);
  *(_QWORD *)(a1 + 88) = 0;
  if ( !v7 || *(_BYTE *)v7 || *(_QWORD *)(v7 + 24) != *(_QWORD *)(a2 + 80) )
    goto LABEL_24;
  result = *(unsigned int *)(v7 + 36);
  if ( (_DWORD)result == 69 )
  {
    *(_BYTE *)(a1 + 64) = 0;
    return result;
  }
  if ( (_DWORD)result == 71 )
  {
    *(_BYTE *)(a1 + 64) = 1;
    return result;
  }
  if ( (_DWORD)result != 68 )
LABEL_24:
    BUG();
  v9 = *(_DWORD *)(a2 + 4);
  *(_BYTE *)(a1 + 64) = 2;
  v10 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (4LL - (v9 & 0x7FFFFFF))) + 24LL);
  sub_B91340(a1 + 40, 1);
  *(_QWORD *)(a1 + 48) = v10;
  sub_B96F50(a1 + 40, 1);
  sub_B11F20(v15, *(_QWORD *)(*(_QWORD *)(a2 + 32 * (5LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL));
  if ( (_QWORD *)(a1 + 88) == v15 )
  {
    if ( v15[0] )
      sub_B91220(a1 + 88);
  }
  else
  {
    if ( *(_QWORD *)(a1 + 88) )
      sub_B91220(a1 + 88);
    v14 = v15[0];
    *(_QWORD *)(a1 + 88) = v15[0];
    if ( v14 )
      sub_B976B0(v15, v14, a1 + 88, v11, v12, v13);
  }
  return sub_B13D10(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL));
}
