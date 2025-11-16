// Function: sub_156C4F0
// Address: 0x156c4f0
//
__int64 __fastcall sub_156C4F0(__int64 *a1, __int64 a2, int a3, char a4)
{
  __int64 v6; // r8
  _BYTE *v7; // rsi
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // r13
  __int64 v12; // rax
  int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // ecx
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  _BYTE v26[16]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v27; // [rsp+10h] [rbp-40h]

  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v7 = *(_BYTE **)(a2 - 24 * v6);
  v8 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
  if ( a3 == 3 )
  {
    v9 = sub_1643320(a1[3]);
    v10 = sub_16463B0(v9, (unsigned int)v8);
    v11 = (_QWORD *)sub_15A06D0(v10);
  }
  else if ( a3 == 7 )
  {
    v24 = sub_1643320(a1[3]);
    v25 = sub_16463B0(v24, (unsigned int)v8);
    v11 = (_QWORD *)sub_15A04A0(v25);
  }
  else
  {
    switch ( a3 )
    {
      case 0:
        v22 = 32;
        break;
      case 1:
        v22 = a4 == 0 ? 36 : 40;
        break;
      case 2:
        v22 = a4 == 0 ? 37 : 41;
        break;
      case 4:
        v22 = 33;
        break;
      case 5:
        v22 = a4 == 0 ? 35 : 39;
        break;
      case 6:
        v22 = a4 == 0 ? 34 : 38;
        break;
    }
    v27 = 257;
    v23 = *(_QWORD *)(a2 + 24 * (1 - v6));
    if ( v7[16] > 0x10u || *(_BYTE *)(v23 + 16) > 0x10u )
      v11 = sub_15641F0(a1, v22, (__int64)v7, *(_QWORD *)(a2 + 24 * (1 - v6)), (__int64)v26);
    else
      v11 = (_QWORD *)sub_15A37B0(v22, v7, v23, 0);
  }
  v12 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( *(char *)(a2 + 23) >= 0 )
  {
    v20 = 0;
    return sub_156C2F0(a1, (__int64)v11, *(_BYTE **)(a2 + 24 * ((unsigned int)(v13 - 2 - v20) - v12)));
  }
  v14 = sub_1648A40(a2);
  v16 = v14 + v15;
  if ( *(char *)(a2 + 23) >= 0 )
  {
    if ( (unsigned int)(v16 >> 4) )
LABEL_26:
      BUG();
LABEL_14:
    v20 = 0;
    v12 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    return sub_156C2F0(a1, (__int64)v11, *(_BYTE **)(a2 + 24 * ((unsigned int)(v13 - 2 - v20) - v12)));
  }
  if ( !(unsigned int)((v16 - sub_1648A40(a2)) >> 4) )
    goto LABEL_14;
  if ( *(char *)(a2 + 23) >= 0 )
    goto LABEL_26;
  v17 = *(_DWORD *)(sub_1648A40(a2) + 8);
  if ( *(char *)(a2 + 23) >= 0 )
    BUG();
  v18 = sub_1648A40(a2);
  v20 = *(_DWORD *)(v18 + v19 - 4) - v17;
  v12 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  return sub_156C2F0(a1, (__int64)v11, *(_BYTE **)(a2 + 24 * ((unsigned int)(v13 - 2 - v20) - v12)));
}
