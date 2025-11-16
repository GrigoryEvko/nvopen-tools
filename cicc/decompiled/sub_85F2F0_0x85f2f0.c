// Function: sub_85F2F0
// Address: 0x85f2f0
//
__int64 *__fastcall sub_85F2F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v6; // r10
  __int64 v7; // r15
  __int64 v9; // r13
  __int64 i; // r12
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r11
  __int64 v15; // rsi
  char v16; // al
  unsigned int v17; // eax
  int v18; // edx
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // [rsp+0h] [rbp-50h]
  __int64 v26; // [rsp+0h] [rbp-50h]
  __int64 v27; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  char v34; // [rsp+18h] [rbp-38h]
  unsigned int v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]

  v6 = a5;
  v7 = a4;
  v9 = a1;
  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v12 = *(_QWORD *)i;
  if ( (a6 & 0x80000) != 0 || (unsigned __int8)(*(_BYTE *)(v12 + 80) - 4) > 1u )
  {
    if ( a1 )
    {
LABEL_6:
      v34 = 0;
      v13 = a3;
      v33 = 0;
      v14 = a1;
      goto LABEL_7;
    }
    if ( (unsigned __int8)(*(_BYTE *)(v12 + 80) - 4) > 1u )
    {
      v34 = 0;
      v13 = a3;
      v33 = 0;
      v14 = 0;
      goto LABEL_7;
    }
    goto LABEL_32;
  }
  v33 = *(_QWORD *)(*(_QWORD *)(v12 + 96) + 72LL);
  if ( !v33 || (*(_BYTE *)(*(_QWORD *)(v12 + 88) + 178LL) & 1) != 0 )
  {
    if ( a1 )
      goto LABEL_6;
LABEL_32:
    if ( (*(_DWORD *)(*(_QWORD *)(v12 + 88) + 176LL) & 0x11000) == 0x1000 )
    {
      v26 = a5;
      v31 = *(_QWORD *)i;
      v33 = sub_878920(v12);
      switch ( *(_BYTE *)(v33 + 80) )
      {
        case 4:
        case 5:
          v22 = *(_QWORD *)(*(_QWORD *)(v33 + 96) + 80LL);
          break;
        case 6:
          v22 = *(_QWORD *)(*(_QWORD *)(v33 + 96) + 32LL);
          break;
        case 9:
        case 0xA:
          v22 = *(_QWORD *)(*(_QWORD *)(v33 + 96) + 56LL);
          break;
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
          v22 = *(_QWORD *)(v33 + 88);
          break;
        default:
          v22 = 0;
          break;
      }
      v23 = sub_892400(v22);
      v6 = v26;
      v12 = v31;
      a4 = v7;
      v9 = *(_QWORD *)(v23 + 32);
      v34 = 0;
      v13 = a3;
      a5 = v26;
      v14 = v9;
    }
    else
    {
      v34 = 0;
      v13 = a3;
      v33 = 0;
      v14 = 0;
      v9 = 0;
    }
    goto LABEL_7;
  }
  if ( !a1 )
  {
    v32 = a5;
    v36 = *(_QWORD *)i;
    v24 = sub_892400(*(_QWORD *)(v33 + 88));
    v6 = v32;
    v12 = v36;
    v9 = *(_QWORD *)(v24 + 32);
  }
  v14 = *(_QWORD *)(v9 + 24);
  v34 = 1;
  a5 = 0;
  a4 = 0;
  v13 = 0;
  if ( !a3 )
  {
    v6 = 0;
    v7 = i;
    a3 = v12;
  }
LABEL_7:
  if ( (*(_BYTE *)(v12 + 81) & 0x10) != 0 )
  {
    v30 = v6;
    sub_85F2F0(v14, *(_QWORD *)(v12 + 64), v13, a4, a5);
    v6 = v30;
  }
  else
  {
    if ( (*(_BYTE *)(i + 89) & 2) != 0 )
    {
      v25 = v6;
      v28 = v14;
      v21 = sub_72F070(i);
      v14 = v28;
      v6 = v25;
      v15 = v21;
    }
    else
    {
      v15 = *(_QWORD *)(i + 40);
    }
    if ( v15 )
    {
      v16 = *(_BYTE *)(v15 + 28);
      if ( ((v16 - 15) & 0xFD) == 0 || v16 == 2 )
      {
        v29 = v6;
        sub_85F1C0(v14, v15, 0, a3, v7, v6, a6);
        v6 = v29;
      }
    }
  }
  if ( v34 )
  {
    v17 = *(unsigned __int8 *)(i + 177);
    v18 = 2;
    if ( (v17 & 0x80u) == 0 )
      v18 = (v17 >> 4) & 4;
    v27 = v6;
    if ( (a6 & 0x20000) != 0 )
      v18 |= 0x20000u;
    v35 = v18;
    v19 = sub_892330(i);
    sub_85C120(9u, *(_DWORD *)(v9 + 8), v7, v27, 0, a3, v33, v19, v9, 0, 0, 0, v35);
  }
  return sub_85C120(7u, *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(i + 168) + 152LL) + 24LL), i, 0, 0, 0, 0, 0, 0, 0, 0, 0, a6);
}
