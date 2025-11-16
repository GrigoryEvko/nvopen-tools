// Function: sub_887160
// Address: 0x887160
//
_QWORD *__fastcall sub_887160(__int64 a1, __int64 a2, int a3, _BYTE *a4)
{
  __int64 v4; // rax
  int i; // edx
  __int64 v6; // r14
  __int64 v7; // rax
  _QWORD *v8; // r15
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rax
  char v15; // al
  __int64 v16; // r15
  char v17; // cl
  char v18; // al
  char v19; // dl
  char v20; // al
  char v21; // al
  char v22; // al
  char v23; // dl
  char v24; // dl
  __int64 v25; // rax
  __int64 v26; // rax
  int v28; // eax

  if ( *(_BYTE *)(a2 + 80) == 17 )
  {
    v12 = (_QWORD *)a2;
    goto LABEL_32;
  }
  if ( !a3 )
  {
    if ( (*(_BYTE *)(a2 + 82) & 8) != 0 )
    {
      v28 = sub_7CF970();
      if ( v28 == -1 )
        BUG();
      v4 = qword_4F04C68[0] + 776LL * v28;
    }
    else
    {
      v4 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
      for ( i = *(_DWORD *)(a2 + 40); *(_DWORD *)v4 != i; v4 -= 776 )
        ;
    }
    goto LABEL_6;
  }
  if ( !a4 )
  {
    v4 = qword_4F04C68[0];
LABEL_6:
    v6 = *(_QWORD *)(v4 + 24);
    v7 = v4 + 32;
    if ( !v6 )
      v6 = v7;
    goto LABEL_8;
  }
  if ( (a4[124] & 1) != 0 )
    a4 = (_BYTE *)sub_735B70((__int64)a4);
  v6 = *(_QWORD *)(*(_QWORD *)a4 + 96LL);
LABEL_8:
  v8 = *(_QWORD **)a2;
  v12 = sub_87EBB0(0x11u, *(_QWORD *)a2, (_QWORD *)(a2 + 48));
  *((_DWORD *)v12 + 10) = *(_DWORD *)(a2 + 40);
  *((_DWORD *)v12 + 11) = *(_DWORD *)(a2 + 44);
  *((_BYTE *)v12 + 84) = *(_BYTE *)(a2 + 84) & 4 | *((_BYTE *)v12 + 84) & 0xFB;
  v13 = *(_QWORD *)(a2 + 64);
  if ( (*(_BYTE *)(a2 + 81) & 0x10) != 0 )
  {
    sub_877E20((__int64)v12, 0, v13, v9, v10, v11);
  }
  else if ( v13 )
  {
    sub_877E90((__int64)v12, 0, v13);
  }
  if ( (*(_BYTE *)(a2 + 83) & 1) != 0 )
    goto LABEL_21;
  if ( (*(_BYTE *)(a2 + 82) & 8) != 0 )
  {
    v14 = v8[5];
    if ( a2 == v14 )
    {
      v8[5] = v12;
      goto LABEL_20;
    }
    while ( 1 )
    {
LABEL_17:
      if ( !v14 )
      {
        MEMORY[8] = v12;
        BUG();
      }
      if ( *(_QWORD *)(v14 + 8) == a2 )
        break;
      v14 = *(_QWORD *)(v14 + 8);
    }
    *(_QWORD *)(v14 + 8) = v12;
    goto LABEL_20;
  }
  if ( (*(_BYTE *)(v6 + 144) & 1) == 0 )
  {
    v14 = v8[3];
    if ( a2 == v14 )
    {
      v8[3] = v12;
      goto LABEL_20;
    }
    goto LABEL_17;
  }
  v14 = v8[4];
  if ( a2 != v14 )
    goto LABEL_17;
  v8[4] = v12;
LABEL_20:
  v12[1] = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a2 + 8) = 0;
LABEL_21:
  v15 = *(_BYTE *)(a2 + 82);
  if ( (v15 & 8) != 0 )
  {
    v16 = *(_QWORD *)(v6 + 8);
    if ( a2 == v16 )
    {
      *(_QWORD *)(v6 + 8) = v12;
      v15 = *(_BYTE *)(a2 + 82);
    }
    v17 = *((_BYTE *)v12 + 83) & 0xFE;
    v18 = v15 & 8 | *((_BYTE *)v12 + 82) & 0xF7;
    *((_BYTE *)v12 + 82) = v18;
    v19 = v17 | *(_BYTE *)(a2 + 83) & 1;
    *((_BYTE *)v12 + 83) = v19;
    v20 = *(_BYTE *)(a2 + 82) & 0x10 | v18 & 0xEF;
    *((_BYTE *)v12 + 82) = v20;
    v21 = *(_BYTE *)(a2 + 82) & 0x20 | v20 & 0xDF;
    *((_BYTE *)v12 + 82) = v21;
    v22 = *(_BYTE *)(a2 + 82) & 0x40 | v21 & 0xBF;
    *((_BYTE *)v12 + 82) = v22;
    *((_BYTE *)v12 + 82) = *(_BYTE *)(a2 + 82) & 0x80 | v22 & 0x7F;
    v23 = *(_BYTE *)(a2 + 83) & 4 | v19 & 0xFB;
    *((_BYTE *)v12 + 83) = v23;
    v24 = *(_BYTE *)(a2 + 83) & 8 | v23 & 0xF7;
    *((_BYTE *)v12 + 83) = v24;
    *((_BYTE *)v12 + 83) = *(_BYTE *)(a2 + 83) & 2 | v24 & 0xFD;
  }
  else
  {
    v16 = *(_QWORD *)v6;
    if ( a2 == *(_QWORD *)v6 )
      *(_QWORD *)v6 = v12;
    sub_881D30((__int64 *)a2, v6);
    sub_885590(v12, *(unsigned __int8 **)(v6 + 136));
  }
  v12[2] = *(_QWORD *)(a2 + 16);
  v25 = *(_QWORD *)(a2 + 24);
  v12[3] = v25;
  if ( v16 != a2 )
    *(_QWORD *)(v25 + 16) = v12;
  v26 = *(_QWORD *)(a2 + 16);
  if ( v26 )
    *(_QWORD *)(v26 + 24) = v12;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 24) = 0;
  if ( *(_QWORD *)(v6 + 16) == a2 )
    *(_QWORD *)(v6 + 16) = v12;
  v12[11] = a2;
  *(_BYTE *)(a2 + 83) |= 0x20u;
LABEL_32:
  *(_QWORD *)(a1 + 8) = v12[11];
  v12[11] = a1;
  *(_BYTE *)(a1 + 83) |= 0x20u;
  return v12;
}
