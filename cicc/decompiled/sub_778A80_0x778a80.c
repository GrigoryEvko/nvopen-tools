// Function: sub_778A80
// Address: 0x778a80
//
__int64 __fastcall sub_778A80(__int64 a1, __int64 a2, FILE *a3, _QWORD *a4, __int64 a5, __int64 a6, __int64 a7)
{
  int v8; // r14d
  _QWORD *v9; // r12
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned __int8 v12; // al
  unsigned __int64 v13; // rsi
  char k; // al
  unsigned int v15; // r11d
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // r9
  __int64 v19; // rbx
  int v20; // eax
  unsigned __int64 v22; // rcx
  __int64 i; // rsi
  unsigned int j; // edx
  __int64 v25; // rax
  int v26; // eax
  char v27; // al
  char v28; // al
  __int64 v29; // rax
  int v30; // r8d
  unsigned __int64 v31; // r10
  int v32; // r15d
  unsigned __int64 v33; // rbx
  __int64 m; // rsi
  unsigned int n; // edx
  __int64 v36; // rax
  int v37; // ecx
  int v38; // r9d
  int v39; // ecx
  unsigned __int64 v40; // r15
  int v41; // r12d
  unsigned int ii; // ecx
  __int64 v43; // rax
  int v44; // ecx
  int v45; // r9d
  int v46; // ecx
  unsigned int v47; // eax
  __int64 v48; // [rsp+0h] [rbp-60h]
  __int64 v49; // [rsp+10h] [rbp-50h]
  unsigned __int64 v50; // [rsp+10h] [rbp-50h]
  __int64 v51; // [rsp+10h] [rbp-50h]
  __int64 v52; // [rsp+18h] [rbp-48h]
  int v53; // [rsp+18h] [rbp-48h]
  int v54; // [rsp+18h] [rbp-48h]
  unsigned int v55[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v8 = (int)a3;
  v9 = a4;
  v10 = a6;
  v55[0] = 1;
  if ( ((unsigned __int8)(1 << (((_BYTE)a4 - a5) & 7)) & *(_BYTE *)(a5 + -(((unsigned int)((_DWORD)a4 - a5) >> 3) + 10))) != 0 )
  {
    v11 = -(((unsigned int)(a6 - a7) >> 3) + 10);
    *(_BYTE *)(a7 + v11) |= 1 << ((a6 - a7) & 7);
    v12 = *(_BYTE *)(a2 + 140);
    if ( v12 > 0xAu )
    {
      if ( v12 == 11 )
      {
        v22 = *a4;
        if ( *v9 )
        {
          for ( i = *(_QWORD *)(v22 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          for ( j = qword_4F08388 & (v22 >> 3); ; j = qword_4F08388 & (j + 1) )
          {
            v25 = qword_4F08380 + 16LL * j;
            if ( v22 == *(_QWORD *)v25 )
              break;
            if ( !*(_QWORD *)v25 )
              goto LABEL_23;
          }
          v26 = *(_DWORD *)(v25 + 8);
          LODWORD(v10) = v26 + a6;
          LODWORD(v9) = v26 + (_DWORD)v9;
LABEL_23:
          if ( !(unsigned int)sub_778A80(a1, i, v8, (_DWORD)v9, a5, v10, a7) )
            return 0;
          return v55[0];
        }
      }
      return 1;
    }
    if ( v12 <= 8u )
    {
      if ( v12 == 8 )
      {
        v13 = *(_QWORD *)(a2 + 160);
        for ( k = *(_BYTE *)(v13 + 140); k == 12; k = *(_BYTE *)(v13 + 140) )
          v13 = *(_QWORD *)(v13 + 160);
        v15 = 16;
        if ( (unsigned __int8)(k - 2) > 1u )
        {
          v51 = a5;
          v47 = sub_7764B0(a1, v13, v55);
          a5 = v51;
          v15 = v47;
        }
        v16 = *(_QWORD *)(a2 + 176);
        v17 = v15;
        v48 = v16;
        if ( v16 )
        {
          v18 = v10;
          v19 = 0;
          while ( 1 )
          {
            v49 = v18;
            v52 = a5;
            v20 = sub_778A80(a1, v13, v8, (_DWORD)v9, a5, v18, a7);
            a5 = v52;
            if ( !v20 )
              break;
            LODWORD(v9) = v17 + (_DWORD)v9;
            v18 = v17 + v49;
            if ( v48 == ++v19 )
              return v55[0];
          }
          return 0;
        }
        return v55[0];
      }
      return 1;
    }
    goto LABEL_31;
  }
  v27 = *(_BYTE *)(a2 + 140);
  if ( v27 == 11 )
  {
    if ( (*(_BYTE *)(a2 + 179) & 1) != 0 )
      return 1;
LABEL_56:
    v55[0] = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
      return 0;
    sub_6855B0(0xABFu, a3, (_QWORD *)(a1 + 96));
    sub_770D30(a1);
    return v55[0];
  }
  if ( (unsigned __int8)(v27 - 9) > 1u )
    goto LABEL_56;
  v28 = *(_BYTE *)(a2 + 179);
  if ( (v28 & 2) == 0 && (dword_4F077C4 != 2 || unk_4F07778 <= 202001) )
    goto LABEL_56;
  if ( (v28 & 1) != 0 )
    return 1;
LABEL_31:
  v29 = sub_76FF70(*(_QWORD *)(a2 + 160));
  if ( v29 )
  {
    v50 = v31;
    v32 = v30;
    v53 = v10;
    v33 = v29;
    while ( 1 )
    {
      for ( m = *(_QWORD *)(v33 + 120); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
        ;
      for ( n = qword_4F08388 & (v33 >> 3); ; n = qword_4F08388 & (n + 1) )
      {
        v36 = qword_4F08380 + 16LL * n;
        if ( *(_QWORD *)v36 == v33 )
        {
          v37 = *(_DWORD *)(v36 + 8);
          v38 = v53 + v37;
          v39 = (_DWORD)v9 + v37;
          goto LABEL_40;
        }
        if ( !*(_QWORD *)v36 )
          break;
      }
      v38 = v53;
      v39 = (int)v9;
LABEL_40:
      if ( !(unsigned int)sub_778A80(a1, m, v8, v39, v32, v38, a7) )
        break;
      v33 = sub_76FF70(*(_QWORD *)(v33 + 112));
      if ( !v33 )
      {
        v31 = v50;
        LODWORD(v10) = v53;
        v30 = v32;
        goto LABEL_43;
      }
    }
    v55[0] = 0;
    v31 = v50;
    v30 = v32;
    LODWORD(v10) = v53;
  }
LABEL_43:
  if ( !v31 )
    return v55[0];
  v54 = (int)v9;
  v40 = v31;
  v41 = v30;
  while ( 1 )
  {
    if ( (*(_BYTE *)(v40 + 96) & 3) != 0 )
    {
      for ( ii = qword_4F08388 & (v40 >> 3); ; ii = qword_4F08388 & (ii + 1) )
      {
        v43 = qword_4F08380 + 16LL * ii;
        if ( v40 == *(_QWORD *)v43 )
        {
          v44 = *(_DWORD *)(v43 + 8);
          v45 = v10 + v44;
          v46 = v54 + v44;
          goto LABEL_52;
        }
        if ( !*(_QWORD *)v43 )
          break;
      }
      v46 = v54;
      v45 = v10;
LABEL_52:
      if ( !(unsigned int)sub_778A80(a1, *(_QWORD *)(v40 + 40), v8, v46, v41, v45, a7) )
        return 0;
    }
    v40 = *(_QWORD *)v40;
    if ( !v40 )
      return v55[0];
  }
}
