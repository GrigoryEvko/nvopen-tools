// Function: sub_2FADE80
// Address: 0x2fade80
//
char __fastcall sub_2FADE80(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4)
{
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r13
  int v7; // r8d
  __int64 i; // r13
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rax
  __int64 j; // r8
  __int16 v12; // dx
  unsigned int v13; // edi
  __int64 v14; // r8
  unsigned int v15; // r11d
  __int64 *v16; // rdx
  __int64 v17; // r9
  __int64 v18; // r8
  __int64 v19; // r15
  unsigned __int64 k; // rax
  __int64 m; // rsi
  __int16 v22; // dx
  unsigned int v23; // ecx
  __int64 v24; // rsi
  unsigned int v25; // r10d
  __int64 *v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // r12
  unsigned __int64 v29; // r15
  _QWORD *v30; // r12
  int v31; // ecx
  __int64 v32; // r9
  unsigned __int64 v33; // rsi
  unsigned __int64 v34; // rdi
  __int64 n; // rbx
  int v36; // edi
  __int64 v37; // r10
  int v38; // edi
  int v39; // r11d
  unsigned int v40; // edx
  unsigned __int64 v41; // rdi
  __int64 ii; // rbx
  int v43; // edx
  int v44; // r9d
  int v45; // edx
  int v46; // r10d
  bool v48; // [rsp+Fh] [rbp-61h]
  __int64 *v49; // [rsp+10h] [rbp-60h]
  __int64 v51; // [rsp+20h] [rbp-50h]
  __int64 v52; // [rsp+28h] [rbp-48h]
  unsigned __int8 v53; // [rsp+28h] [rbp-48h]
  unsigned __int8 v54; // [rsp+30h] [rbp-40h]
  unsigned __int64 v55; // [rsp+30h] [rbp-40h]
  __int64 v56; // [rsp+38h] [rbp-38h]
  unsigned __int8 v57; // [rsp+38h] [rbp-38h]

  v5 = a4;
  v6 = *(_QWORD *)(a2 + 56);
  v49 = (__int64 *)v6;
  v48 = v6 == (_QWORD)a3;
  if ( (__int64 *)v6 != a3 )
  {
    v56 = *a3;
    v6 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v6 )
      BUG();
    v7 = *(_DWORD *)(v6 + 44);
    if ( (*(_QWORD *)v6 & 4) != 0 )
    {
      v9 = v56 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = v56 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v7 & 4) != 0 )
      {
        do
          v10 = *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v10 + 44) & 4) != 0 );
      }
    }
    else
    {
      if ( (v7 & 4) != 0 )
      {
        for ( i = *(_QWORD *)v6; ; i = *(_QWORD *)v6 )
        {
          v6 = i & 0xFFFFFFFFFFFFFFF8LL;
          v7 = *(_DWORD *)(v6 + 44) & 0xFFFFFF;
          if ( (*(_DWORD *)(v6 + 44) & 4) == 0 )
            break;
        }
      }
      v9 = v6;
      v10 = v6;
    }
    if ( (v7 & 8) != 0 )
    {
      do
        v9 = *(_QWORD *)(v9 + 8);
      while ( (*(_BYTE *)(v9 + 44) & 8) != 0 );
    }
    for ( j = *(_QWORD *)(v9 + 8); j != v10; v10 = *(_QWORD *)(v10 + 8) )
    {
      v12 = *(_WORD *)(v10 + 68);
      if ( (unsigned __int16)(v12 - 14) > 4u && v12 != 24 )
        break;
    }
    v13 = *(_DWORD *)(a1 + 144);
    v14 = *(_QWORD *)(a1 + 128);
    if ( v13 )
    {
      v15 = (v13 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( v10 == *v16 )
        goto LABEL_17;
      v45 = 1;
      while ( v17 != -4096 )
      {
        v46 = v45 + 1;
        v15 = (v13 - 1) & (v45 + v15);
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( *v16 == v10 )
          goto LABEL_17;
        v45 = v46;
      }
    }
    v16 = (__int64 *)(v14 + 16LL * v13);
LABEL_17:
    v18 = a2 + 48;
    v19 = v16[1];
    if ( a4 != a2 + 48 )
      goto LABEL_18;
    goto LABEL_91;
  }
  v18 = a2 + 48;
  v19 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 16LL * *(unsigned int *)(a2 + 24));
  if ( a4 != a2 + 48 )
  {
LABEL_18:
    for ( k = a4; (*(_BYTE *)(k + 44) & 4) != 0; k = *(_QWORD *)k & 0xFFFFFFFFFFFFFFF8LL )
      ;
    for ( ; (*(_BYTE *)(a4 + 44) & 8) != 0; a4 = *(_QWORD *)(a4 + 8) )
      ;
    for ( m = *(_QWORD *)(a4 + 8); m != k; k = *(_QWORD *)(k + 8) )
    {
      v22 = *(_WORD *)(k + 68);
      if ( (unsigned __int16)(v22 - 14) > 4u && v22 != 24 )
        break;
    }
    v23 = *(_DWORD *)(a1 + 144);
    v24 = *(_QWORD *)(a1 + 128);
    if ( v23 )
    {
      v25 = (v23 - 1) & (((unsigned int)k >> 9) ^ ((unsigned int)k >> 4));
      v26 = (__int64 *)(v24 + 16LL * v25);
      v27 = *v26;
      if ( k == *v26 )
      {
LABEL_28:
        v28 = v26[1];
        goto LABEL_29;
      }
      v43 = 1;
      while ( v27 != -4096 )
      {
        v44 = v43 + 1;
        v25 = (v23 - 1) & (v43 + v25);
        v26 = (__int64 *)(v24 + 16LL * v25);
        v27 = *v26;
        if ( k == *v26 )
          goto LABEL_28;
        v43 = v44;
      }
    }
    v26 = (__int64 *)(v24 + 16LL * v23);
    goto LABEL_28;
  }
LABEL_91:
  k = *(_QWORD *)(a1 + 152) + 16LL * *(unsigned int *)(a2 + 24);
  v28 = *(_QWORD *)(k + 8);
LABEL_29:
  v57 = 0;
  v29 = v19 & 0xFFFFFFFFFFFFFFF8LL;
  v30 = (_QWORD *)(v28 & 0xFFFFFFFFFFFFFFF8LL);
  v31 = 0;
LABEL_30:
  if ( v30 != (_QWORD *)v29 )
  {
LABEL_31:
    v32 = v30[2];
    if ( v18 != v5 )
      goto LABEL_32;
    if ( v18 == v6 )
    {
      LOBYTE(k) = v49 != a3;
      v33 = 0;
      LODWORD(k) = (v31 | k) ^ 1;
      goto LABEL_34;
    }
    goto LABEL_50;
  }
  while ( v5 != v6 )
  {
    v32 = v30[2];
    if ( v18 != v5 )
    {
LABEL_32:
      if ( (_BYTE)v31 )
      {
        LOBYTE(k) = v5 != v6;
        v33 = 0;
        goto LABEL_34;
      }
      if ( v5 == v6 )
      {
LABEL_83:
        LOBYTE(k) = v49 != a3;
        LODWORD(k) = (v31 | k) ^ 1;
      }
      else
      {
        LODWORD(k) = 1;
      }
      if ( v5 )
      {
        v36 = *(_DWORD *)(a1 + 144);
        v37 = *(_QWORD *)(a1 + 128);
        if ( !v36 )
          goto LABEL_65;
        v38 = v36 - 1;
        v39 = 1;
        v40 = v38 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v33 = *(_QWORD *)(v37 + 16LL * v40);
        if ( v5 != v33 )
        {
          while ( v33 != -4096 )
          {
            v40 = v38 & (v39 + v40);
            v33 = *(_QWORD *)(v37 + 16LL * v40);
            if ( v33 == v5 )
              goto LABEL_34;
            ++v39;
          }
LABEL_65:
          LOBYTE(k) = (v5 == v32) & k;
          if ( (_BYTE)k )
          {
            v33 = v5;
            v30 = (_QWORD *)(*v30 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v5 != v6 )
              goto LABEL_37;
            v31 = k;
            goto LABEL_44;
          }
          v33 = v5;
          LODWORD(k) = 1;
          if ( v5 == v6 )
          {
            v31 = 1;
            goto LABEL_44;
          }
          goto LABEL_72;
        }
        goto LABEL_34;
      }
      goto LABEL_86;
    }
LABEL_50:
    v33 = 0;
    LODWORD(k) = 1;
    LOBYTE(k) = v32 == 0;
    if ( !v32 )
    {
LABEL_35:
      v30 = (_QWORD *)(*v30 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v5 != v6 )
      {
        LODWORD(k) = 0;
LABEL_37:
        v34 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v34 )
          BUG();
        v5 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v34 & 4) == 0 && (*(_BYTE *)(v34 + 44) & 4) != 0 )
        {
          for ( n = *(_QWORD *)v34; ; n = *(_QWORD *)v5 )
          {
            v5 = n & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v5 + 44) & 4) == 0 )
              break;
          }
        }
        goto LABEL_43;
      }
      v31 = k;
      goto LABEL_30;
    }
LABEL_51:
    if ( !v57 )
    {
      if ( v30 == (_QWORD *)v29 )
        v57 = 1;
      else
        v30 = (_QWORD *)(*v30 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v32 )
      {
        v51 = v18;
        v53 = v31;
        v55 = v32;
        sub_2FAD510(a1, v32);
        v31 = v53;
        v18 = v51;
        v33 = v55;
        goto LABEL_44;
      }
      goto LABEL_30;
    }
    if ( v5 == v6 )
    {
      v31 = v57;
      goto LABEL_30;
    }
LABEL_72:
    v41 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v41 )
      BUG();
    v5 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_QWORD *)v41 & 4) == 0 && (*(_BYTE *)(v41 + 44) & 4) != 0 )
    {
      for ( ii = *(_QWORD *)v41; ; ii = *(_QWORD *)v5 )
      {
        v5 = ii & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v5 + 44) & 4) == 0 )
          break;
      }
    }
LABEL_43:
    if ( !(_BYTE)k )
      goto LABEL_30;
LABEL_44:
    if ( !*(_QWORD *)(v33 + 24) )
      goto LABEL_30;
    LODWORD(k) = *(unsigned __int16 *)(v33 + 68);
    if ( (unsigned __int16)(k - 14) <= 4u || (_WORD)k == 24 )
      goto LABEL_30;
    v52 = v18;
    v54 = v31;
    k = sub_2E192D0(a1, v33, 0);
    v31 = v54;
    v18 = v52;
    if ( v30 != (_QWORD *)v29 )
      goto LABEL_31;
  }
  LODWORD(k) = v31 ^ 1;
  LOBYTE(k) = v48 & (v31 ^ 1);
  if ( (_BYTE)k )
  {
    v32 = v30[2];
    if ( v18 != v5 )
      goto LABEL_83;
LABEL_86:
    v33 = 0;
LABEL_34:
    LOBYTE(k) = (v32 == v33) & k;
    if ( (_BYTE)k )
      goto LABEL_35;
    goto LABEL_51;
  }
  return k;
}
