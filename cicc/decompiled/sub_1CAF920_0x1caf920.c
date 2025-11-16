// Function: sub_1CAF920
// Address: 0x1caf920
//
__int64 __fastcall sub_1CAF920(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  int v6; // r10d
  __int64 *v7; // rdx
  unsigned int v8; // r13d
  unsigned int v9; // edi
  __int64 *v10; // rax
  __int64 v11; // rcx
  unsigned int v12; // r13d
  int v14; // eax
  int v15; // ecx
  __int64 v16; // rax
  unsigned __int8 v17; // cl
  unsigned int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // ecx
  __int64 *v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rdi
  int v24; // eax
  int v25; // esi
  __int64 v26; // r8
  unsigned int v27; // eax
  __int64 v28; // rdi
  int v29; // r10d
  __int64 *v30; // r9
  int v31; // eax
  int v32; // eax
  __int64 v33; // rdi
  __int64 *v34; // r8
  unsigned int v35; // r13d
  int v36; // r9d
  __int64 v37; // rsi
  int v38; // r11d
  __int64 *v39; // r10
  int v40; // ecx
  int v41; // edi
  int v42; // r10d
  int v43; // r10d
  __int64 v44; // r11
  unsigned int v45; // ecx
  __int64 v46; // r9
  int v47; // r8d
  __int64 *v48; // rsi
  int v49; // r10d
  int v50; // r10d
  __int64 v51; // r11
  int v52; // r8d
  unsigned int v53; // ecx
  __int64 v54; // r9
  __int64 v55[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a2 + 24);
  v55[0] = a1;
  if ( v4 )
  {
    v5 = *(_QWORD *)(a2 + 8);
    v6 = 1;
    v7 = 0;
    v8 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
    v9 = (v4 - 1) & v8;
    v10 = (__int64 *)(v5 + 16LL * v9);
    v11 = *v10;
    if ( a1 == *v10 )
      return *((unsigned __int8 *)v10 + 8);
    while ( v11 != -8 )
    {
      if ( v11 == -16 && !v7 )
        v7 = v10;
      v9 = (v4 - 1) & (v6 + v9);
      v10 = (__int64 *)(v5 + 16LL * v9);
      v11 = *v10;
      if ( a1 == *v10 )
        return *((unsigned __int8 *)v10 + 8);
      ++v6;
    }
    if ( !v7 )
      v7 = v10;
    v14 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a2 + 20) - v15 > v4 >> 3 )
        goto LABEL_15;
      sub_1CAF520(a2, v4);
      v31 = *(_DWORD *)(a2 + 24);
      if ( v31 )
      {
        v32 = v31 - 1;
        v33 = *(_QWORD *)(a2 + 8);
        v34 = 0;
        v35 = v32 & v8;
        v36 = 1;
        v15 = *(_DWORD *)(a2 + 16) + 1;
        v7 = (__int64 *)(v33 + 16LL * v35);
        v37 = *v7;
        if ( a1 != *v7 )
        {
          while ( v37 != -8 )
          {
            if ( v37 == -16 && !v34 )
              v34 = v7;
            v35 = v32 & (v36 + v35);
            v7 = (__int64 *)(v33 + 16LL * v35);
            v37 = *v7;
            if ( a1 == *v7 )
              goto LABEL_15;
            ++v36;
          }
          if ( v34 )
            v7 = v34;
        }
        goto LABEL_15;
      }
LABEL_88:
      ++*(_DWORD *)(a2 + 16);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)a2;
  }
  sub_1CAF520(a2, 2 * v4);
  v24 = *(_DWORD *)(a2 + 24);
  if ( !v24 )
    goto LABEL_88;
  v25 = v24 - 1;
  v26 = *(_QWORD *)(a2 + 8);
  v27 = (v24 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v15 = *(_DWORD *)(a2 + 16) + 1;
  v7 = (__int64 *)(v26 + 16LL * v27);
  v28 = *v7;
  if ( a1 != *v7 )
  {
    v29 = 1;
    v30 = 0;
    while ( v28 != -8 )
    {
      if ( !v30 && v28 == -16 )
        v30 = v7;
      v27 = v25 & (v29 + v27);
      v7 = (__int64 *)(v26 + 16LL * v27);
      v28 = *v7;
      if ( a1 == *v7 )
        goto LABEL_15;
      ++v29;
    }
    if ( v30 )
      v7 = v30;
  }
LABEL_15:
  *(_DWORD *)(a2 + 16) = v15;
  if ( *v7 != -8 )
    --*(_DWORD *)(a2 + 20);
  *v7 = a1;
  v16 = v55[0];
  *((_BYTE *)v7 + 8) = 0;
  v17 = *(_BYTE *)(v16 + 16);
  if ( v17 > 0x11u )
  {
    if ( v17 == 71 || v17 == 72 )
    {
      v23 = *(_QWORD *)(v16 - 24);
    }
    else
    {
      if ( v17 != 56 )
        goto LABEL_21;
      v23 = *(_QWORD *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF));
    }
    v12 = sub_1CAF920(v23, a2);
    *((_BYTE *)sub_1CAF6E0(a2, v55) + 8) = v12;
    return v12;
  }
  if ( ((0x2002FuLL >> v17) & 1) == 0 )
  {
LABEL_21:
    v18 = *(_DWORD *)(a2 + 24);
    if ( v18 )
    {
      v19 = *(_QWORD *)(a2 + 8);
      v20 = (v18 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v21 = (__int64 *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( v16 == *v21 )
      {
LABEL_23:
        *((_BYTE *)v21 + 8) = 1;
        return 1;
      }
      v38 = 1;
      v39 = 0;
      while ( v22 != -8 )
      {
        if ( v22 == -16 && !v39 )
          v39 = v21;
        v20 = (v18 - 1) & (v38 + v20);
        v21 = (__int64 *)(v19 + 16LL * v20);
        v22 = *v21;
        if ( v16 == *v21 )
          goto LABEL_23;
        ++v38;
      }
      v40 = *(_DWORD *)(a2 + 16);
      if ( v39 )
        v21 = v39;
      ++*(_QWORD *)a2;
      v41 = v40 + 1;
      if ( 4 * (v40 + 1) < 3 * v18 )
      {
        if ( v18 - *(_DWORD *)(a2 + 20) - v41 > v18 >> 3 )
        {
LABEL_49:
          *(_DWORD *)(a2 + 16) = v41;
          if ( *v21 != -8 )
            --*(_DWORD *)(a2 + 20);
          *v21 = v16;
          *((_BYTE *)v21 + 8) = 0;
          goto LABEL_23;
        }
        sub_1CAF520(a2, v18);
        v49 = *(_DWORD *)(a2 + 24);
        if ( v49 )
        {
          v16 = v55[0];
          v50 = v49 - 1;
          v51 = *(_QWORD *)(a2 + 8);
          v48 = 0;
          v52 = 1;
          v41 = *(_DWORD *)(a2 + 16) + 1;
          v53 = v50 & ((LODWORD(v55[0]) >> 9) ^ (LODWORD(v55[0]) >> 4));
          v21 = (__int64 *)(v51 + 16LL * v53);
          v54 = *v21;
          if ( *v21 == v55[0] )
            goto LABEL_49;
          while ( v54 != -8 )
          {
            if ( v54 == -16 && !v48 )
              v48 = v21;
            v53 = v50 & (v52 + v53);
            v21 = (__int64 *)(v51 + 16LL * v53);
            v54 = *v21;
            if ( v55[0] == *v21 )
              goto LABEL_49;
            ++v52;
          }
          goto LABEL_57;
        }
        goto LABEL_89;
      }
    }
    else
    {
      ++*(_QWORD *)a2;
    }
    sub_1CAF520(a2, 2 * v18);
    v42 = *(_DWORD *)(a2 + 24);
    if ( v42 )
    {
      v16 = v55[0];
      v43 = v42 - 1;
      v44 = *(_QWORD *)(a2 + 8);
      v41 = *(_DWORD *)(a2 + 16) + 1;
      v45 = v43 & ((LODWORD(v55[0]) >> 9) ^ (LODWORD(v55[0]) >> 4));
      v21 = (__int64 *)(v44 + 16LL * v45);
      v46 = *v21;
      if ( *v21 == v55[0] )
        goto LABEL_49;
      v47 = 1;
      v48 = 0;
      while ( v46 != -8 )
      {
        if ( !v48 && v46 == -16 )
          v48 = v21;
        v45 = v43 & (v47 + v45);
        v21 = (__int64 *)(v44 + 16LL * v45);
        v46 = *v21;
        if ( v55[0] == *v21 )
          goto LABEL_49;
        ++v47;
      }
LABEL_57:
      if ( v48 )
        v21 = v48;
      goto LABEL_49;
    }
LABEL_89:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  return 0;
}
