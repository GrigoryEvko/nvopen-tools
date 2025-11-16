// Function: sub_18CDCF0
// Address: 0x18cdcf0
//
__int64 *__fastcall sub_18CDCF0(__int64 a1, __int64 *a2)
{
  __int64 v3; // r13
  __int64 *v5; // r14
  __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  unsigned int v8; // eax
  __int64 *v9; // r8
  __int64 v10; // rcx
  int v12; // r9d
  __int64 *v13; // rdi
  int v14; // eax
  int v15; // edx
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rcx
  int v20; // esi
  int v21; // esi
  unsigned int v22; // eax
  __int64 v23; // r9
  int v24; // r11d
  __int64 *v25; // r10
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rcx
  int v30; // esi
  int v31; // esi
  int v32; // r11d
  unsigned int v33; // eax
  __int64 v34; // r9
  __int64 v35; // rax
  _QWORD *v36; // rdx
  _QWORD *v37; // rax
  __int64 v38; // rax
  _QWORD *v39; // rdx
  _QWORD *v40; // rax

  v3 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
  if ( !(_DWORD)v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_14;
  }
  v6 = *a2;
  v7 = (unsigned int)(v3 - 1);
  v8 = v7 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v9 = &v5[24 * v8];
  v10 = *v9;
  if ( v6 == *v9 )
    return v9;
  v12 = 1;
  v13 = 0;
  while ( v10 != -8 )
  {
    if ( v10 == -16 && !v13 )
      v13 = v9;
    v8 = v7 & (v12 + v8);
    v9 = &v5[24 * v8];
    v10 = *v9;
    if ( v6 == *v9 )
      return v9;
    ++v12;
  }
  v14 = *(_DWORD *)(a1 + 16);
  if ( v13 )
    v9 = v13;
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= (unsigned int)(3 * v3) )
  {
LABEL_14:
    v16 = (((((((unsigned int)(2 * v3 - 1) | ((unsigned __int64)(unsigned int)(2 * v3 - 1) >> 1)) >> 2)
            | (unsigned int)(2 * v3 - 1)
            | ((unsigned __int64)(unsigned int)(2 * v3 - 1) >> 1)) >> 4)
          | (((unsigned int)(2 * v3 - 1) | ((unsigned __int64)(unsigned int)(2 * v3 - 1) >> 1)) >> 2)
          | (unsigned int)(2 * v3 - 1)
          | ((unsigned __int64)(unsigned int)(2 * v3 - 1) >> 1)) >> 8)
        | (((((unsigned int)(2 * v3 - 1) | ((unsigned __int64)(unsigned int)(2 * v3 - 1) >> 1)) >> 2)
          | (unsigned int)(2 * v3 - 1)
          | ((unsigned __int64)(unsigned int)(2 * v3 - 1) >> 1)) >> 4)
        | (((unsigned int)(2 * v3 - 1) | ((unsigned __int64)(unsigned int)(2 * v3 - 1) >> 1)) >> 2)
        | (unsigned int)(2 * v3 - 1)
        | ((unsigned __int64)(unsigned int)(2 * v3 - 1) >> 1);
    v17 = ((v16 >> 16) | v16) + 1;
    if ( (unsigned int)v17 < 0x40 )
      LODWORD(v17) = 64;
    *(_DWORD *)(a1 + 24) = v17;
    v18 = sub_22077B0(192LL * (unsigned int)v17);
    *(_QWORD *)(a1 + 8) = v18;
    v19 = (_QWORD *)v18;
    if ( v5 )
    {
      sub_18CD8E0(a1, v5, &v5[24 * v3]);
      j___libc_free_0(v5);
      v19 = *(_QWORD **)(a1 + 8);
      v20 = *(_DWORD *)(a1 + 24);
      v15 = *(_DWORD *)(a1 + 16) + 1;
    }
    else
    {
      v35 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      v20 = v35;
      v36 = &v19[24 * v35];
      if ( v19 != v36 )
      {
        v37 = v19;
        do
        {
          if ( v37 )
            *v37 = -8;
          v37 += 24;
        }
        while ( v36 != v37 );
      }
      v15 = 1;
    }
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = v21 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v9 = &v19[24 * v22];
      v23 = *v9;
      if ( *v9 == *a2 )
        goto LABEL_10;
      v24 = 1;
      v25 = 0;
      while ( v23 != -8 )
      {
        if ( !v25 && v23 == -16 )
          v25 = v9;
        v22 = v21 & (v24 + v22);
        v9 = &v19[24 * v22];
        v23 = *v9;
        if ( *a2 == *v9 )
          goto LABEL_10;
        ++v24;
      }
LABEL_22:
      if ( v25 )
        v9 = v25;
      goto LABEL_10;
    }
LABEL_59:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( (int)v3 - *(_DWORD *)(a1 + 20) - v15 <= (unsigned int)v3 >> 3 )
  {
    v26 = (((v7 >> 1) | v7 | (((v7 >> 1) | v7) >> 2)) >> 4) | (v7 >> 1) | v7 | (((v7 >> 1) | v7) >> 2);
    v27 = ((((v26 >> 8) | v26) >> 16) | (v26 >> 8) | v26) + 1;
    if ( (unsigned int)v27 < 0x40 )
      LODWORD(v27) = 64;
    *(_DWORD *)(a1 + 24) = v27;
    v28 = sub_22077B0(192LL * (unsigned int)v27);
    *(_QWORD *)(a1 + 8) = v28;
    v29 = (_QWORD *)v28;
    if ( v5 )
    {
      sub_18CD8E0(a1, v5, &v5[24 * v3]);
      j___libc_free_0(v5);
      v29 = *(_QWORD **)(a1 + 8);
      v30 = *(_DWORD *)(a1 + 24);
      v15 = *(_DWORD *)(a1 + 16) + 1;
    }
    else
    {
      v38 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      v30 = v38;
      v39 = &v29[24 * v38];
      if ( v29 != v39 )
      {
        v40 = v29;
        do
        {
          if ( v40 )
            *v40 = -8;
          v40 += 24;
        }
        while ( v39 != v40 );
      }
      v15 = 1;
    }
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = 1;
      v25 = 0;
      v33 = v31 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v9 = &v29[24 * v33];
      v34 = *v9;
      if ( *v9 == *a2 )
        goto LABEL_10;
      while ( v34 != -8 )
      {
        if ( v34 == -16 && !v25 )
          v25 = v9;
        v33 = v31 & (v32 + v33);
        v9 = &v29[24 * v33];
        v34 = *v9;
        if ( *a2 == *v9 )
          goto LABEL_10;
        ++v32;
      }
      goto LABEL_22;
    }
    goto LABEL_59;
  }
LABEL_10:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *v9 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v9 = *a2;
  memset(v9 + 1, 0, 0xB8u);
  v9[20] = (__int64)(v9 + 22);
  v9[16] = (__int64)(v9 + 18);
  v9[17] = 0x200000000LL;
  v9[21] = 0x200000000LL;
  return v9;
}
