// Function: sub_270EC80
// Address: 0x270ec80
//
__int64 *__fastcall sub_270EC80(__int64 a1, __int64 *a2)
{
  __int64 v3; // r13
  __int64 *v5; // r14
  __int64 v6; // rcx
  unsigned __int64 v7; // rsi
  int v8; // r9d
  unsigned int v9; // eax
  __int64 *v10; // rdx
  __int64 *v11; // r8
  __int64 v12; // rdi
  int v14; // eax
  int v15; // ecx
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rsi
  __int64 v20; // r13
  int v21; // edx
  int v22; // edi
  unsigned int v23; // eax
  __int64 v24; // r9
  int v25; // r11d
  __int64 *v26; // r10
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rsi
  __int64 v31; // r13
  int v32; // edx
  int v33; // edi
  int v34; // r11d
  unsigned int v35; // eax
  __int64 v36; // r9
  __int64 v37; // rax
  _QWORD *v38; // rcx
  _QWORD *v39; // rax
  __int64 v40; // rax
  _QWORD *v41; // rcx
  _QWORD *v42; // rax

  v3 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
  if ( !(_DWORD)v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v6 = *a2;
  v7 = (unsigned int)(v3 - 1);
  v8 = 1;
  v9 = v7 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v10 = 0;
  v11 = &v5[24 * v9];
  v12 = *v11;
  if ( v6 == *v11 )
    return v11 + 1;
  while ( v12 != -4096 )
  {
    if ( v12 == -8192 && !v10 )
      v10 = v11;
    v9 = v7 & (v8 + v9);
    v11 = &v5[24 * v9];
    v12 = *v11;
    if ( v6 == *v11 )
      return v11 + 1;
    ++v8;
  }
  v14 = *(_DWORD *)(a1 + 16);
  if ( !v10 )
    v10 = v11;
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= (unsigned int)(3 * v3) )
  {
LABEL_18:
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
    v18 = sub_C7D670(192LL * (unsigned int)v17, 8);
    *(_QWORD *)(a1 + 8) = v18;
    v19 = (_QWORD *)v18;
    if ( v5 )
    {
      v20 = 24 * v3;
      sub_270E700(a1, v5, &v5[v20]);
      sub_C7D6A0((__int64)v5, v20 * 8, 8);
      v19 = *(_QWORD **)(a1 + 8);
      v21 = *(_DWORD *)(a1 + 24);
      v15 = *(_DWORD *)(a1 + 16) + 1;
    }
    else
    {
      v37 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      v21 = v37;
      v38 = &v19[24 * v37];
      if ( v19 != v38 )
      {
        v39 = v19;
        do
        {
          if ( v39 )
            *v39 = -4096;
          v39 += 24;
        }
        while ( v38 != v39 );
      }
      v15 = 1;
    }
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = (v21 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v10 = &v19[24 * v23];
      v24 = *v10;
      if ( *v10 == *a2 )
        goto LABEL_14;
      v25 = 1;
      v26 = 0;
      while ( v24 != -4096 )
      {
        if ( !v26 && v24 == -8192 )
          v26 = v10;
        v23 = v22 & (v25 + v23);
        v10 = &v19[24 * v23];
        v24 = *v10;
        if ( *a2 == *v10 )
          goto LABEL_14;
        ++v25;
      }
LABEL_26:
      if ( v26 )
        v10 = v26;
      goto LABEL_14;
    }
LABEL_58:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( (int)v3 - *(_DWORD *)(a1 + 20) - v15 <= (unsigned int)v3 >> 3 )
  {
    v27 = (((((v7 >> 1) | v7 | (((v7 >> 1) | v7) >> 2)) >> 4) | (v7 >> 1) | v7 | (((v7 >> 1) | v7) >> 2)) >> 8)
        | (((v7 >> 1) | v7 | (((v7 >> 1) | v7) >> 2)) >> 4)
        | (v7 >> 1)
        | v7
        | (((v7 >> 1) | v7) >> 2);
    v28 = ((v27 >> 16) | v27) + 1;
    if ( (unsigned int)v28 < 0x40 )
      LODWORD(v28) = 64;
    *(_DWORD *)(a1 + 24) = v28;
    v29 = sub_C7D670(192LL * (unsigned int)v28, 8);
    *(_QWORD *)(a1 + 8) = v29;
    v30 = (_QWORD *)v29;
    if ( v5 )
    {
      v31 = 24 * v3;
      sub_270E700(a1, v5, &v5[v31]);
      sub_C7D6A0((__int64)v5, v31 * 8, 8);
      v30 = *(_QWORD **)(a1 + 8);
      v32 = *(_DWORD *)(a1 + 24);
      v15 = *(_DWORD *)(a1 + 16) + 1;
    }
    else
    {
      v40 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      v32 = v40;
      v41 = &v30[24 * v40];
      if ( v30 != v41 )
      {
        v42 = v30;
        do
        {
          if ( v42 )
            *v42 = -4096;
          v42 += 24;
        }
        while ( v41 != v42 );
      }
      v15 = 1;
    }
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = 1;
      v26 = 0;
      v35 = (v32 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v10 = &v30[24 * v35];
      v36 = *v10;
      if ( *a2 == *v10 )
        goto LABEL_14;
      while ( v36 != -4096 )
      {
        if ( !v26 && v36 == -8192 )
          v26 = v10;
        v35 = v33 & (v34 + v35);
        v10 = &v30[24 * v35];
        v36 = *v10;
        if ( *a2 == *v10 )
          goto LABEL_14;
        ++v34;
      }
      goto LABEL_26;
    }
    goto LABEL_58;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v10 = *a2;
  memset(v10 + 1, 0, 0xB8u);
  v10[16] = (__int64)(v10 + 18);
  v10[17] = 0x200000000LL;
  v10[21] = 0x200000000LL;
  v10[20] = (__int64)(v10 + 22);
  return v10 + 1;
}
