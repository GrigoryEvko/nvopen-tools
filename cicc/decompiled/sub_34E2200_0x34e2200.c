// Function: sub_34E2200
// Address: 0x34e2200
//
__int64 __fastcall sub_34E2200(__int64 a1, __int64 *a2, unsigned __int8 a3)
{
  __int64 v6; // r15
  __int64 v7; // r13
  unsigned int v8; // r8d
  __int64 v9; // r9
  int v10; // r11d
  unsigned int v11; // edi
  __int64 *v12; // rsi
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r13
  char *v16; // rax
  size_t v17; // rdx
  int v19; // ecx
  int v20; // esi
  int v21; // eax
  int v22; // ecx
  __int64 v23; // rdi
  unsigned int v24; // edx
  __int64 v25; // r8
  int v26; // r10d
  __int64 *v27; // r9
  int v28; // eax
  int v29; // ecx
  __int64 v30; // r8
  int v31; // r10d
  unsigned int v32; // edx
  __int64 v33; // rdi

  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 16) + 32LL * *(unsigned int *)(a1 + 24) - 32;
  v8 = *(_DWORD *)(v7 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)v7;
    goto LABEL_19;
  }
  v9 = *(_QWORD *)(v7 + 8);
  v10 = 1;
  v11 = (v8 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v12 = (__int64 *)(v9 + 72LL * v11);
  v13 = 0;
  v14 = *v12;
  if ( v6 == *v12 )
  {
LABEL_3:
    v15 = (__int64)(v12 + 1);
    goto LABEL_4;
  }
  while ( v14 != -4096 )
  {
    if ( v14 == -8192 && !v13 )
      v13 = v12;
    v11 = (v8 - 1) & (v10 + v11);
    v12 = (__int64 *)(v9 + 72LL * v11);
    v14 = *v12;
    if ( v6 == *v12 )
      goto LABEL_3;
    ++v10;
  }
  v19 = *(_DWORD *)(v7 + 16);
  if ( !v13 )
    v13 = v12;
  ++*(_QWORD *)v7;
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v8 )
  {
LABEL_19:
    sub_22EBA60(v7, 2 * v8);
    v21 = *(_DWORD *)(v7 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(v7 + 8);
      v24 = (v21 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v20 = *(_DWORD *)(v7 + 16) + 1;
      v13 = (__int64 *)(v23 + 72LL * v24);
      v25 = *v13;
      if ( v6 == *v13 )
        goto LABEL_15;
      v26 = 1;
      v27 = 0;
      while ( v25 != -4096 )
      {
        if ( !v27 && v25 == -8192 )
          v27 = v13;
        v24 = v22 & (v26 + v24);
        v13 = (__int64 *)(v23 + 72LL * v24);
        v25 = *v13;
        if ( v6 == *v13 )
          goto LABEL_15;
        ++v26;
      }
LABEL_23:
      if ( v27 )
        v13 = v27;
      goto LABEL_15;
    }
LABEL_39:
    ++*(_DWORD *)(v7 + 16);
    BUG();
  }
  if ( v8 - *(_DWORD *)(v7 + 20) - v20 <= v8 >> 3 )
  {
    sub_22EBA60(v7, v8);
    v28 = *(_DWORD *)(v7 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(v7 + 8);
      v31 = 1;
      v27 = 0;
      v32 = (v28 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v20 = *(_DWORD *)(v7 + 16) + 1;
      v13 = (__int64 *)(v30 + 72LL * v32);
      v33 = *v13;
      if ( v6 == *v13 )
        goto LABEL_15;
      while ( v33 != -4096 )
      {
        if ( v33 == -8192 && !v27 )
          v27 = v13;
        v32 = v29 & (v31 + v32);
        v13 = (__int64 *)(v30 + 72LL * v32);
        v33 = *v13;
        if ( v6 == *v13 )
          goto LABEL_15;
        ++v31;
      }
      goto LABEL_23;
    }
    goto LABEL_39;
  }
LABEL_15:
  *(_DWORD *)(v7 + 16) = v20;
  if ( *v13 != -4096 )
    --*(_DWORD *)(v7 + 20);
  *v13 = v6;
  v15 = (__int64)(v13 + 1);
  *(_OWORD *)(v13 + 1) = 0;
  *(_OWORD *)(v13 + 3) = 0;
  *(_OWORD *)(v13 + 5) = 0;
  *(_OWORD *)(v13 + 7) = 0;
LABEL_4:
  v16 = (char *)sub_2E791E0(a2);
  *(_QWORD *)(a1 + 152) = a2;
  return sub_3141C60(a1, v15, v16, v17, a3);
}
