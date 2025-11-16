// Function: sub_22EC110
// Address: 0x22ec110
//
__int64 __fastcall sub_22EC110(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v8; // r13
  unsigned int v9; // r8d
  __int64 v10; // rdi
  int v11; // r10d
  unsigned int v12; // r15d
  unsigned int v13; // ecx
  __int64 *v14; // rsi
  __int64 *v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // r13
  const char *v18; // rax
  __int64 v19; // rdx
  int v21; // esi
  int v22; // ecx
  int v23; // eax
  int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // edx
  __int64 v27; // rdi
  int v28; // r10d
  __int64 *v29; // r9
  int v30; // eax
  int v31; // edx
  __int64 v32; // rdi
  __int64 *v33; // r8
  unsigned int v34; // r15d
  int v35; // r9d
  __int64 v36; // rsi

  v8 = *(_QWORD *)(a1 + 16) + 32LL * *(unsigned int *)(a1 + 24) - 32;
  v9 = *(_DWORD *)(v8 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)v8;
    goto LABEL_19;
  }
  v10 = *(_QWORD *)(v8 + 8);
  v11 = 1;
  v12 = ((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4);
  v13 = (v9 - 1) & v12;
  v14 = (__int64 *)(v10 + 72LL * v13);
  v15 = 0;
  v16 = *v14;
  if ( *v14 == a4 )
  {
LABEL_3:
    v17 = v14 + 1;
    goto LABEL_4;
  }
  while ( v16 != -4096 )
  {
    if ( v16 == -8192 && !v15 )
      v15 = v14;
    v13 = (v9 - 1) & (v11 + v13);
    v14 = (__int64 *)(v10 + 72LL * v13);
    v16 = *v14;
    if ( *v14 == a4 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v15 )
    v15 = v14;
  v21 = *(_DWORD *)(v8 + 16);
  ++*(_QWORD *)v8;
  v22 = v21 + 1;
  if ( 4 * (v21 + 1) >= 3 * v9 )
  {
LABEL_19:
    sub_22EBA60(v8, 2 * v9);
    v23 = *(_DWORD *)(v8 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(v8 + 8);
      v26 = (v23 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v22 = *(_DWORD *)(v8 + 16) + 1;
      v15 = (__int64 *)(v25 + 72LL * v26);
      v27 = *v15;
      if ( *v15 != a4 )
      {
        v28 = 1;
        v29 = 0;
        while ( v27 != -4096 )
        {
          if ( !v29 && v27 == -8192 )
            v29 = v15;
          v26 = v24 & (v28 + v26);
          v15 = (__int64 *)(v25 + 72LL * v26);
          v27 = *v15;
          if ( *v15 == a4 )
            goto LABEL_15;
          ++v28;
        }
        if ( v29 )
          v15 = v29;
      }
      goto LABEL_15;
    }
    goto LABEL_42;
  }
  if ( v9 - *(_DWORD *)(v8 + 20) - v22 <= v9 >> 3 )
  {
    sub_22EBA60(v8, v9);
    v30 = *(_DWORD *)(v8 + 24);
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = *(_QWORD *)(v8 + 8);
      v33 = 0;
      v34 = (v30 - 1) & v12;
      v35 = 1;
      v22 = *(_DWORD *)(v8 + 16) + 1;
      v15 = (__int64 *)(v32 + 72LL * v34);
      v36 = *v15;
      if ( *v15 != a4 )
      {
        while ( v36 != -4096 )
        {
          if ( v36 == -8192 && !v33 )
            v33 = v15;
          v34 = v31 & (v35 + v34);
          v15 = (__int64 *)(v32 + 72LL * v34);
          v36 = *v15;
          if ( *v15 == a4 )
            goto LABEL_15;
          ++v35;
        }
        if ( v33 )
          v15 = v33;
      }
      goto LABEL_15;
    }
LABEL_42:
    ++*(_DWORD *)(v8 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(v8 + 16) = v22;
  if ( *v15 != -4096 )
    --*(_DWORD *)(v8 + 20);
  *v15 = a4;
  v17 = v15 + 1;
  *(_OWORD *)(v15 + 1) = 0;
  *(_OWORD *)(v15 + 3) = 0;
  *(_OWORD *)(v15 + 5) = 0;
  *(_OWORD *)(v15 + 7) = 0;
LABEL_4:
  v18 = sub_BD5D20(a4);
  *(_QWORD *)(a1 + 152) = a4;
  return sub_3141C60(a1, v17, v18, v19, a5);
}
