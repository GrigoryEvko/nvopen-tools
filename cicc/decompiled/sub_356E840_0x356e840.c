// Function: sub_356E840
// Address: 0x356e840
//
__int64 *__fastcall sub_356E840(__int64 *a1, __int64 a2, int a3)
{
  __int64 v4; // r12
  unsigned int v6; // esi
  __int64 v7; // r8
  int v8; // r11d
  __int64 *v9; // rcx
  unsigned int v10; // edi
  __int64 *v11; // rax
  __int64 v12; // rdx
  int v14; // eax
  int v15; // edx
  int v16; // eax
  int v17; // edi
  __int64 v18; // r8
  unsigned int v19; // esi
  __int64 v20; // rax
  int v21; // r10d
  __int64 *v22; // r9
  int v23; // eax
  int v24; // esi
  __int64 v25; // rdi
  __int64 *v26; // r8
  unsigned int v27; // r14d
  int v28; // r9d
  __int64 v29; // rax

  v4 = *a1;
  v6 = *(_DWORD *)(*a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)v4;
    goto LABEL_18;
  }
  v7 = *(_QWORD *)(v4 + 8);
  v8 = 1;
  v9 = 0;
  v10 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (__int64 *)(v7 + 16LL * v10);
  v12 = *v11;
  if ( *v11 == a2 )
  {
LABEL_3:
    *((_DWORD *)v11 + 2) = a3;
    return v11 + 1;
  }
  while ( v12 != -4096 )
  {
    if ( !v9 && v12 == -8192 )
      v9 = v11;
    v10 = (v6 - 1) & (v8 + v10);
    v11 = (__int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == a2 )
      goto LABEL_3;
    ++v8;
  }
  if ( !v9 )
    v9 = v11;
  v14 = *(_DWORD *)(v4 + 16);
  ++*(_QWORD *)v4;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v6 )
  {
LABEL_18:
    sub_34F9190(v4, 2 * v6);
    v16 = *(_DWORD *)(v4 + 24);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(v4 + 8);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(v4 + 16) + 1;
      v9 = (__int64 *)(v18 + 16LL * v19);
      v20 = *v9;
      if ( *v9 != a2 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -4096 )
        {
          if ( !v22 && v20 == -8192 )
            v22 = v9;
          v19 = v17 & (v21 + v19);
          v9 = (__int64 *)(v18 + 16LL * v19);
          v20 = *v9;
          if ( *v9 == a2 )
            goto LABEL_14;
          ++v21;
        }
        if ( v22 )
          v9 = v22;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v6 - *(_DWORD *)(v4 + 20) - v15 <= v6 >> 3 )
  {
    sub_34F9190(v4, v6);
    v23 = *(_DWORD *)(v4 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(v4 + 8);
      v26 = 0;
      v27 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = 1;
      v15 = *(_DWORD *)(v4 + 16) + 1;
      v9 = (__int64 *)(v25 + 16LL * v27);
      v29 = *v9;
      if ( *v9 != a2 )
      {
        while ( v29 != -4096 )
        {
          if ( !v26 && v29 == -8192 )
            v26 = v9;
          v27 = v24 & (v28 + v27);
          v9 = (__int64 *)(v25 + 16LL * v27);
          v29 = *v9;
          if ( *v9 == a2 )
            goto LABEL_14;
          ++v28;
        }
        if ( v26 )
          v9 = v26;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(v4 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(v4 + 16) = v15;
  if ( *v9 != -4096 )
    --*(_DWORD *)(v4 + 20);
  *v9 = a2;
  *((_DWORD *)v9 + 2) = 0;
  *((_DWORD *)v9 + 2) = a3;
  return v9 + 1;
}
