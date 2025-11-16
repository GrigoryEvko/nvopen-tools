// Function: sub_3115B10
// Address: 0x3115b10
//
__int64 *__fastcall sub_3115B10(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  unsigned int v4; // esi
  int v5; // r14d
  __int64 v6; // r8
  int v7; // r11d
  __int64 *v8; // rcx
  unsigned int v9; // edi
  __int64 *v10; // rax
  __int64 v11; // rdx
  int v13; // edx
  int v14; // eax
  int v15; // edi
  __int64 v16; // r8
  unsigned int v17; // esi
  __int64 v18; // rax
  int v19; // r10d
  __int64 *v20; // r9
  int v21; // eax
  int v22; // esi
  __int64 v23; // rdi
  int v24; // r9d
  unsigned int v25; // r13d
  __int64 *v26; // r8
  __int64 v27; // rax

  v2 = *a1;
  v3 = *a2;
  v4 = *(_DWORD *)(*a1 + 24);
  v5 = *(_DWORD *)(*a1 + 16);
  if ( !v4 )
  {
    ++*(_QWORD *)v2;
    goto LABEL_18;
  }
  v6 = *(_QWORD *)(v2 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v4 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v3 == *v10 )
  {
LABEL_3:
    *((_DWORD *)v10 + 2) = v5;
    return v10 + 1;
  }
  while ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    v9 = (v4 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v3 == *v10 )
      goto LABEL_3;
    ++v7;
  }
  v13 = v5 + 1;
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)v2;
  if ( 4 * v13 >= 3 * v4 )
  {
LABEL_18:
    sub_3115930(v2, 2 * v4);
    v14 = *(_DWORD *)(v2 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(v2 + 8);
      v17 = (v14 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v13 = *(_DWORD *)(v2 + 16) + 1;
      v8 = (__int64 *)(v16 + 16LL * v17);
      v18 = *v8;
      if ( v3 != *v8 )
      {
        v19 = 1;
        v20 = 0;
        while ( v18 != -4096 )
        {
          if ( !v20 && v18 == -8192 )
            v20 = v8;
          v17 = v15 & (v19 + v17);
          v8 = (__int64 *)(v16 + 16LL * v17);
          v18 = *v8;
          if ( v3 == *v8 )
            goto LABEL_14;
          ++v19;
        }
        if ( v20 )
          v8 = v20;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v4 - *(_DWORD *)(v2 + 20) - v13 <= v4 >> 3 )
  {
    sub_3115930(v2, v4);
    v21 = *(_DWORD *)(v2 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(v2 + 8);
      v24 = 1;
      v25 = (v21 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v26 = 0;
      v13 = *(_DWORD *)(v2 + 16) + 1;
      v8 = (__int64 *)(v23 + 16LL * v25);
      v27 = *v8;
      if ( v3 != *v8 )
      {
        while ( v27 != -4096 )
        {
          if ( v27 == -8192 && !v26 )
            v26 = v8;
          v25 = v22 & (v24 + v25);
          v8 = (__int64 *)(v23 + 16LL * v25);
          v27 = *v8;
          if ( v3 == *v8 )
            goto LABEL_14;
          ++v24;
        }
        if ( v26 )
          v8 = v26;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(v2 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(v2 + 16) = v13;
  if ( *v8 != -4096 )
    --*(_DWORD *)(v2 + 20);
  *v8 = v3;
  *((_DWORD *)v8 + 2) = 0;
  *((_DWORD *)v8 + 2) = v5;
  return v8 + 1;
}
