// Function: sub_11F65C0
// Address: 0x11f65c0
//
__int64 __fastcall sub_11F65C0(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // esi
  __int64 v7; // rdi
  __int64 v8; // r8
  int v9; // r14d
  __int64 *v10; // rdx
  unsigned int v11; // r13d
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r10
  int v16; // eax
  int v17; // ecx
  int v18; // eax
  int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // eax
  __int64 v22; // rdi
  int v23; // r10d
  __int64 *v24; // r9
  int v25; // eax
  int v26; // eax
  __int64 v27; // rdi
  int v28; // r9d
  unsigned int v29; // r13d
  __int64 *v30; // r8
  __int64 v31; // rsi

  v3 = sub_B46EC0(a2, a3);
  v4 = *a1;
  v5 = v3;
  v6 = *(_DWORD *)(*a1 + 32);
  v7 = *a1 + 8;
  if ( !v6 )
  {
    ++*(_QWORD *)(v4 + 8);
    goto LABEL_18;
  }
  v8 = *(_QWORD *)(v4 + 16);
  v9 = 1;
  v10 = 0;
  v11 = ((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4);
  v12 = (v6 - 1) & v11;
  v13 = (__int64 *)(v8 + 16LL * v12);
  v14 = *v13;
  if ( v5 == *v13 )
    return *((unsigned __int8 *)v13 + 8) ^ 1u;
  while ( v14 != -4096 )
  {
    if ( !v10 && v14 == -8192 )
      v10 = v13;
    v12 = (v6 - 1) & (v9 + v12);
    v13 = (__int64 *)(v8 + 16LL * v12);
    v14 = *v13;
    if ( v5 == *v13 )
      return *((unsigned __int8 *)v13 + 8) ^ 1u;
    ++v9;
  }
  if ( !v10 )
    v10 = v13;
  v16 = *(_DWORD *)(v4 + 24);
  ++*(_QWORD *)(v4 + 8);
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v6 )
  {
LABEL_18:
    sub_11F63E0(v7, 2 * v6);
    v18 = *(_DWORD *)(v4 + 32);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(v4 + 16);
      v21 = (v18 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v17 = *(_DWORD *)(v4 + 24) + 1;
      v10 = (__int64 *)(v20 + 16LL * v21);
      v22 = *v10;
      if ( v5 != *v10 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -4096 )
        {
          if ( !v24 && v22 == -8192 )
            v24 = v10;
          v21 = v19 & (v23 + v21);
          v10 = (__int64 *)(v20 + 16LL * v21);
          v22 = *v10;
          if ( v5 == *v10 )
            goto LABEL_14;
          ++v23;
        }
        if ( v24 )
          v10 = v24;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v6 - *(_DWORD *)(v4 + 28) - v17 <= v6 >> 3 )
  {
    sub_11F63E0(v7, v6);
    v25 = *(_DWORD *)(v4 + 32);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(v4 + 16);
      v28 = 1;
      v29 = v26 & v11;
      v30 = 0;
      v17 = *(_DWORD *)(v4 + 24) + 1;
      v10 = (__int64 *)(v27 + 16LL * v29);
      v31 = *v10;
      if ( v5 != *v10 )
      {
        while ( v31 != -4096 )
        {
          if ( !v30 && v31 == -8192 )
            v30 = v10;
          v29 = v26 & (v28 + v29);
          v10 = (__int64 *)(v27 + 16LL * v29);
          v31 = *v10;
          if ( v5 == *v10 )
            goto LABEL_14;
          ++v28;
        }
        if ( v30 )
          v10 = v30;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(v4 + 24);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(v4 + 24) = v17;
  if ( *v10 != -4096 )
    --*(_DWORD *)(v4 + 28);
  *v10 = v5;
  *((_BYTE *)v10 + 8) = 0;
  return 1;
}
