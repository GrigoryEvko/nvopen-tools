// Function: sub_B6FBA0
// Address: 0xb6fba0
//
__int64 __fastcall sub_B6FBA0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  unsigned int v4; // esi
  __int64 v5; // rdi
  __int64 v6; // r8
  int v7; // r14d
  _QWORD *v8; // rdx
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r10
  _DWORD *v12; // rdx
  __int64 result; // rax
  int v14; // eax
  int v15; // ecx
  _DWORD *v16; // rdx
  int v17; // eax
  int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // eax
  __int64 v21; // rdi
  int v22; // r10d
  _QWORD *v23; // r9
  int v24; // eax
  int v25; // eax
  __int64 v26; // rdi
  int v27; // r9d
  unsigned int v28; // r13d
  _QWORD *v29; // r8
  __int64 v30; // rsi

  v2 = *a1;
  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_DWORD *)(*a1 + 88);
  v5 = *a1 + 64;
  if ( !v4 )
  {
    ++*(_QWORD *)(v2 + 64);
    goto LABEL_18;
  }
  v6 = *(_QWORD *)(v2 + 72);
  v7 = 1;
  v8 = 0;
  v9 = (v4 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v3 == *v10 )
  {
LABEL_3:
    v12 = v10 + 1;
    result = *((unsigned int *)v10 + 2);
    *v12 = result + 1;
    return result;
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
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(v2 + 80);
  ++*(_QWORD *)(v2 + 64);
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v4 )
  {
LABEL_18:
    sub_B6F9C0(v5, 2 * v4);
    v17 = *(_DWORD *)(v2 + 88);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(v2 + 72);
      v20 = (v17 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v15 = *(_DWORD *)(v2 + 80) + 1;
      v8 = (_QWORD *)(v19 + 16LL * v20);
      v21 = *v8;
      if ( v3 != *v8 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -4096 )
        {
          if ( !v23 && v21 == -8192 )
            v23 = v8;
          v20 = v18 & (v22 + v20);
          v8 = (_QWORD *)(v19 + 16LL * v20);
          v21 = *v8;
          if ( v3 == *v8 )
            goto LABEL_14;
          ++v22;
        }
        if ( v23 )
          v8 = v23;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v4 - *(_DWORD *)(v2 + 84) - v15 <= v4 >> 3 )
  {
    sub_B6F9C0(v5, v4);
    v24 = *(_DWORD *)(v2 + 88);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(v2 + 72);
      v27 = 1;
      v28 = v25 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v29 = 0;
      v15 = *(_DWORD *)(v2 + 80) + 1;
      v8 = (_QWORD *)(v26 + 16LL * v28);
      v30 = *v8;
      if ( v3 != *v8 )
      {
        while ( v30 != -4096 )
        {
          if ( !v29 && v30 == -8192 )
            v29 = v8;
          v28 = v25 & (v27 + v28);
          v8 = (_QWORD *)(v26 + 16LL * v28);
          v30 = *v8;
          if ( v3 == *v8 )
            goto LABEL_14;
          ++v27;
        }
        if ( v29 )
          v8 = v29;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(v2 + 80);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(v2 + 80) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(v2 + 84);
  *v8 = v3;
  v16 = v8 + 1;
  *v16 = 0;
  *v16 = 1;
  return 0;
}
