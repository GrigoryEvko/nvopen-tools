// Function: sub_2BFE5E0
// Address: 0x2bfe5e0
//
__int64 __fastcall sub_2BFE5E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  unsigned int v5; // esi
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // r8
  int v9; // r11d
  _QWORD *v10; // rcx
  unsigned int v11; // edi
  _QWORD *v12; // rax
  __int64 v13; // rdx
  int v15; // eax
  int v16; // edx
  int v17; // eax
  int v18; // edi
  __int64 v19; // r8
  unsigned int v20; // esi
  __int64 v21; // rax
  int v22; // r10d
  _QWORD *v23; // r9
  int v24; // eax
  int v25; // esi
  __int64 v26; // rdi
  int v27; // r9d
  unsigned int v28; // r14d
  _QWORD *v29; // r8
  __int64 v30; // rax

  v4 = sub_2BFD6A0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
  v5 = *(_DWORD *)(a1 + 24);
  v6 = v4;
  v7 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v8 = *(_QWORD *)(a1 + 8);
  v9 = 1;
  v10 = 0;
  v11 = (v5 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v12 = (_QWORD *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( *v12 == v7 )
  {
LABEL_3:
    v12[1] = v6;
    return v6;
  }
  while ( v13 != -4096 )
  {
    if ( !v10 && v13 == -8192 )
      v10 = v12;
    v11 = (v5 - 1) & (v9 + v11);
    v12 = (_QWORD *)(v8 + 16LL * v11);
    v13 = *v12;
    if ( v7 == *v12 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v10 )
    v10 = v12;
  v15 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v5 )
  {
LABEL_18:
    sub_2BFD020(a1, 2 * v5);
    v17 = *(_DWORD *)(a1 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 8);
      v20 = (v17 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v16 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (_QWORD *)(v19 + 16LL * v20);
      v21 = *v10;
      if ( v7 != *v10 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -4096 )
        {
          if ( !v23 && v21 == -8192 )
            v23 = v10;
          v20 = v18 & (v22 + v20);
          v10 = (_QWORD *)(v19 + 16LL * v20);
          v21 = *v10;
          if ( v7 == *v10 )
            goto LABEL_14;
          ++v22;
        }
        if ( v23 )
          v10 = v23;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v16 <= v5 >> 3 )
  {
    sub_2BFD020(a1, v5);
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 8);
      v27 = 1;
      v28 = (v24 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v29 = 0;
      v16 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (_QWORD *)(v26 + 16LL * v28);
      v30 = *v10;
      if ( v7 != *v10 )
      {
        while ( v30 != -4096 )
        {
          if ( !v29 && v30 == -8192 )
            v29 = v10;
          v28 = v25 & (v27 + v28);
          v10 = (_QWORD *)(v26 + 16LL * v28);
          v30 = *v10;
          if ( v7 == *v10 )
            goto LABEL_14;
          ++v27;
        }
        if ( v29 )
          v10 = v29;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v16;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v10 = v7;
  v10[1] = 0;
  v10[1] = v6;
  return v6;
}
