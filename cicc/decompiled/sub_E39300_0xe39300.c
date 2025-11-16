// Function: sub_E39300
// Address: 0xe39300
//
_QWORD *__fastcall sub_E39300(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // rcx
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r8
  _QWORD *v9; // r13
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // r14d
  unsigned int v15; // r9d
  __int64 *v16; // rdx
  __int64 v17; // r8
  _QWORD *v18; // rax
  __int64 v19; // rdi
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // r11
  __int64 *v23; // r9
  int v24; // r15d
  int v25; // eax
  int v26; // edx
  int v27; // r9d
  int v28; // edx
  int v29; // r11d
  int v30; // eax
  int v31; // edi
  __int64 v32; // rsi
  unsigned int v33; // r14d
  __int64 v34; // rcx
  __int64 *v35; // rax
  int v36; // r8d
  int v37; // eax
  int v38; // edi
  __int64 v39; // rsi
  unsigned int v40; // r14d
  __int64 v41; // rcx
  int v42; // r8d

  v4 = *(_DWORD *)(a1 + 64);
  v5 = *(_QWORD *)(a1 + 48);
  if ( v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( *v7 == a2 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16LL * v4) )
        return (_QWORD *)v7[1];
    }
    else
    {
      v11 = 1;
      while ( v8 != -4096 )
      {
        v27 = v11 + 1;
        v6 = (v4 - 1) & (v11 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        v11 = v27;
      }
    }
  }
  v12 = *(unsigned int *)(a1 + 32);
  v13 = *(_QWORD *)(a1 + 16);
  if ( !(_DWORD)v12 )
    return 0;
  v14 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v15 = (v12 - 1) & v14;
  v16 = (__int64 *)(v13 + 16LL * v15);
  v17 = *v16;
  if ( *v16 != a2 )
  {
    v28 = 1;
    while ( v17 != -4096 )
    {
      v29 = v28 + 1;
      v15 = (v12 - 1) & (v28 + v15);
      v16 = (__int64 *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == a2 )
        goto LABEL_10;
      v28 = v29;
    }
    return 0;
  }
LABEL_10:
  if ( v16 == (__int64 *)(v13 + 16 * v12) )
    return 0;
  v18 = (_QWORD *)v16[1];
  do
  {
    v9 = v18;
    v18 = (_QWORD *)*v18;
  }
  while ( v18 );
  v19 = a1 + 40;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_32;
  }
  v20 = (v4 - 1) & v14;
  v21 = (__int64 *)(v5 + 16LL * v20);
  v22 = *v21;
  if ( *v21 == a2 )
    return v9;
  v23 = 0;
  v24 = 1;
  while ( v22 != -4096 )
  {
    if ( !v23 && v22 == -8192 )
      v23 = v21;
    v20 = (v4 - 1) & (v24 + v20);
    v21 = (__int64 *)(v5 + 16LL * v20);
    v22 = *v21;
    if ( *v21 == a2 )
      return v9;
    ++v24;
  }
  if ( !v23 )
    v23 = v21;
  v25 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  v26 = v25 + 1;
  if ( 4 * (v25 + 1) >= 3 * v4 )
  {
LABEL_32:
    sub_E39120(v19, 2 * v4);
    v30 = *(_DWORD *)(a1 + 64);
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = *(_QWORD *)(a1 + 48);
      v33 = (v30 - 1) & v14;
      v26 = *(_DWORD *)(a1 + 56) + 1;
      v23 = (__int64 *)(v32 + 16LL * v33);
      v34 = *v23;
      if ( *v23 == a2 )
        goto LABEL_21;
      v35 = (__int64 *)(v32 + 16LL * v33);
      v36 = 1;
      v23 = 0;
      while ( v34 != -4096 )
      {
        if ( !v23 && v34 == -8192 )
          v23 = v35;
        v33 = v31 & (v36 + v33);
        v35 = (__int64 *)(v32 + 16LL * v33);
        v34 = *v35;
        if ( *v35 == a2 )
        {
LABEL_47:
          v23 = v35;
          goto LABEL_21;
        }
        ++v36;
      }
LABEL_36:
      if ( !v23 )
        v23 = v35;
      goto LABEL_21;
    }
LABEL_59:
    ++*(_DWORD *)(a1 + 56);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 60) - v26 <= v4 >> 3 )
  {
    sub_E39120(v19, v4);
    v37 = *(_DWORD *)(a1 + 64);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a1 + 48);
      v40 = (v37 - 1) & v14;
      v26 = *(_DWORD *)(a1 + 56) + 1;
      v23 = (__int64 *)(v39 + 16LL * v40);
      v41 = *v23;
      if ( *v23 == a2 )
        goto LABEL_21;
      v35 = (__int64 *)(v39 + 16LL * v40);
      v42 = 1;
      v23 = 0;
      while ( v41 != -4096 )
      {
        if ( v41 == -8192 && !v23 )
          v23 = v35;
        v40 = v38 & (v42 + v40);
        v35 = (__int64 *)(v39 + 16LL * v40);
        v41 = *v35;
        if ( *v35 == a2 )
          goto LABEL_47;
        ++v42;
      }
      goto LABEL_36;
    }
    goto LABEL_59;
  }
LABEL_21:
  *(_DWORD *)(a1 + 56) = v26;
  if ( *v23 != -4096 )
    --*(_DWORD *)(a1 + 60);
  *v23 = a2;
  v23[1] = (__int64)v9;
  return v9;
}
