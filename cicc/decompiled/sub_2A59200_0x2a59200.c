// Function: sub_2A59200
// Address: 0x2a59200
//
_QWORD *__fastcall sub_2A59200(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  unsigned int v7; // esi
  __int64 v8; // r8
  int v9; // r11d
  _QWORD *v10; // rcx
  unsigned int v11; // r14d
  unsigned int v12; // edi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  int v16; // eax
  int v17; // edx
  int v18; // eax
  int v19; // edi
  __int64 v20; // r8
  unsigned int v21; // esi
  __int64 v22; // rax
  int v23; // r10d
  _QWORD *v24; // r9
  int v25; // eax
  int v26; // esi
  __int64 v27; // rdi
  _QWORD *v28; // r8
  unsigned int v29; // r14d
  int v30; // r9d
  __int64 v31; // rax

  v6 = *a1 + 104LL * a2;
  v7 = *(_DWORD *)(v6 + 24);
  if ( !v7 )
  {
    ++*(_QWORD *)v6;
    goto LABEL_18;
  }
  v8 = *(_QWORD *)(v6 + 8);
  v9 = 1;
  v10 = 0;
  v11 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v12 = (v7 - 1) & v11;
  v13 = (_QWORD *)(v8 + 16LL * v12);
  v14 = *v13;
  if ( *v13 == a3 )
  {
LABEL_3:
    v13[1] = a4;
    return v13 + 1;
  }
  while ( v14 != -4096 )
  {
    if ( !v10 && v14 == -8192 )
      v10 = v13;
    v12 = (v7 - 1) & (v9 + v12);
    v13 = (_QWORD *)(v8 + 16LL * v12);
    v14 = *v13;
    if ( *v13 == a3 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v10 )
    v10 = v13;
  v16 = *(_DWORD *)(v6 + 16);
  ++*(_QWORD *)v6;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v7 )
  {
LABEL_18:
    sub_116E750(v6, 2 * v7);
    v18 = *(_DWORD *)(v6 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(v6 + 8);
      v21 = (v18 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v17 = *(_DWORD *)(v6 + 16) + 1;
      v10 = (_QWORD *)(v20 + 16LL * v21);
      v22 = *v10;
      if ( *v10 != a3 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -4096 )
        {
          if ( !v24 && v22 == -8192 )
            v24 = v10;
          v21 = v19 & (v23 + v21);
          v10 = (_QWORD *)(v20 + 16LL * v21);
          v22 = *v10;
          if ( *v10 == a3 )
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
  if ( v7 - *(_DWORD *)(v6 + 20) - v17 <= v7 >> 3 )
  {
    sub_116E750(v6, v7);
    v25 = *(_DWORD *)(v6 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(v6 + 8);
      v28 = 0;
      v29 = (v25 - 1) & v11;
      v30 = 1;
      v17 = *(_DWORD *)(v6 + 16) + 1;
      v10 = (_QWORD *)(v27 + 16LL * v29);
      v31 = *v10;
      if ( *v10 != a3 )
      {
        while ( v31 != -4096 )
        {
          if ( !v28 && v31 == -8192 )
            v28 = v10;
          v29 = v26 & (v30 + v29);
          v10 = (_QWORD *)(v27 + 16LL * v29);
          v31 = *v10;
          if ( *v10 == a3 )
            goto LABEL_14;
          ++v30;
        }
        if ( v28 )
          v10 = v28;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(v6 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(v6 + 16) = v17;
  if ( *v10 != -4096 )
    --*(_DWORD *)(v6 + 20);
  *v10 = a3;
  v10[1] = 0;
  v10[1] = a4;
  return v10 + 1;
}
