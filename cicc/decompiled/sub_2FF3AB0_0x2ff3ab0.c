// Function: sub_2FF3AB0
// Address: 0x2ff3ab0
//
_QWORD *__fastcall sub_2FF3AB0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // r8
  int v10; // r10d
  _QWORD *v11; // rdx
  unsigned int v12; // edi
  _QWORD *v13; // rax
  __int64 v14; // rcx
  _QWORD *result; // rax
  int v16; // eax
  int v17; // ecx
  int v18; // eax
  int v19; // eax
  __int64 v20; // r8
  unsigned int v21; // esi
  __int64 v22; // rdi
  int v23; // r10d
  _QWORD *v24; // r9
  int v25; // eax
  int v26; // eax
  __int64 v27; // rdi
  _QWORD *v28; // r8
  unsigned int v29; // r14d
  int v30; // r9d
  __int64 v31; // rsi

  v7 = *(_QWORD *)(a1 + 264);
  v8 = *(_DWORD *)(v7 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)v7;
    goto LABEL_19;
  }
  v9 = *(_QWORD *)(v7 + 8);
  v10 = 1;
  v11 = 0;
  v12 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (_QWORD *)(v9 + 24LL * v12);
  v14 = *v13;
  if ( *v13 == a2 )
  {
LABEL_3:
    result = v13 + 1;
    goto LABEL_4;
  }
  while ( v14 != -4096 )
  {
    if ( !v11 && v14 == -8192 )
      v11 = v13;
    v12 = (v8 - 1) & (v10 + v12);
    v13 = (_QWORD *)(v9 + 24LL * v12);
    v14 = *v13;
    if ( *v13 == a2 )
      goto LABEL_3;
    ++v10;
  }
  if ( !v11 )
    v11 = v13;
  v16 = *(_DWORD *)(v7 + 16);
  ++*(_QWORD *)v7;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v8 )
  {
LABEL_19:
    sub_2FF38B0(v7, 2 * v8);
    v18 = *(_DWORD *)(v7 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(v7 + 8);
      v21 = v19 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = *(_DWORD *)(v7 + 16) + 1;
      v11 = (_QWORD *)(v20 + 24LL * v21);
      v22 = *v11;
      if ( *v11 != a2 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -4096 )
        {
          if ( !v24 && v22 == -8192 )
            v24 = v11;
          v21 = v19 & (v23 + v21);
          v11 = (_QWORD *)(v20 + 24LL * v21);
          v22 = *v11;
          if ( *v11 == a2 )
            goto LABEL_15;
          ++v23;
        }
        if ( v24 )
          v11 = v24;
      }
      goto LABEL_15;
    }
    goto LABEL_42;
  }
  if ( v8 - *(_DWORD *)(v7 + 20) - v17 <= v8 >> 3 )
  {
    sub_2FF38B0(v7, v8);
    v25 = *(_DWORD *)(v7 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(v7 + 8);
      v28 = 0;
      v29 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v30 = 1;
      v17 = *(_DWORD *)(v7 + 16) + 1;
      v11 = (_QWORD *)(v27 + 24LL * v29);
      v31 = *v11;
      if ( *v11 != a2 )
      {
        while ( v31 != -4096 )
        {
          if ( !v28 && v31 == -8192 )
            v28 = v11;
          v29 = v26 & (v30 + v29);
          v11 = (_QWORD *)(v27 + 24LL * v29);
          v31 = *v11;
          if ( *v11 == a2 )
            goto LABEL_15;
          ++v30;
        }
        if ( v28 )
          v11 = v28;
      }
      goto LABEL_15;
    }
LABEL_42:
    ++*(_DWORD *)(v7 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(v7 + 16) = v17;
  if ( *v11 != -4096 )
    --*(_DWORD *)(v7 + 20);
  *v11 = a2;
  result = v11 + 1;
  v11[1] = 0;
  *((_BYTE *)v11 + 16) = 0;
LABEL_4:
  *result = a3;
  *((_BYTE *)result + 8) = a4;
  return result;
}
