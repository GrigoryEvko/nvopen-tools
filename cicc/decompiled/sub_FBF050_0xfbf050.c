// Function: sub_FBF050
// Address: 0xfbf050
//
_QWORD *__fastcall sub_FBF050(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  bool v5; // zf
  _QWORD *result; // rax
  __int64 v7; // rdx
  _QWORD *i; // rdx
  __int64 v9; // r8
  int v10; // edi
  int v11; // r10d
  __int64 v12; // r9
  __int64 v13; // rcx
  _QWORD *v14; // rdx
  _QWORD *v15; // rsi
  __int64 v16; // rdi
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rdx
  _QWORD *v21; // r12
  __int64 v22; // r13
  _QWORD *v23; // r14
  _QWORD *v24; // rdi
  int v25; // edx

  v4 = a2;
  v5 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v5 )
  {
    result = *(_QWORD **)(a1 + 16);
    v7 = 31LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = (_QWORD *)(a1 + 16);
    v7 = 124;
  }
  for ( i = &result[v7]; i != result; result += 31 )
  {
    if ( result )
      *result = -4096;
  }
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      result = *(_QWORD **)v4;
      if ( *(_QWORD *)v4 != -8192 && result != (_QWORD *)-4096LL )
        break;
      v22 = v4 + 248;
LABEL_26:
      v4 = v22;
      if ( a3 == v22 )
        return result;
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v9 = a1 + 16;
      v10 = 3;
    }
    else
    {
      v25 = *(_DWORD *)(a1 + 24);
      v9 = *(_QWORD *)(a1 + 16);
      if ( !v25 )
      {
        MEMORY[0] = *(_QWORD *)v4;
        BUG();
      }
      v10 = v25 - 1;
    }
    v11 = 1;
    v12 = 0;
    v13 = v10 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v14 = (_QWORD *)(v9 + 248 * v13);
    v15 = (_QWORD *)*v14;
    if ( result != (_QWORD *)*v14 )
    {
      while ( v15 != (_QWORD *)-4096LL )
      {
        if ( v15 == (_QWORD *)-8192LL && !v12 )
          v12 = (__int64)v14;
        v13 = v10 & (unsigned int)(v11 + v13);
        v14 = (_QWORD *)(v9 + 248LL * (unsigned int)v13);
        v15 = (_QWORD *)*v14;
        if ( result == (_QWORD *)*v14 )
          goto LABEL_13;
        ++v11;
      }
      if ( v12 )
        v14 = (_QWORD *)v12;
    }
LABEL_13:
    *v14 = result;
    v16 = (__int64)(v14 + 1);
    v17 = v14 + 3;
    v18 = (__int64)(v14 + 31);
    *(_QWORD *)(v18 - 240) = 0;
    v19 = v4 + 8;
    *(_QWORD *)(v18 - 232) = 1;
    do
    {
      if ( v17 )
        *v17 = -4096;
      v17 += 7;
    }
    while ( v17 != (_QWORD *)v18 );
    sub_FBED40(v16, (char **)v19, v18, v13, v9, v12);
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    if ( (*(_BYTE *)(v4 + 16) & 1) != 0 )
    {
      v22 = v4 + 248;
      v21 = (_QWORD *)(v4 + 24);
      v23 = (_QWORD *)(v4 + 248);
    }
    else
    {
      v20 = *(unsigned int *)(v4 + 32);
      v21 = *(_QWORD **)(v4 + 24);
      v22 = v4 + 248;
      v19 = 56 * v20;
      if ( !(_DWORD)v20 || (v23 = &v21[7 * v20], v22 = v4 + 248, v23 == v21) )
      {
LABEL_32:
        result = (_QWORD *)sub_C7D6A0((__int64)v21, v19, 8);
        goto LABEL_26;
      }
    }
    do
    {
      result = (_QWORD *)*v21;
      if ( *v21 != -8192 && result != (_QWORD *)-4096LL )
      {
        v24 = (_QWORD *)v21[1];
        result = v21 + 3;
        if ( v24 != v21 + 3 )
          result = (_QWORD *)_libc_free(v24, v19);
      }
      v21 += 7;
    }
    while ( v23 != v21 );
    if ( (*(_BYTE *)(v4 + 16) & 1) != 0 )
      goto LABEL_26;
    v21 = *(_QWORD **)(v4 + 24);
    v19 = 56LL * *(unsigned int *)(v4 + 32);
    goto LABEL_32;
  }
  return result;
}
