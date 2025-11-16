// Function: sub_1F08EC0
// Address: 0x1f08ec0
//
_QWORD *__fastcall sub_1F08EC0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // rbx
  _QWORD *i; // rdx
  __int64 *v10; // rdx
  __int64 v11; // rcx
  int v12; // r8d
  int v13; // r8d
  __int64 v14; // r10
  __int64 v15; // r11
  int v16; // r9d
  int v17; // r14d
  unsigned __int64 v18; // rcx
  unsigned int j; // r9d
  __int64 v20; // rsi
  __int64 v21; // r15
  unsigned __int64 v22; // r15
  unsigned int v23; // r9d
  __int64 v24; // rax
  _QWORD *k; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[2 * v3];
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
    v10 = v4;
    if ( v8 == v4 )
      return (_QWORD *)j___libc_free_0(v4);
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = *v10;
        if ( ((*v10 >> 2) & 1) != 0 || (v11 & 0xFFFFFFFFFFFFFFF0LL) != 0xFFFFFFFFFFFFFFF0LL )
          break;
        v10 += 2;
        if ( v8 == v10 )
          return (_QWORD *)j___libc_free_0(v4);
      }
      v12 = *(_DWORD *)(a1 + 24);
      if ( !v12 )
      {
        MEMORY[0] = *v10;
        BUG();
      }
      v13 = v12 - 1;
      v14 = *(_QWORD *)(a1 + 8);
      v15 = 0;
      v16 = 37 * v11;
      v17 = 1;
      v18 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      for ( j = v13 & v16; ; j = v13 & v23 )
      {
        v20 = v14 + 16LL * j;
        v21 = *(_QWORD *)v20;
        if ( !((*v10 >> 2) & 1) != !((*(__int64 *)v20 >> 2) & 1) )
          break;
        v22 = v21 & 0xFFFFFFFFFFFFFFF8LL;
        if ( ((*v10 >> 2) & 1) == 0 )
        {
          if ( v18 == v22 )
            goto LABEL_29;
          goto LABEL_20;
        }
        if ( v18 == v22 )
          goto LABEL_29;
LABEL_26:
        v23 = v17 + j;
        ++v17;
      }
      if ( ((*(__int64 *)v20 >> 2) & 1) != 0 )
        goto LABEL_26;
      v22 = v21 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_20:
      if ( v22 != -8 )
      {
        if ( !v15 && v22 == -16 )
          v15 = v14 + 16LL * j;
        goto LABEL_26;
      }
      if ( v15 )
        v20 = v15;
LABEL_29:
      v24 = *v10;
      v10 += 2;
      *(_QWORD *)v20 = v24;
      *(_DWORD *)(v20 + 8) = *((_DWORD *)v10 - 2);
      ++*(_DWORD *)(a1 + 16);
      if ( v8 == v10 )
        return (_QWORD *)j___libc_free_0(v4);
    }
  }
  *(_QWORD *)(a1 + 16) = 0;
  for ( k = &result[2 * *(unsigned int *)(a1 + 24)]; k != result; result += 2 )
  {
    if ( result )
      *result = -8;
  }
  return result;
}
