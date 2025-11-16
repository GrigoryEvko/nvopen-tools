// Function: sub_15C08C0
// Address: 0x15c08c0
//
_QWORD *__fastcall sub_15C08C0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // r15d
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rsi
  int v16; // eax
  int v17; // r15d
  int v18; // eax
  __int64 v19; // rsi
  unsigned int v20; // eax
  __int64 *v21; // rdx
  __int64 v22; // rdi
  int v23; // r10d
  __int64 *v24; // r9
  __int64 v25; // rdx
  _QWORD *k; // rdx
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h] BYREF
  __int64 v29; // [rsp+18h] [rbp-48h] BYREF
  int v30[16]; // [rsp+20h] [rbp-40h] BYREF

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
  result = (_QWORD *)sub_22077B0(8LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[v3];
    for ( i = &result[v7]; i != result; ++result )
    {
      if ( result )
        *result = -8;
    }
    for ( j = v4; v8 != j; ++j )
    {
      v11 = *j;
      if ( *j != -16 && v11 != -8 )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v13 = *(unsigned int *)(v11 + 8);
        v14 = *(_QWORD *)(a1 + 8);
        v15 = *j;
        v28 = *(_QWORD *)(v11 + 8 * (1 - v13));
        if ( *(_BYTE *)v11 != 15 )
          v15 = *(_QWORD *)(v11 - 8 * v13);
        v16 = *(_DWORD *)(v11 + 24);
        v29 = v15;
        v27 = v14;
        v17 = v12 - 1;
        v30[0] = v16;
        v18 = sub_15B2A30(&v28, &v29, v30);
        v19 = *j;
        v20 = v17 & v18;
        v21 = (__int64 *)(v27 + 8LL * v20);
        v22 = *v21;
        if ( *v21 != *j )
        {
          v23 = 1;
          v24 = 0;
          while ( v22 != -8 )
          {
            if ( v22 != -16 || v24 )
              v21 = v24;
            v20 = v17 & (v23 + v20);
            v22 = *(_QWORD *)(v27 + 8LL * v20);
            if ( v22 == v19 )
            {
              v21 = (__int64 *)(v27 + 8LL * v20);
              goto LABEL_23;
            }
            ++v23;
            v24 = v21;
            v21 = (__int64 *)(v27 + 8LL * v20);
          }
          if ( v24 )
            v21 = v24;
        }
LABEL_23:
        *v21 = v19;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v25]; k != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
