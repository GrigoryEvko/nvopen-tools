// Function: sub_15C51C0
// Address: 0x15c51c0
//
_QWORD *__fastcall sub_15C51C0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rcx
  __int64 *v8; // r14
  _QWORD *i; // rcx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // r15d
  __int64 v13; // rcx
  int v14; // r15d
  int v15; // eax
  __int64 v16; // rsi
  unsigned int v17; // eax
  _QWORD *v18; // rcx
  __int64 v19; // rdi
  int v20; // r10d
  _QWORD *v21; // r9
  __int64 v22; // rdx
  _QWORD *k; // rdx
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h] BYREF
  __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

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
        v14 = v12 - 1;
        v24 = *(_QWORD *)(a1 + 8);
        v25 = *(_QWORD *)(v11 - 8 * v13);
        v26[0] = *(_QWORD *)(v11 + 8 * (1 - v13));
        v15 = sub_15B2F00(&v25, v26);
        v16 = *j;
        v17 = v14 & v15;
        v18 = (_QWORD *)(v24 + 8LL * v17);
        v19 = *v18;
        if ( *j != *v18 )
        {
          v20 = 1;
          v21 = 0;
          while ( v19 != -8 )
          {
            if ( v19 != -16 || v21 )
              v18 = v21;
            v17 = v14 & (v20 + v17);
            v19 = *(_QWORD *)(v24 + 8LL * v17);
            if ( v19 == v16 )
            {
              v18 = (_QWORD *)(v24 + 8LL * v17);
              goto LABEL_21;
            }
            ++v20;
            v21 = v18;
            v18 = (_QWORD *)(v24 + 8LL * v17);
          }
          if ( v21 )
            v18 = v21;
        }
LABEL_21:
        *v18 = v16;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v22]; k != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
