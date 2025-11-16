// Function: sub_15C64E0
// Address: 0x15c64e0
//
_QWORD *__fastcall sub_15C64E0(__int64 a1, int a2)
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
  __int64 v13; // r9
  int v14; // r15d
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rcx
  unsigned int v18; // eax
  _QWORD *v19; // rdx
  __int64 v20; // rsi
  int v21; // r10d
  _QWORD *v22; // rdi
  __int64 v23; // rdx
  _QWORD *k; // rdx
  __int64 v25; // [rsp+8h] [rbp-58h]
  int v26; // [rsp+10h] [rbp-50h] BYREF
  int v27; // [rsp+14h] [rbp-4Ch] BYREF
  __int64 v28; // [rsp+18h] [rbp-48h] BYREF
  __int64 v29[8]; // [rsp+20h] [rbp-40h] BYREF

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
        v13 = *(_QWORD *)(a1 + 8);
        v14 = v12 - 1;
        v26 = *(unsigned __int16 *)(v11 + 2);
        v25 = v13;
        v27 = *(_DWORD *)(v11 + 24);
        v15 = *(unsigned int *)(v11 + 8);
        v28 = *(_QWORD *)(v11 - 8 * v15);
        v29[0] = *(_QWORD *)(v11 + 8 * (1 - v15));
        v16 = sub_15B6820(&v26, &v27, &v28, v29);
        v17 = *j;
        v18 = v14 & v16;
        v19 = (_QWORD *)(v25 + 8LL * v18);
        v20 = *v19;
        if ( *v19 != *j )
        {
          v21 = 1;
          v22 = 0;
          while ( v20 != -8 )
          {
            if ( v20 != -16 || v22 )
              v19 = v22;
            v18 = v14 & (v21 + v18);
            v20 = *(_QWORD *)(v25 + 8LL * v18);
            if ( v17 == v20 )
            {
              v19 = (_QWORD *)(v25 + 8LL * v18);
              goto LABEL_21;
            }
            ++v21;
            v22 = v19;
            v19 = (_QWORD *)(v25 + 8LL * v18);
          }
          if ( v22 )
            v19 = v22;
        }
LABEL_21:
        *v19 = v17;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v23]; k != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
