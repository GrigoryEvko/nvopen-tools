// Function: sub_15B9A30
// Address: 0x15b9a30
//
_QWORD *__fastcall sub_15B9A30(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r15
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // r14d
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // rdx
  int v16; // r14d
  int v17; // eax
  __int64 v18; // rcx
  unsigned int v19; // eax
  _QWORD *v20; // rdx
  __int64 v21; // rsi
  int v22; // r9d
  _QWORD *v23; // rdi
  __int64 v24; // rdx
  _QWORD *k; // rdx
  __int64 v26; // [rsp+8h] [rbp-58h]
  int v27; // [rsp+10h] [rbp-50h] BYREF
  int v28; // [rsp+14h] [rbp-4Ch] BYREF
  __int64 v29; // [rsp+18h] [rbp-48h] BYREF
  __int64 v30[8]; // [rsp+20h] [rbp-40h] BYREF

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
        v27 = *(_DWORD *)(v11 + 4);
        v28 = *(unsigned __int16 *)(v11 + 2);
        v14 = *(unsigned int *)(v11 + 8);
        v29 = *(_QWORD *)(v11 - 8 * v14);
        v15 = 0;
        if ( (_DWORD)v14 == 2 )
          v15 = *(_QWORD *)(v11 - 8);
        v30[0] = v15;
        v16 = v12 - 1;
        v26 = v13;
        v17 = sub_15B3100(&v27, &v28, &v29, v30);
        v18 = *j;
        v19 = v16 & v17;
        v20 = (_QWORD *)(v26 + 8LL * v19);
        v21 = *v20;
        if ( *v20 != *j )
        {
          v22 = 1;
          v23 = 0;
          while ( v21 != -8 )
          {
            if ( v21 != -16 || v23 )
              v20 = v23;
            v19 = v16 & (v22 + v19);
            v21 = *(_QWORD *)(v26 + 8LL * v19);
            if ( v21 == v18 )
            {
              v20 = (_QWORD *)(v26 + 8LL * v19);
              goto LABEL_23;
            }
            ++v22;
            v23 = v20;
            v20 = (_QWORD *)(v26 + 8LL * v19);
          }
          if ( v23 )
            v20 = v23;
        }
LABEL_23:
        *v20 = v18;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v24]; k != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
