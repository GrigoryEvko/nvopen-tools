// Function: sub_1644C70
// Address: 0x1644c70
//
_QWORD *__fastcall sub_1644C70(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // rcx
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // r15d
  _QWORD *v13; // rsi
  __int64 v14; // rdx
  int v15; // r15d
  int v16; // eax
  __int64 v17; // rsi
  _QWORD *v18; // r9
  unsigned int v19; // eax
  int v20; // r10d
  _QWORD *v21; // rdx
  __int64 v22; // rdi
  _QWORD *k; // rdx
  __int64 v24; // [rsp+0h] [rbp-70h]
  __int64 *v25; // [rsp+8h] [rbp-68h]
  unsigned __int64 v26; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v27[3]; // [rsp+20h] [rbp-50h] BYREF
  char v28[56]; // [rsp+38h] [rbp-38h] BYREF

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
    v8 = &v4[v3];
    *(_QWORD *)(a1 + 16) = 0;
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
        v25 = v8;
        if ( !v12 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v13 = *(_QWORD **)(v11 + 16);
        v24 = *(_QWORD *)(a1 + 8);
        v27[0] = *v13;
        v14 = *(unsigned int *)(v11 + 12);
        v27[1] = v13 + 1;
        v14 *= 8;
        v27[2] = (v14 - 8) >> 3;
        v28[0] = *(_DWORD *)(v11 + 8) >> 8 != 0;
        v15 = v12 - 1;
        v26 = sub_1644300(v13 + 1, (__int64)v13 + v14);
        v16 = sub_1644240(v27, (__int64 *)&v26, v28);
        v17 = *j;
        v18 = 0;
        v19 = v15 & v16;
        v8 = v25;
        v20 = 1;
        v21 = (_QWORD *)(v24 + 8LL * v19);
        v22 = *v21;
        if ( *j != *v21 )
        {
          while ( v22 != -8 )
          {
            if ( !v18 && v22 == -16 )
              v18 = v21;
            v19 = v15 & (v20 + v19);
            v21 = (_QWORD *)(v24 + 8LL * v19);
            v22 = *v21;
            if ( v17 == *v21 )
              goto LABEL_13;
            ++v20;
          }
          if ( v18 )
            v21 = v18;
        }
LABEL_13:
        *v21 = v17;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[*(unsigned int *)(a1 + 24)]; k != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
