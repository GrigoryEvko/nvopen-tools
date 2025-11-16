// Function: sub_16453F0
// Address: 0x16453f0
//
_QWORD *__fastcall sub_16453F0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rcx
  __int64 *v8; // rdx
  _QWORD *i; // rcx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // r15d
  _QWORD *v13; // rdi
  int v14; // r15d
  __int64 v15; // rcx
  int v16; // eax
  int v17; // eax
  __int64 v18; // rdi
  _QWORD *v19; // r9
  unsigned int v20; // eax
  int v21; // r10d
  _QWORD *v22; // rsi
  __int64 v23; // r8
  _QWORD *k; // rdx
  __int64 v25; // [rsp+0h] [rbp-70h]
  __int64 *v26; // [rsp+8h] [rbp-68h]
  __int64 v27[2]; // [rsp+18h] [rbp-58h] BYREF
  __int64 v28; // [rsp+28h] [rbp-48h]
  char v29[64]; // [rsp+30h] [rbp-40h] BYREF

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
        v26 = v8;
        if ( !v12 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v13 = *(_QWORD **)(v11 + 16);
        v14 = v12 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v28 = *(unsigned int *)(v11 + 12);
        v16 = *(_DWORD *)(v11 + 8);
        v25 = v15;
        v27[1] = (__int64)v13;
        v29[0] = (v16 & 0x200) != 0;
        v27[0] = sub_1644300(v13, (__int64)&v13[v28]);
        v17 = sub_1644190(v27, v29);
        v18 = *j;
        v19 = 0;
        v20 = v14 & v17;
        v8 = v26;
        v21 = 1;
        v22 = (_QWORD *)(v25 + 8LL * v20);
        v23 = *v22;
        if ( *v22 != *j )
        {
          while ( v23 != -8 )
          {
            if ( !v19 && v23 == -16 )
              v19 = v22;
            v20 = v14 & (v21 + v20);
            v22 = (_QWORD *)(v25 + 8LL * v20);
            v23 = *v22;
            if ( *v22 == v18 )
              goto LABEL_13;
            ++v21;
          }
          if ( v19 )
            v22 = v19;
        }
LABEL_13:
        *v22 = v18;
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
