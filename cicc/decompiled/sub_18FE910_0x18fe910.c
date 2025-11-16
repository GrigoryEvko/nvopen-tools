// Function: sub_18FE910
// Address: 0x18fe910
//
_QWORD *__fastcall sub_18FE910(__int64 a1, int a2)
{
  __int64 v3; // r12
  __int64 *v4; // r14
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r12
  _QWORD *i; // rdx
  __int64 v10; // rax
  __int64 *v11; // rbx
  __int64 *v12; // r13
  __int64 v13; // r14
  __int64 v14; // rdi
  int v15; // r15d
  int v16; // eax
  int v17; // edx
  __int64 *v18; // r8
  unsigned int j; // ecx
  __int64 v20; // rdi
  __int64 *v21; // r15
  __int64 v22; // rsi
  bool v23; // r9
  bool v24; // r10
  char v25; // al
  unsigned int v26; // ecx
  __int64 v27; // rdx
  _QWORD *k; // rdx
  __int64 *v29; // [rsp+0h] [rbp-50h]
  unsigned int v30; // [rsp+Ch] [rbp-44h]
  int v31; // [rsp+10h] [rbp-40h]
  int v32; // [rsp+14h] [rbp-3Ch]
  __int64 v33; // [rsp+18h] [rbp-38h]

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
    if ( v8 != v4 )
    {
      v10 = a1;
      v11 = v4;
      v12 = v4;
      v13 = v10;
      while ( 1 )
      {
        v14 = *v11;
        if ( *v11 == -8 || v14 == -16 )
          goto LABEL_18;
        v15 = *(_DWORD *)(v13 + 24);
        if ( !v15 )
        {
          MEMORY[0] = *v11;
          BUG();
        }
        v33 = *(_QWORD *)(v13 + 8);
        v16 = sub_18FE780(v14);
        v17 = v15 - 1;
        v32 = 1;
        v18 = 0;
        for ( j = (v15 - 1) & v16; ; j = v17 & v26 )
        {
          v20 = *v11;
          v21 = (__int64 *)(v33 + 16LL * j);
          v22 = *v21;
          v23 = *v21 == -8;
          v24 = *v21 == -16;
          if ( v23 || *v11 == -8 || *v11 == -16 || *v21 == -16 )
          {
            if ( v22 == v20 )
              goto LABEL_17;
          }
          else
          {
            v29 = v18;
            v30 = j;
            v31 = v17;
            v25 = sub_15F41F0(v20, v22);
            v17 = v31;
            j = v30;
            v18 = v29;
            if ( v25 )
              goto LABEL_17;
            v22 = *v21;
            v24 = *v21 == -16;
            v23 = *v21 == -8;
          }
          if ( v23 || v24 )
            break;
LABEL_25:
          v26 = v32 + j;
          ++v32;
        }
        if ( v22 != -8 )
          break;
        if ( v18 )
          v21 = v18;
LABEL_17:
        *v21 = *v11;
        v21[1] = v11[1];
        ++*(_DWORD *)(v13 + 16);
LABEL_18:
        v11 += 2;
        if ( v8 == v11 )
        {
          v4 = v12;
          return (_QWORD *)j___libc_free_0(v4);
        }
      }
      if ( !v18 && v24 )
        v18 = v21;
      goto LABEL_25;
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[2 * v27]; k != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
