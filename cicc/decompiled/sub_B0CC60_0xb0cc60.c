// Function: sub_B0CC60
// Address: 0xb0cc60
//
_QWORD *__fastcall sub_B0CC60(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 *v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rcx
  __int64 *v9; // r14
  _QWORD *i; // rcx
  __int64 *j; // rbx
  __int64 v12; // rax
  int v13; // r15d
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
  __int64 v24; // [rsp+0h] [rbp-40h]
  __int64 v25; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(8LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v24 = 8 * v4;
    v9 = &v5[v4];
    for ( i = &result[v8]; i != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; v9 != j; ++j )
    {
      v12 = *j;
      if ( *j != -8192 && v12 != -4096 )
      {
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v14 = v13 - 1;
        v25 = *(_QWORD *)(a1 + 8);
        v15 = sub_AF66D0(*(__int64 **)(v12 + 16), *(_QWORD *)(v12 + 24));
        v16 = *j;
        v17 = v14 & v15;
        v18 = (_QWORD *)(v25 + 8LL * v17);
        v19 = *v18;
        if ( *j != *v18 )
        {
          v20 = 1;
          v21 = 0;
          while ( v19 != -4096 )
          {
            if ( v19 != -8192 || v21 )
              v18 = v21;
            v17 = v14 & (v20 + v17);
            v19 = *(_QWORD *)(v25 + 8LL * v17);
            if ( v19 == v16 )
            {
              v18 = (_QWORD *)(v25 + 8LL * v17);
              goto LABEL_21;
            }
            ++v20;
            v21 = v18;
            v18 = (_QWORD *)(v25 + 8LL * v17);
          }
          if ( v21 )
            v18 = v21;
        }
LABEL_21:
        *v18 = v16;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v24, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v22]; k != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
