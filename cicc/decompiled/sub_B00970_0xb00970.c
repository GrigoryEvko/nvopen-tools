// Function: sub_B00970
// Address: 0xb00970
//
_QWORD *__fastcall sub_B00970(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 *v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 *v10; // r15
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v13; // rax
  int v14; // r14d
  int v15; // r14d
  int v16; // eax
  __int64 v17; // rcx
  _QWORD *v18; // rdi
  unsigned int v19; // eax
  int v20; // r10d
  _QWORD *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdx
  _QWORD *k; // rdx
  __int64 v25; // [rsp+0h] [rbp-40h]
  __int64 v26; // [rsp+8h] [rbp-38h]

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
    v9 = 8 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = &v5[v4];
    for ( i = &result[v8]; i != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; v10 != j; ++j )
    {
      v13 = *j;
      if ( *j != -8192 && v13 != -4096 )
      {
        v14 = *(_DWORD *)(a1 + 24);
        v26 = v9;
        if ( !v14 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v15 = v14 - 1;
        v25 = *(_QWORD *)(a1 + 8);
        v16 = sub_AF6940(*(__int64 **)(v13 + 136), *(_QWORD *)(v13 + 136) + 8LL * *(unsigned int *)(v13 + 144));
        v17 = *j;
        v18 = 0;
        v19 = v15 & v16;
        v9 = v26;
        v20 = 1;
        v21 = (_QWORD *)(v25 + 8LL * v19);
        v22 = *v21;
        if ( *j != *v21 )
        {
          while ( v22 != -4096 )
          {
            if ( !v18 && v22 == -8192 )
              v18 = v21;
            v19 = v15 & (v20 + v19);
            v21 = (_QWORD *)(v25 + 8LL * v19);
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
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v23]; k != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
