// Function: sub_AD1D70
// Address: 0xad1d70
//
_QWORD *__fastcall sub_AD1D70(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 *v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  _QWORD *v10; // rdx
  __int64 *i; // r15
  __int64 *j; // rbx
  __int64 v13; // rdi
  int v14; // r14d
  int v15; // r14d
  int v16; // eax
  __int64 v17; // rcx
  __int64 *v18; // r9
  unsigned int v19; // eax
  int v20; // r10d
  __int64 *v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rdx
  _QWORD *k; // rdx
  __int64 v25; // [rsp+0h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
    v9 = v4;
    v10 = &result[v8];
    for ( i = &v5[v9]; v10 != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; i != j; ++j )
    {
      v13 = *j;
      if ( *j != -8192 && v13 != -4096 )
      {
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v15 = v14 - 1;
        v25 = *(_QWORD *)(a1 + 8);
        v16 = sub_ACF550(v13);
        v17 = *j;
        v18 = 0;
        v19 = v15 & v16;
        v20 = 1;
        v21 = (__int64 *)(v25 + 8LL * v19);
        v22 = *v21;
        if ( *v21 != *j )
        {
          while ( v22 != -4096 )
          {
            if ( !v18 && v22 == -8192 )
              v18 = v21;
            v19 = v15 & (v20 + v19);
            v21 = (__int64 *)(v25 + 8LL * v19);
            v22 = *v21;
            if ( *v21 == v17 )
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
    return (_QWORD *)sub_C7D6A0(v5, v9 * 8, 8);
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
