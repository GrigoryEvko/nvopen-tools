// Function: sub_1058DA0
// Address: 0x1058da0
//
_QWORD *__fastcall sub_1058DA0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r13
  _QWORD *i; // rdx
  __int64 *v11; // rbx
  __int64 v12; // rax
  int v13; // edx
  int v14; // ecx
  __int64 v15; // rdi
  int v16; // r10d
  _QWORD *v17; // r9
  unsigned int v18; // esi
  _QWORD *v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r14
  __int64 v22; // rsi
  __int64 v23; // rdx
  _QWORD *j; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
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
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 16 * v4;
    v9 = (__int64 *)(v5 + 16 * v4);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v9 != (__int64 *)v5 )
    {
      v11 = (__int64 *)v5;
      do
      {
        v12 = *v11;
        if ( *v11 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *v11;
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = (_QWORD *)(v15 + 16LL * v18);
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -4096 )
            {
              if ( v20 == -8192 && !v17 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = (_QWORD *)(v15 + 16LL * v18);
              v20 = *v19;
              if ( v12 == *v19 )
                goto LABEL_14;
              ++v16;
            }
            if ( v17 )
              v19 = v17;
          }
LABEL_14:
          *v19 = v12;
          v19[1] = v11[1];
          v11[1] = 0;
          ++*(_DWORD *)(a1 + 16);
          v21 = v11[1];
          if ( v21 )
          {
            v22 = 16LL * *(unsigned int *)(v21 + 152);
            sub_C7D6A0(*(_QWORD *)(v21 + 136), v22, 8);
            if ( !*(_BYTE *)(v21 + 92) )
              _libc_free(*(_QWORD *)(v21 + 72), v22);
            if ( !*(_BYTE *)(v21 + 28) )
              _libc_free(*(_QWORD *)(v21 + 8), v22);
            j_j___libc_free_0(v21, 160);
          }
        }
        v11 += 2;
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v25, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v23]; j != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
