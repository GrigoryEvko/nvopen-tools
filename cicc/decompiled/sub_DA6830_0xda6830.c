// Function: sub_DA6830
// Address: 0xda6830
//
_QWORD *__fastcall sub_DA6830(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v12; // rcx
  int v13; // edi
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 *v16; // r10
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int k; // eax
  __int64 *v20; // rdi
  __int64 v21; // r11
  __int64 v22; // rax
  __int64 v23; // rdi
  int v24; // eax
  __int64 v25; // rdx
  _QWORD *m; // rdx
  __int64 v27; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670((unsigned __int64)v6 << 6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v27 = v4 << 6;
    v9 = v5 + (v4 << 6);
    for ( i = &result[8 * v8]; i != result; result += 8 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
    if ( v9 != v5 )
    {
      for ( j = v5; v9 != j; j += 64 )
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 )
            break;
          if ( *(_QWORD *)(j + 8) == -4096 )
          {
LABEL_22:
            j += 64;
            if ( v9 == j )
              return (_QWORD *)sub_C7D6A0(v5, v27, 8);
          }
          else
          {
LABEL_12:
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
              MEMORY[0] = *(_QWORD *)j;
              BUG();
            }
            v14 = *(_QWORD *)(j + 8);
            v15 = *(_QWORD *)(a1 + 8);
            v16 = 0;
            v17 = (unsigned int)(v13 - 1);
            v18 = 1;
            for ( k = v17
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)
                        | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))) >> 31)
                     ^ (484763065 * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)))); ; k = v17 & v24 )
            {
              v20 = (__int64 *)(v15 + ((unsigned __int64)k << 6));
              v21 = *v20;
              if ( v12 == *v20 && v20[1] == v14 )
                break;
              if ( v21 == -4096 )
              {
                if ( v20[1] == -4096 )
                {
                  if ( v16 )
                    v20 = v16;
                  break;
                }
              }
              else if ( v21 == -8192 && v20[1] == -8192 && !v16 )
              {
                v16 = (__int64 *)(v15 + ((unsigned __int64)k << 6));
              }
              v24 = v18 + k;
              v18 = (unsigned int)(v18 + 1);
            }
            *v20 = v12;
            v20[1] = *(_QWORD *)(j + 8);
            v22 = *(_QWORD *)(j + 16);
            v20[4] = 0x300000000LL;
            v20[2] = v22;
            v20[3] = (__int64)(v20 + 5);
            if ( *(_DWORD *)(j + 32) )
            {
              v15 = j + 24;
              sub_D91460((__int64)(v20 + 3), (char **)(j + 24), v14, v12, v17, v18);
            }
            ++*(_DWORD *)(a1 + 16);
            v23 = *(_QWORD *)(j + 24);
            if ( v23 == j + 40 )
              goto LABEL_22;
            _libc_free(v23, v15);
            j += 64;
            if ( v9 == j )
              return (_QWORD *)sub_C7D6A0(v5, v27, 8);
          }
        }
        if ( v12 != -8192 || *(_QWORD *)(j + 8) != -8192 )
          goto LABEL_12;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v27, 8);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[8 * v25]; m != result; result += 8 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
  }
  return result;
}
