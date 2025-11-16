// Function: sub_313A720
// Address: 0x313a720
//
_QWORD *__fastcall sub_313A720(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r8
  _QWORD *i; // rdx
  _QWORD *v11; // rdx
  __int64 v12; // rcx
  int v13; // r9d
  int v14; // r9d
  __int64 v15; // rdi
  __int64 *v16; // r14
  int v17; // r13d
  unsigned int j; // eax
  __int64 *v19; // r11
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rdx
  _QWORD *k; // rdx
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
  result = (_QWORD *)sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 24 * v4;
    v9 = (_QWORD *)(v5 + 24 * v4);
    for ( i = &result[3 * v8]; i != result; result += 3 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -1;
      }
    }
    if ( v9 != (_QWORD *)v5 )
    {
      v11 = (_QWORD *)v5;
      while ( 1 )
      {
        v12 = *v11;
        if ( *v11 == -4096 )
        {
          if ( v11[1] != -1 )
            goto LABEL_12;
          v11 += 3;
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, v25, 8);
        }
        else if ( v12 == -8192 && v11[1] == -2 )
        {
          v11 += 3;
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, v25, 8);
        }
        else
        {
LABEL_12:
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *v11;
            BUG();
          }
          v14 = v13 - 1;
          v16 = 0;
          v17 = 1;
          for ( j = v14
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)((0xBF58476D1CE4E5B9LL * v11[1]) >> 31) ^ (484763065 * *((_DWORD *)v11 + 2))
                      | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v11[1]) >> 31) ^ (484763065 * *((_DWORD *)v11 + 2)))));
                ;
                j = v14 & v22 )
          {
            v15 = *(_QWORD *)(a1 + 8);
            v19 = (__int64 *)(v15 + 24LL * j);
            v20 = *v19;
            if ( v12 == *v19 && v19[1] == v11[1] )
              break;
            if ( v20 == -4096 )
            {
              if ( v19[1] == -1 )
              {
                if ( v16 )
                  v19 = v16;
                break;
              }
            }
            else if ( v20 == -8192 && v19[1] == -2 && !v16 )
            {
              v16 = (__int64 *)(v15 + 24LL * j);
            }
            v22 = v17 + j;
            ++v17;
          }
          *v19 = v12;
          v21 = v11[1];
          v11 += 3;
          v19[1] = v21;
          v19[2] = *(v11 - 1);
          ++*(_DWORD *)(a1 + 16);
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, v25, 8);
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v25, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[3 * v23]; k != result; result += 3 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -1;
      }
    }
  }
  return result;
}
