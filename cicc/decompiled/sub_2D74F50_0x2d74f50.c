// Function: sub_2D74F50
// Address: 0x2d74f50
//
_QWORD *__fastcall sub_2D74F50(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // r13d
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r9
  _QWORD *i; // rdx
  __int64 *v11; // rdx
  __int64 v12; // rcx
  int v13; // r8d
  __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 *v16; // r14
  int v17; // r10d
  int v18; // r11d
  unsigned int j; // eax
  __int64 *v20; // r8
  __int64 v21; // r15
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rdx
  _QWORD *k; // rdx
  __int64 v26; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
    v26 = 16LL * v4;
    v9 = (__int64 *)(v5 + v26);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
    if ( v9 != (__int64 *)v5 )
    {
      v11 = (__int64 *)v5;
      while ( 1 )
      {
        v12 = *v11;
        if ( *v11 == -4096 )
        {
          if ( v11[1] != -4096 )
            goto LABEL_12;
          v11 += 2;
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, v26, 8);
        }
        else if ( v12 == -8192 && v11[1] == -8192 )
        {
          v11 += 2;
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, v26, 8);
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
          v14 = v11[1];
          v16 = 0;
          v17 = v13 - 1;
          v18 = 1;
          for ( j = (v13 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)
                      | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)))); ; j = v17 & v23 )
          {
            v15 = *(_QWORD *)(a1 + 8);
            v20 = (__int64 *)(v15 + 16LL * j);
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
              v16 = (__int64 *)(v15 + 16LL * j);
            }
            v23 = v18 + j;
            ++v18;
          }
          *v20 = v12;
          v22 = v11[1];
          v11 += 2;
          v20[1] = v22;
          ++*(_DWORD *)(a1 + 16);
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, v26, 8);
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v26, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[2 * v24]; k != result; result += 2 )
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
