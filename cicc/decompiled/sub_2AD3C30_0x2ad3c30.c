// Function: sub_2AD3C30
// Address: 0x2ad3c30
//
_QWORD *__fastcall sub_2AD3C30(__int64 a1, int a2)
{
  __int64 v3; // r13
  __int64 v4; // r12
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r9
  _QWORD *i; // rdx
  __int64 *v10; // rdx
  __int64 v11; // rcx
  int v12; // r8d
  __int64 v13; // rsi
  int v14; // r8d
  __int64 v15; // rdi
  __int64 *v16; // r14
  int v17; // r11d
  unsigned int j; // eax
  __int64 *v19; // r10
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rdx
  _QWORD *k; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(24LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = (__int64 *)(v4 + 24 * v3);
    for ( i = &result[3 * v7]; i != result; result += 3 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
    if ( v8 != (__int64 *)v4 )
    {
      v10 = (__int64 *)v4;
      while ( 1 )
      {
        v11 = *v10;
        if ( *v10 == -4096 )
        {
          if ( v10[1] != -4096 )
            goto LABEL_12;
          v10 += 3;
          if ( v8 == v10 )
            return (_QWORD *)sub_C7D6A0(v4, 24 * v3, 8);
        }
        else if ( v11 == -8192 && v10[1] == -8192 )
        {
          v10 += 3;
          if ( v8 == v10 )
            return (_QWORD *)sub_C7D6A0(v4, 24 * v3, 8);
        }
        else
        {
LABEL_12:
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *v10;
            BUG();
          }
          v13 = v10[1];
          v14 = v12 - 1;
          v16 = 0;
          v17 = 1;
          for ( j = v14
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                      | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; j = v14 & v22 )
          {
            v15 = *(_QWORD *)(a1 + 8);
            v19 = (__int64 *)(v15 + 24LL * j);
            v20 = *v19;
            if ( v11 == *v19 && v19[1] == v13 )
              break;
            if ( v20 == -4096 )
            {
              if ( v19[1] == -4096 )
              {
                if ( v16 )
                  v19 = v16;
                break;
              }
            }
            else if ( v20 == -8192 && v19[1] == -8192 && !v16 )
            {
              v16 = (__int64 *)(v15 + 24LL * j);
            }
            v22 = v17 + j;
            ++v17;
          }
          *v19 = v11;
          v21 = v10[1];
          v10 += 3;
          v19[1] = v21;
          v19[2] = *(v10 - 1);
          ++*(_DWORD *)(a1 + 16);
          if ( v8 == v10 )
            return (_QWORD *)sub_C7D6A0(v4, 24 * v3, 8);
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v4, 24 * v3, 8);
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
        result[1] = -4096;
      }
    }
  }
  return result;
}
