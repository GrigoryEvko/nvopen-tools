// Function: sub_256DD30
// Address: 0x256dd30
//
_QWORD *__fastcall sub_256DD30(__int64 a1, int a2)
{
  unsigned int v3; // r13d
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
  __int64 v14; // rdi
  __int64 *v15; // r14
  int v16; // r10d
  int v17; // r11d
  unsigned int j; // eax
  __int64 *v19; // r8
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rdx
  _QWORD *k; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(16LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 16LL * v3;
    v8 = (__int64 *)(v4 + v25);
    for ( i = &result[2 * v7]; i != result; result += 2 )
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
          v10 += 2;
          if ( v8 == v10 )
            return (_QWORD *)sub_C7D6A0(v4, v25, 8);
        }
        else if ( v11 == -8192 && v10[1] == -8192 )
        {
          v10 += 2;
          if ( v8 == v10 )
            return (_QWORD *)sub_C7D6A0(v4, v25, 8);
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
          v15 = 0;
          v16 = v12 - 1;
          v17 = 1;
          for ( j = (v12 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                      | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; j = v16 & v22 )
          {
            v14 = *(_QWORD *)(a1 + 8);
            v19 = (__int64 *)(v14 + 16LL * j);
            v20 = *v19;
            if ( v11 == *v19 && v19[1] == v13 )
              break;
            if ( v20 == -4096 )
            {
              if ( v19[1] == -4096 )
              {
                if ( v15 )
                  v19 = v15;
                break;
              }
            }
            else if ( v20 == -8192 && v19[1] == -8192 && !v15 )
            {
              v15 = (__int64 *)(v14 + 16LL * j);
            }
            v22 = v17 + j;
            ++v17;
          }
          *v19 = v11;
          v21 = v10[1];
          v10 += 2;
          v19[1] = v21;
          ++*(_DWORD *)(a1 + 16);
          if ( v8 == v10 )
            return (_QWORD *)sub_C7D6A0(v4, v25, 8);
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v4, v25, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[2 * v23]; k != result; result += 2 )
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
