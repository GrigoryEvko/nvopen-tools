// Function: sub_2D2E6F0
// Address: 0x2d2e6f0
//
_QWORD *__fastcall sub_2D2E6F0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // r12
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v10; // rcx
  int v11; // edi
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r8
  __int64 *v15; // r10
  __int64 v16; // r9
  unsigned int k; // eax
  __int64 *v18; // rdi
  __int64 v19; // r11
  __int64 v20; // rax
  unsigned __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rdx
  _QWORD *m; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(352LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 352 * v3;
    v7 = v4 + 352 * v3;
    for ( i = &result[44 * *(unsigned int *)(a1 + 24)]; i != result; result += 44 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
    if ( v7 != v4 )
    {
      for ( j = v4; v7 != j; j += 352 )
      {
        while ( 1 )
        {
          v10 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 )
            break;
          if ( *(_QWORD *)(j + 8) == -4096 )
          {
LABEL_22:
            j += 352;
            if ( v7 == j )
              return (_QWORD *)sub_C7D6A0(v4, v25, 8);
          }
          else
          {
LABEL_12:
            v11 = *(_DWORD *)(a1 + 24);
            if ( !v11 )
            {
              MEMORY[0] = *(_QWORD *)j;
              BUG();
            }
            v12 = *(_QWORD *)(j + 8);
            v13 = *(_QWORD *)(a1 + 8);
            v14 = (unsigned int)(v11 - 1);
            v15 = 0;
            v16 = 1;
            for ( k = v14
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)
                        | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))) >> 31)
                     ^ (484763065 * (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)))); ; k = v14 & v22 )
            {
              v18 = (__int64 *)(v13 + 352LL * k);
              v19 = *v18;
              if ( v10 == *v18 && v18[1] == v12 )
                break;
              if ( v19 == -4096 )
              {
                if ( v18[1] == -4096 )
                {
                  if ( v15 )
                    v18 = v15;
                  break;
                }
              }
              else if ( v19 == -8192 && v18[1] == -8192 && !v15 )
              {
                v15 = (__int64 *)(v13 + 352LL * k);
              }
              v22 = v16 + k;
              v16 = (unsigned int)(v16 + 1);
            }
            *v18 = v10;
            v20 = *(_QWORD *)(j + 8);
            v18[3] = 0x800000000LL;
            v18[1] = v20;
            v18[2] = (__int64)(v18 + 4);
            if ( *(_DWORD *)(j + 24) )
              sub_2D23140((__int64)(v18 + 2), (char **)(j + 16), v12, v10, v14, v16);
            ++*(_DWORD *)(a1 + 16);
            v21 = *(_QWORD *)(j + 16);
            if ( v21 == j + 32 )
              goto LABEL_22;
            _libc_free(v21);
            j += 352;
            if ( v7 == j )
              return (_QWORD *)sub_C7D6A0(v4, v25, 8);
          }
        }
        if ( v10 != -8192 || *(_QWORD *)(j + 8) != -8192 )
          goto LABEL_12;
      }
    }
    return (_QWORD *)sub_C7D6A0(v4, v25, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[44 * v23]; m != result; result += 44 )
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
