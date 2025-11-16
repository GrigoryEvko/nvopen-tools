// Function: sub_35BD210
// Address: 0x35bd210
//
_QWORD *__fastcall sub_35BD210(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r15
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v12; // rcx
  int v13; // edi
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 *v16; // r10
  int v17; // r8d
  int v18; // r9d
  unsigned int k; // eax
  __int64 *v20; // rdi
  __int64 v21; // r11
  __int64 v22; // rax
  __int64 v23; // rax
  volatile signed __int32 *v24; // rdi
  unsigned int v25; // eax
  __int64 v26; // rdx
  _QWORD *m; // rdx
  __int64 v28; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v28 = 32 * v4;
    v9 = (__int64 *)(v5 + 32 * v4);
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
    if ( v9 != (__int64 *)v5 )
    {
      for ( j = (__int64 *)v5; v9 != j; j += 4 )
      {
        while ( 1 )
        {
          v12 = *j;
          if ( *j != -4096 )
            break;
          if ( j[1] == -4096 )
          {
LABEL_22:
            j += 4;
            if ( v9 == j )
              return (_QWORD *)sub_C7D6A0(v5, v28, 8);
          }
          else
          {
LABEL_12:
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
              MEMORY[0] = *j;
              BUG();
            }
            v14 = j[1];
            v15 = *(_QWORD *)(a1 + 8);
            v16 = 0;
            v17 = v13 - 1;
            v18 = 1;
            for ( k = (v13 - 1)
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)
                        | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))) >> 31)
                     ^ (484763065 * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)))); ; k = v17 & v25 )
            {
              v20 = (__int64 *)(v15 + 32LL * k);
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
                v16 = (__int64 *)(v15 + 32LL * k);
              }
              v25 = v18 + k;
              ++v18;
            }
            *v20 = v12;
            v20[1] = j[1];
            v22 = j[2];
            v20[3] = 0;
            v20[2] = v22;
            v23 = j[3];
            j[3] = 0;
            v20[3] = v23;
            j[2] = 0;
            ++*(_DWORD *)(a1 + 16);
            v24 = (volatile signed __int32 *)j[3];
            if ( !v24 )
              goto LABEL_22;
            sub_A191D0(v24);
            j += 4;
            if ( v9 == j )
              return (_QWORD *)sub_C7D6A0(v5, v28, 8);
          }
        }
        if ( v12 != -8192 || j[1] != -8192 )
          goto LABEL_12;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v28, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[4 * v26]; m != result; result += 4 )
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
