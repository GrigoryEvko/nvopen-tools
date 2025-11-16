// Function: sub_28F0570
// Address: 0x28f0570
//
_QWORD *__fastcall sub_28F0570(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r12
  _QWORD *i; // rdx
  __int64 j; // rbx
  unsigned __int64 v11; // rdx
  int v12; // edi
  __int64 v13; // rcx
  int v14; // edi
  __int64 v15; // rsi
  unsigned __int64 *v16; // r10
  int v17; // r9d
  unsigned int k; // eax
  unsigned __int64 *v19; // r15
  unsigned __int64 v20; // r8
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // rdx
  _QWORD *m; // rdx
  __int64 v30; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(72LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v30 = 72 * v4;
    v8 = v5 + 72 * v4;
    for ( i = &result[9 * *(unsigned int *)(a1 + 24)]; i != result; result += 9 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
    if ( v8 != v5 )
    {
      for ( j = v5; v8 != j; j += 72 )
      {
        while ( 1 )
        {
          v11 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 )
            break;
          if ( *(_QWORD *)(j + 8) == -4096 )
          {
LABEL_22:
            j += 72;
            if ( v8 == j )
              return (_QWORD *)sub_C7D6A0(v5, v30, 8);
          }
          else
          {
LABEL_12:
            v12 = *(_DWORD *)(a1 + 24);
            if ( !v12 )
            {
              MEMORY[0] = *(_QWORD *)j;
              BUG();
            }
            v13 = *(_QWORD *)(j + 8);
            v14 = v12 - 1;
            v15 = *(_QWORD *)(a1 + 8);
            v16 = 0;
            v17 = 1;
            for ( k = v14
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                        | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
                     ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; k = v14 & v27 )
            {
              v19 = (unsigned __int64 *)(v15 + 72LL * k);
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
                v16 = (unsigned __int64 *)(v15 + 72LL * k);
              }
              v27 = v17 + k;
              ++v17;
            }
            *v19 = v11;
            v21 = *(_QWORD *)(j + 8);
            v19[2] = 4;
            v19[1] = v21;
            v19[3] = 0;
            v22 = *(_QWORD *)(j + 32);
            v19[4] = v22;
            if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
              sub_BD6050(v19 + 2, *(_QWORD *)(j + 16) & 0xFFFFFFFFFFFFFFF8LL);
            v19[5] = 4;
            v19[6] = 0;
            v23 = *(_QWORD *)(j + 56);
            v19[7] = v23;
            if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
              sub_BD6050(v19 + 5, *(_QWORD *)(j + 40) & 0xFFFFFFFFFFFFFFF8LL);
            *((_DWORD *)v19 + 16) = *(_DWORD *)(j + 64);
            ++*(_DWORD *)(a1 + 16);
            v24 = *(_QWORD *)(j + 56);
            if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
              sub_BD60C0((_QWORD *)(j + 40));
            v25 = *(_QWORD *)(j + 32);
            if ( v25 == -4096 || v25 == 0 || v25 == -8192 )
              goto LABEL_22;
            v26 = (_QWORD *)(j + 16);
            j += 72;
            sub_BD60C0(v26);
            if ( v8 == j )
              return (_QWORD *)sub_C7D6A0(v5, v30, 8);
          }
        }
        if ( v11 != -8192 || *(_QWORD *)(j + 8) != -8192 )
          goto LABEL_12;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v30, 8);
  }
  else
  {
    v28 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[9 * v28]; m != result; result += 9 )
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
