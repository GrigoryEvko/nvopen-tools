// Function: sub_F835C0
// Address: 0xf835c0
//
_QWORD *__fastcall sub_F835C0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 *v8; // r12
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rdx
  int v12; // edi
  __int64 v13; // rcx
  __int64 v14; // rsi
  int v15; // r10d
  _QWORD *v16; // r11
  int v17; // r9d
  unsigned int k; // eax
  _QWORD *v19; // rdi
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // rdi
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
  result = (_QWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v28 = 40 * v4;
    v8 = (__int64 *)(v5 + 40 * v4);
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
    if ( v8 != (__int64 *)v5 )
    {
      for ( j = (__int64 *)v5; v8 != j; j += 5 )
      {
        while ( 1 )
        {
          v11 = *j;
          if ( *j != -4096 )
            break;
          if ( j[1] == -4096 )
          {
LABEL_22:
            j += 5;
            if ( v8 == j )
              return (_QWORD *)sub_C7D6A0(v5, v28, 8);
          }
          else
          {
LABEL_12:
            v12 = *(_DWORD *)(a1 + 24);
            if ( !v12 )
            {
              MEMORY[0] = *j;
              BUG();
            }
            v13 = j[1];
            v14 = *(_QWORD *)(a1 + 8);
            v15 = 1;
            v16 = 0;
            v17 = v12 - 1;
            for ( k = (v12 - 1)
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                        | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
                     ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; k = v17 & v25 )
            {
              v19 = (_QWORD *)(v14 + 40LL * k);
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
                v16 = (_QWORD *)(v14 + 40LL * k);
              }
              v25 = v15 + k;
              ++v15;
            }
            *v19 = v11;
            v21 = j[1];
            v19[2] = 6;
            v19[1] = v21;
            v19[3] = 0;
            v22 = j[4];
            v19[4] = v22;
            if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
              sub_BD6050(v19 + 2, j[2] & 0xFFFFFFFFFFFFFFF8LL);
            ++*(_DWORD *)(a1 + 16);
            v23 = j[4];
            if ( v23 == 0 || v23 == -4096 || v23 == -8192 )
              goto LABEL_22;
            v24 = j + 2;
            j += 5;
            sub_BD60C0(v24);
            if ( v8 == j )
              return (_QWORD *)sub_C7D6A0(v5, v28, 8);
          }
        }
        if ( v11 != -8192 || j[1] != -8192 )
          goto LABEL_12;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v28, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[5 * v26]; m != result; result += 5 )
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
