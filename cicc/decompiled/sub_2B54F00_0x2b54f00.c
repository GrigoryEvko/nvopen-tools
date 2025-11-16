// Function: sub_2B54F00
// Address: 0x2b54f00
//
_QWORD *__fastcall sub_2B54F00(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  char *v8; // r14
  _QWORD *i; // rdx
  char *v10; // rbx
  __int64 v11; // rdx
  int v12; // r8d
  __int64 v13; // rcx
  int v14; // r8d
  __int64 v15; // rsi
  __int64 *v16; // r9
  int v17; // edi
  unsigned int j; // eax
  __int64 *v19; // r12
  __int64 v20; // r10
  char *v21; // rax
  __int64 v22; // rax
  void *v23; // rdi
  __int64 v24; // r8
  unsigned __int64 v25; // rdi
  __int64 v26; // r9
  char *v27; // rsi
  size_t v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rdx
  _QWORD *k; // rdx
  __int64 v32; // [rsp+0h] [rbp-50h]
  __int64 v33; // [rsp+0h] [rbp-50h]
  int v34; // [rsp+Ch] [rbp-44h]
  int v35; // [rsp+Ch] [rbp-44h]
  __int64 v36; // [rsp+10h] [rbp-40h]
  __int64 v37; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v37 = v5;
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
  result = (_QWORD *)sub_C7D670(80LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v36 = 80 * v4;
    v8 = (char *)(80 * v4 + v5);
    for ( i = &result[10 * *(unsigned int *)(a1 + 24)]; i != result; result += 10 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -4096;
      }
    }
    v10 = (char *)(v37 + 32);
    if ( v8 != (char *)v37 )
    {
      while ( 1 )
      {
        v11 = *((_QWORD *)v10 - 4);
        if ( v11 == -1 )
          break;
        if ( v11 != -2 || *((_QWORD *)v10 - 3) != -8192 )
          goto LABEL_11;
        v21 = v10 + 80;
        if ( v8 == v10 + 48 )
          return (_QWORD *)sub_C7D6A0(v37, v36, 8);
LABEL_22:
        v10 = v21;
      }
      if ( *((_QWORD *)v10 - 3) != -4096 )
      {
LABEL_11:
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v13 = *((_QWORD *)v10 - 3);
        v14 = v12 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 0;
        v17 = 1;
        for ( j = v14
                & (((0xBF58476D1CE4E5B9LL
                   * (((((0xBF58476D1CE4E5B9LL * v11) >> 31) ^ (0xBF58476D1CE4E5B9LL * v11)) << 32)
                    | ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4))) >> 31)
                 ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; j = v14 & v29 )
        {
          v19 = (__int64 *)(v15 + 80LL * j);
          v20 = *v19;
          if ( v11 == *v19 && v19[1] == v13 )
            break;
          if ( v20 == -1 )
          {
            if ( v19[1] == -4096 )
            {
              if ( v16 )
                v19 = v16;
              break;
            }
          }
          else if ( v20 == -2 && v19[1] == -8192 && !v16 )
          {
            v16 = (__int64 *)(v15 + 80LL * j);
          }
          v29 = v17 + j;
          ++v17;
        }
        *v19 = v11;
        v22 = *((_QWORD *)v10 - 3);
        v23 = v19 + 4;
        v19[2] = (__int64)(v19 + 4);
        v19[1] = v22;
        v19[3] = 0x600000000LL;
        v24 = *((unsigned int *)v10 - 2);
        if ( (_DWORD)v24 && v19 + 2 != (__int64 *)(v10 - 16) )
        {
          v26 = *((_QWORD *)v10 - 2);
          if ( v10 == (char *)v26 )
          {
            v27 = v10;
            v28 = 8LL * (unsigned int)v24;
            if ( (unsigned int)v24 <= 6 )
              goto LABEL_33;
            v33 = *((_QWORD *)v10 - 2);
            v35 = *((_DWORD *)v10 - 2);
            sub_C8D5F0((__int64)(v19 + 2), v19 + 4, (unsigned int)v24, 8u, v24, v26);
            v23 = (void *)v19[2];
            v27 = (char *)*((_QWORD *)v10 - 2);
            LODWORD(v24) = v35;
            v28 = 8LL * *((unsigned int *)v10 - 2);
            v26 = v33;
            if ( v28 )
            {
LABEL_33:
              v32 = v26;
              v34 = v24;
              memcpy(v23, v27, v28);
              *((_DWORD *)v19 + 6) = v34;
              *(_DWORD *)(v32 - 8) = 0;
            }
            else
            {
              *((_DWORD *)v19 + 6) = v35;
              *(_DWORD *)(v33 - 8) = 0;
            }
          }
          else
          {
            v19[2] = v26;
            *((_DWORD *)v19 + 6) = *((_DWORD *)v10 - 2);
            *((_DWORD *)v19 + 7) = *((_DWORD *)v10 - 1);
            *((_QWORD *)v10 - 2) = v10;
            *((_DWORD *)v10 - 1) = 0;
            *((_DWORD *)v10 - 2) = 0;
          }
        }
        ++*(_DWORD *)(a1 + 16);
        v25 = *((_QWORD *)v10 - 2);
        if ( (char *)v25 != v10 )
          _libc_free(v25);
      }
      v21 = v10 + 80;
      if ( v8 == v10 + 48 )
        return (_QWORD *)sub_C7D6A0(v37, v36, 8);
      goto LABEL_22;
    }
    return (_QWORD *)sub_C7D6A0(v37, v36, 8);
  }
  else
  {
    v30 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[10 * v30]; k != result; result += 10 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -4096;
      }
    }
  }
  return result;
}
