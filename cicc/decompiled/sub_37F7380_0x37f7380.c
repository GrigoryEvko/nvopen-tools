// Function: sub_37F7380
// Address: 0x37f7380
//
_DWORD *__fastcall sub_37F7380(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // r13
  _DWORD *i; // rdx
  _DWORD *v10; // rbx
  int v11; // edx
  int v12; // edi
  int v13; // ecx
  int v14; // edi
  __int64 v15; // rsi
  int v16; // r10d
  int *v17; // r9
  unsigned int j; // eax
  int *v19; // r14
  int v20; // r8d
  char *v21; // rax
  int v22; // eax
  void *v23; // rdi
  __int64 v24; // r8
  unsigned __int64 v25; // rdi
  __int64 v26; // r9
  const void *v27; // rsi
  size_t v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rdx
  _DWORD *k; // rdx
  __int64 v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+8h] [rbp-48h]
  int v34; // [rsp+14h] [rbp-3Ch]
  int v35; // [rsp+14h] [rbp-3Ch]
  __int64 v36; // [rsp+18h] [rbp-38h]

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
  result = (_DWORD *)sub_C7D670(72LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v36 = 72 * v4;
    v8 = v5 + 72 * v4;
    for ( i = &result[18 * *(unsigned int *)(a1 + 24)]; i != result; result += 18 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0x7FFFFFFF;
      }
    }
    v10 = (_DWORD *)(v5 + 24);
    if ( v8 != v5 )
    {
      while ( 1 )
      {
        v11 = *(v10 - 6);
        if ( v11 == -1 )
          break;
        if ( v11 != -2 || *(v10 - 5) != 0x80000000 )
          goto LABEL_11;
        v21 = (char *)(v10 + 18);
        if ( (_DWORD *)v8 == v10 + 12 )
          return (_DWORD *)sub_C7D6A0(v5, v36, 8);
LABEL_22:
        v10 = v21;
      }
      if ( *(v10 - 5) != 0x7FFFFFFF )
      {
LABEL_11:
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v13 = *(v10 - 5);
        v14 = v12 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = 0;
        for ( j = v14
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)(37 * v13) | ((unsigned __int64)(unsigned int)(37 * v11) << 32))) >> 31)
                 ^ (756364221 * v13)); ; j = v14 & v29 )
        {
          v19 = (int *)(v15 + 72LL * j);
          v20 = *v19;
          if ( v11 == *v19 && v13 == v19[1] )
            break;
          if ( v20 == -1 )
          {
            if ( v19[1] == 0x7FFFFFFF )
            {
              if ( v17 )
                v19 = v17;
              break;
            }
          }
          else if ( v20 == -2 && v19[1] == 0x80000000 && !v17 )
          {
            v17 = (int *)(v15 + 72LL * j);
          }
          v29 = v16 + j;
          ++v16;
        }
        *v19 = v11;
        v22 = *(v10 - 5);
        v23 = v19 + 6;
        *((_QWORD *)v19 + 1) = v19 + 6;
        v19[1] = v22;
        *((_QWORD *)v19 + 2) = 0xC00000000LL;
        v24 = (unsigned int)*(v10 - 2);
        if ( (_DWORD)v24 && v19 + 2 != v10 - 4 )
        {
          v26 = *((_QWORD *)v10 - 2);
          if ( v10 == (_DWORD *)v26 )
          {
            v27 = v10;
            v28 = 4LL * (unsigned int)v24;
            if ( (unsigned int)v24 <= 0xC )
              goto LABEL_33;
            v33 = *((_QWORD *)v10 - 2);
            v35 = *(v10 - 2);
            sub_C8D5F0((__int64)(v19 + 2), v19 + 6, (unsigned int)v24, 4u, v24, v26);
            v23 = (void *)*((_QWORD *)v19 + 1);
            v27 = (const void *)*((_QWORD *)v10 - 2);
            LODWORD(v24) = v35;
            v28 = 4LL * (unsigned int)*(v10 - 2);
            v26 = v33;
            if ( v28 )
            {
LABEL_33:
              v32 = v26;
              v34 = v24;
              memcpy(v23, v27, v28);
              v19[4] = v34;
              *(_DWORD *)(v32 - 8) = 0;
            }
            else
            {
              v19[4] = v35;
              *(_DWORD *)(v33 - 8) = 0;
            }
          }
          else
          {
            *((_QWORD *)v19 + 1) = v26;
            v19[4] = *(v10 - 2);
            v19[5] = *(v10 - 1);
            *((_QWORD *)v10 - 2) = v10;
            *(v10 - 1) = 0;
            *(v10 - 2) = 0;
          }
        }
        ++*(_DWORD *)(a1 + 16);
        v25 = *((_QWORD *)v10 - 2);
        if ( (_DWORD *)v25 != v10 )
          _libc_free(v25);
      }
      v21 = (char *)(v10 + 18);
      if ( (_DWORD *)v8 == v10 + 12 )
        return (_DWORD *)sub_C7D6A0(v5, v36, 8);
      goto LABEL_22;
    }
    return (_DWORD *)sub_C7D6A0(v5, v36, 8);
  }
  else
  {
    v30 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[18 * v30]; k != result; result += 18 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0x7FFFFFFF;
      }
    }
  }
  return result;
}
