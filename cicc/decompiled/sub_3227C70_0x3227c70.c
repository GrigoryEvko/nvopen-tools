// Function: sub_3227C70
// Address: 0x3227c70
//
_QWORD *__fastcall sub_3227C70(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r12
  _QWORD *i; // rdx
  char *v10; // rbx
  char *v11; // rax
  __int64 v12; // rax
  int v13; // edx
  int v14; // edx
  __int64 v15; // rdi
  int v16; // r11d
  unsigned int v17; // ecx
  __int64 *v18; // r8
  __int64 *v19; // r14
  __int64 v20; // rsi
  void *v21; // rdi
  __int64 v22; // r8
  unsigned __int64 v23; // rdi
  char *v24; // rax
  char *v25; // rsi
  size_t v26; // rdx
  __int64 v27; // rdx
  _QWORD *j; // rdx
  int v29; // [rsp+4h] [rbp-3Ch]
  int v30; // [rsp+4h] [rbp-3Ch]
  __int64 v31; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(88LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v31 = 88 * v4;
    v8 = v5 + 88 * v4;
    for ( i = &result[11 * *(unsigned int *)(a1 + 24)]; i != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
    v10 = (char *)(v5 + 72);
    if ( v8 != v5 )
    {
      while ( 1 )
      {
        v12 = *((_QWORD *)v10 - 9);
        if ( v12 == -8192 || v12 == -4096 )
          goto LABEL_10;
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = *((_QWORD *)v10 - 9);
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = v14 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v18 = 0;
        v19 = (__int64 *)(v15 + 88LL * v17);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          while ( v20 != -4096 )
          {
            if ( !v18 && v20 == -8192 )
              v18 = v19;
            v17 = v14 & (v16 + v17);
            v19 = (__int64 *)(v15 + 88LL * v17);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_16;
            ++v16;
          }
          if ( v18 )
            v19 = v18;
        }
LABEL_16:
        *v19 = v12;
        sub_C8CF70((__int64)(v19 + 1), v19 + 5, 2, (__int64)(v10 - 32), (__int64)(v10 - 64));
        v21 = v19 + 9;
        v19[7] = (__int64)(v19 + 9);
        v19[8] = 0x200000000LL;
        v22 = *((unsigned int *)v10 - 2);
        if ( (_DWORD)v22 && v19 + 7 != (__int64 *)(v10 - 16) )
        {
          v24 = (char *)*((_QWORD *)v10 - 2);
          if ( v24 == v10 )
          {
            v25 = v10;
            v26 = 8LL * (unsigned int)v22;
            if ( (unsigned int)v22 <= 2
              || (v30 = *((_DWORD *)v10 - 2),
                  sub_C8D5F0((__int64)(v19 + 7), v19 + 9, (unsigned int)v22, 8u, v22, (unsigned int)v22),
                  v21 = (void *)v19[7],
                  v25 = (char *)*((_QWORD *)v10 - 2),
                  LODWORD(v22) = v30,
                  (v26 = 8LL * *((unsigned int *)v10 - 2)) != 0) )
            {
              v29 = v22;
              memcpy(v21, v25, v26);
              *((_DWORD *)v19 + 16) = v29;
              *((_DWORD *)v10 - 2) = 0;
            }
            else
            {
              *((_DWORD *)v19 + 16) = v30;
              *((_DWORD *)v10 - 2) = 0;
            }
          }
          else
          {
            v19[7] = (__int64)v24;
            *((_DWORD *)v19 + 16) = *((_DWORD *)v10 - 2);
            *((_DWORD *)v19 + 17) = *((_DWORD *)v10 - 1);
            *((_QWORD *)v10 - 2) = v10;
            *((_DWORD *)v10 - 1) = 0;
            *((_DWORD *)v10 - 2) = 0;
          }
        }
        ++*(_DWORD *)(a1 + 16);
        v23 = *((_QWORD *)v10 - 2);
        if ( (char *)v23 != v10 )
          _libc_free(v23);
        if ( *(v10 - 36) )
        {
LABEL_10:
          v11 = v10 + 88;
          if ( (char *)v8 == v10 + 16 )
            return (_QWORD *)sub_C7D6A0(v5, v31, 8);
        }
        else
        {
          _libc_free(*((_QWORD *)v10 - 7));
          v11 = v10 + 88;
          if ( (char *)v8 == v10 + 16 )
            return (_QWORD *)sub_C7D6A0(v5, v31, 8);
        }
        v10 = v11;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v31, 8);
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[11 * v27]; j != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
