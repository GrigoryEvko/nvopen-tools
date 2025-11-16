// Function: sub_2A11010
// Address: 0x2a11010
//
_QWORD *__fastcall sub_2A11010(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r9
  __int64 v9; // r14
  _QWORD *i; // rdx
  _QWORD *v11; // rbx
  __int64 v12; // rax
  int v13; // edx
  int v14; // edx
  __int64 v15; // rdi
  int v16; // r11d
  __int64 *v17; // r10
  unsigned int v18; // ecx
  __int64 *v19; // r12
  __int64 v20; // rsi
  void *v21; // rdi
  __int64 v22; // rax
  unsigned int v23; // r10d
  unsigned __int64 v24; // rdi
  _QWORD *v25; // rax
  const void *v26; // rsi
  size_t v27; // rdx
  __int64 v28; // rdx
  _QWORD *j; // rdx
  unsigned int v30; // [rsp+4h] [rbp-3Ch]
  unsigned int v31; // [rsp+4h] [rbp-3Ch]
  __int64 v32; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(96LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v32 = 96 * v4;
    v9 = v5 + 96 * v4;
    for ( i = &result[12 * *(unsigned int *)(a1 + 24)]; i != result; result += 12 )
    {
      if ( result )
        *result = -4096;
    }
    v11 = (_QWORD *)(v5 + 48);
    if ( v9 != v5 )
    {
      while ( 1 )
      {
        v12 = *(v11 - 6);
        if ( v12 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *(v11 - 6);
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = v14 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = (__int64 *)(v15 + 96LL * v18);
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -4096 )
            {
              if ( !v17 && v20 == -8192 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = (__int64 *)(v15 + 96LL * v18);
              v20 = *v19;
              if ( v12 == *v19 )
                goto LABEL_15;
              ++v16;
            }
            if ( v17 )
              v19 = v17;
          }
LABEL_15:
          *v19 = v12;
          v21 = v19 + 6;
          *((_DWORD *)v19 + 2) = *((_DWORD *)v11 - 10);
          *((_DWORD *)v19 + 3) = *((_DWORD *)v11 - 9);
          *((_DWORD *)v19 + 4) = *((_DWORD *)v11 - 8);
          *((_BYTE *)v19 + 20) = *((_BYTE *)v11 - 28);
          v22 = *(v11 - 3);
          v19[4] = (__int64)(v19 + 6);
          v19[3] = v22;
          v19[5] = 0x600000000LL;
          v23 = *((_DWORD *)v11 - 2);
          if ( v23 && v19 + 4 != v11 - 2 )
          {
            v25 = (_QWORD *)*(v11 - 2);
            if ( v25 == v11 )
            {
              v26 = v11;
              v27 = 8LL * v23;
              if ( v23 <= 6
                || (v31 = *((_DWORD *)v11 - 2),
                    sub_C8D5F0((__int64)(v19 + 4), v19 + 6, v23, 8u, v23, v8),
                    v21 = (void *)v19[4],
                    v26 = (const void *)*(v11 - 2),
                    v23 = v31,
                    (v27 = 8LL * *((unsigned int *)v11 - 2)) != 0) )
              {
                v30 = v23;
                memcpy(v21, v26, v27);
                *((_DWORD *)v19 + 10) = v30;
                *((_DWORD *)v11 - 2) = 0;
              }
              else
              {
                *((_DWORD *)v19 + 10) = v31;
                *((_DWORD *)v11 - 2) = 0;
              }
            }
            else
            {
              v19[4] = (__int64)v25;
              *((_DWORD *)v19 + 10) = *((_DWORD *)v11 - 2);
              *((_DWORD *)v19 + 11) = *((_DWORD *)v11 - 1);
              *(v11 - 2) = v11;
              *((_DWORD *)v11 - 1) = 0;
              *((_DWORD *)v11 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v24 = *(v11 - 2);
          if ( (_QWORD *)v24 != v11 )
            _libc_free(v24);
        }
        if ( (_QWORD *)v9 == v11 + 6 )
          break;
        v11 += 12;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v32, 8);
  }
  else
  {
    v28 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[12 * v28]; j != result; result += 12 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
