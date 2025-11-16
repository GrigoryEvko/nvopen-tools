// Function: sub_31B3340
// Address: 0x31b3340
//
_QWORD *__fastcall sub_31B3340(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r14
  _QWORD *i; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // rax
  int v12; // edx
  int v13; // edx
  __int64 v14; // rdi
  int v15; // r11d
  __int64 *v16; // r10
  unsigned int v17; // ecx
  __int64 *v18; // r15
  __int64 v19; // rsi
  void *v20; // rdi
  unsigned int v21; // r10d
  unsigned __int64 v22; // rdi
  _QWORD *v23; // r11
  const void *v24; // rsi
  __int64 v25; // r8
  _QWORD *j; // rdx
  _QWORD *v27; // [rsp+8h] [rbp-48h]
  _QWORD *v28; // [rsp+8h] [rbp-48h]
  unsigned int v29; // [rsp+14h] [rbp-3Ch]
  unsigned int v30; // [rsp+14h] [rbp-3Ch]
  __int64 v31; // [rsp+18h] [rbp-38h]

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
    v31 = 72 * v4;
    v8 = v5 + 72 * v4;
    for ( i = &result[9 * *(unsigned int *)(a1 + 24)]; i != result; result += 9 )
    {
      if ( result )
        *result = -4096;
    }
    v10 = (_QWORD *)(v5 + 24);
    if ( v8 != v5 )
    {
      while ( 1 )
      {
        v11 = *(v10 - 3);
        if ( v11 != -8192 && v11 != -4096 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(v10 - 3);
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (__int64 *)(v14 + 72LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -4096 )
            {
              if ( !v16 && v19 == -8192 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (__int64 *)(v14 + 72LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_15;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_15:
          *v18 = v11;
          v20 = v18 + 3;
          v18[1] = (__int64)(v18 + 3);
          v18[2] = 0x600000000LL;
          v21 = *((_DWORD *)v10 - 2);
          if ( v21 && v18 + 1 != v10 - 2 )
          {
            v23 = (_QWORD *)*(v10 - 2);
            if ( v10 == v23 )
            {
              v24 = v10;
              v25 = 8LL * v21;
              if ( v21 <= 6 )
                goto LABEL_23;
              v28 = (_QWORD *)*(v10 - 2);
              v30 = *((_DWORD *)v10 - 2);
              sub_C8D5F0((__int64)(v18 + 1), v18 + 3, v21, 8u, v25, (__int64)(v18 + 1));
              v20 = (void *)v18[1];
              v24 = (const void *)*(v10 - 2);
              v21 = v30;
              v23 = v28;
              v25 = 8LL * *((unsigned int *)v10 - 2);
              if ( v25 )
              {
LABEL_23:
                v27 = v23;
                v29 = v21;
                memcpy(v20, v24, v25);
                *((_DWORD *)v18 + 4) = v29;
                *((_DWORD *)v27 - 2) = 0;
              }
              else
              {
                *((_DWORD *)v18 + 4) = v30;
                *((_DWORD *)v28 - 2) = 0;
              }
            }
            else
            {
              v18[1] = (__int64)v23;
              *((_DWORD *)v18 + 4) = *((_DWORD *)v10 - 2);
              *((_DWORD *)v18 + 5) = *((_DWORD *)v10 - 1);
              *(v10 - 2) = v10;
              *((_DWORD *)v10 - 1) = 0;
              *((_DWORD *)v10 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v22 = *(v10 - 2);
          if ( (_QWORD *)v22 != v10 )
            _libc_free(v22);
        }
        if ( (_QWORD *)v8 == v10 + 6 )
          break;
        v10 += 9;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v31, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[9 * *(unsigned int *)(a1 + 24)]; j != result; result += 9 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
