// Function: sub_27A41B0
// Address: 0x27a41b0
//
_QWORD *__fastcall sub_27A41B0(__int64 a1, int a2)
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
  __int64 *v19; // r15
  __int64 v20; // rsi
  void *v21; // rdi
  unsigned int v22; // r10d
  unsigned __int64 v23; // rdi
  _DWORD *v24; // r11
  const void *v25; // rsi
  size_t v26; // rdx
  _QWORD *j; // rdx
  _DWORD *v28; // [rsp+8h] [rbp-48h]
  _DWORD *v29; // [rsp+8h] [rbp-48h]
  unsigned int v30; // [rsp+14h] [rbp-3Ch]
  unsigned int v31; // [rsp+14h] [rbp-3Ch]
  __int64 v32; // [rsp+18h] [rbp-38h]

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
    v32 = 72 * v4;
    v9 = v5 + 72 * v4;
    for ( i = &result[9 * *(unsigned int *)(a1 + 24)]; i != result; result += 9 )
    {
      if ( result )
        *result = -4096;
    }
    v11 = (_QWORD *)(v5 + 24);
    if ( v9 == v5 )
      return (_QWORD *)sub_C7D6A0(v5, v32, 8);
    while ( 1 )
    {
      v12 = *(v11 - 3);
      if ( v12 != -8192 && v12 != -4096 )
        break;
LABEL_18:
      if ( (_QWORD *)v9 == v11 + 6 )
        return (_QWORD *)sub_C7D6A0(v5, v32, 8);
      v11 += 9;
    }
    v13 = *(_DWORD *)(a1 + 24);
    if ( !v13 )
    {
      MEMORY[0] = *(v11 - 3);
      BUG();
    }
    v14 = v13 - 1;
    v15 = *(_QWORD *)(a1 + 8);
    v16 = 1;
    v17 = 0;
    v18 = v14 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v19 = (__int64 *)(v15 + 72LL * v18);
    v20 = *v19;
    if ( v12 != *v19 )
    {
      while ( v20 != -4096 )
      {
        if ( !v17 && v20 == -8192 )
          v17 = v19;
        v18 = v14 & (v16 + v18);
        v19 = (__int64 *)(v15 + 72LL * v18);
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
    v21 = v19 + 3;
    v19[1] = (__int64)(v19 + 3);
    v19[2] = 0x200000000LL;
    v22 = *((_DWORD *)v11 - 2);
    if ( v22 && v19 + 1 != v11 - 2 )
    {
      v24 = (_DWORD *)*(v11 - 2);
      if ( v11 == (_QWORD *)v24 )
      {
        if ( v22 > 2 )
        {
          v29 = (_DWORD *)*(v11 - 2);
          v31 = *((_DWORD *)v11 - 2);
          sub_C8D5F0((__int64)(v19 + 1), v19 + 3, v22, 0x18u, (__int64)(v19 + 1), v8);
          v21 = (void *)v19[1];
          v25 = (const void *)*(v11 - 2);
          v22 = v31;
          v24 = v29;
          v26 = 24LL * *((unsigned int *)v11 - 2);
          if ( !v26 )
            goto LABEL_25;
        }
        else
        {
          v25 = v11;
          v26 = 24LL * v22;
        }
        v28 = v24;
        v30 = v22;
        memcpy(v21, v25, v26);
        v24 = v28;
        v22 = v30;
LABEL_25:
        *((_DWORD *)v19 + 4) = v22;
        *(v24 - 2) = 0;
        goto LABEL_16;
      }
      v19[1] = (__int64)v24;
      *((_DWORD *)v19 + 4) = *((_DWORD *)v11 - 2);
      *((_DWORD *)v19 + 5) = *((_DWORD *)v11 - 1);
      *(v11 - 2) = v11;
      *((_DWORD *)v11 - 1) = 0;
      *((_DWORD *)v11 - 2) = 0;
    }
LABEL_16:
    ++*(_DWORD *)(a1 + 16);
    v23 = *(v11 - 2);
    if ( (_QWORD *)v23 != v11 )
      _libc_free(v23);
    goto LABEL_18;
  }
  *(_QWORD *)(a1 + 16) = 0;
  for ( j = &result[9 * *(unsigned int *)(a1 + 24)]; j != result; result += 9 )
  {
    if ( result )
      *result = -4096;
  }
  return result;
}
