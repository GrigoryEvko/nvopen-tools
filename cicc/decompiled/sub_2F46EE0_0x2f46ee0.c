// Function: sub_2F46EE0
// Address: 0x2f46ee0
//
_DWORD *__fastcall sub_2F46EE0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 v5; // r14
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 v10; // r15
  _DWORD *i; // rdx
  _DWORD *v12; // rbx
  char *v13; // rax
  unsigned int v14; // eax
  int v15; // edx
  int v16; // edx
  __int64 v17; // rdi
  int *v18; // r10
  int v19; // r11d
  unsigned int v20; // ecx
  int *v21; // r13
  int v22; // esi
  int v23; // eax
  void *v24; // rdi
  unsigned int v25; // r10d
  unsigned __int64 v26; // rdi
  _DWORD *v27; // rax
  const void *v28; // rsi
  size_t v29; // rdx
  _DWORD *j; // rdx
  unsigned int v31; // [rsp+4h] [rbp-3Ch]
  unsigned int v32; // [rsp+4h] [rbp-3Ch]
  __int64 v33; // [rsp+8h] [rbp-38h]
  __int64 v34; // [rsp+8h] [rbp-38h]
  __int64 v35; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
  result = (_DWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 32LL * v4;
    v10 = v5 + v9;
    for ( i = &result[8 * v8]; i != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
    v12 = (_DWORD *)(v5 + 24);
    if ( v10 != v5 )
    {
      while ( 1 )
      {
        v14 = *(v12 - 6);
        if ( v14 > 0xFFFFFFFD )
          goto LABEL_10;
        v15 = *(_DWORD *)(a1 + 24);
        if ( !v15 )
        {
          MEMORY[0] = *(v12 - 6);
          BUG();
        }
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 8);
        v18 = 0;
        v19 = 1;
        v20 = v16 & (37 * v14);
        v21 = (int *)(v17 + 32LL * v20);
        v22 = *v21;
        if ( v14 != *v21 )
        {
          while ( v22 != -1 )
          {
            if ( !v18 && v22 == -2 )
              v18 = v21;
            v20 = v16 & (v19 + v20);
            v21 = (int *)(v17 + 32LL * v20);
            v22 = *v21;
            if ( v14 == *v21 )
              goto LABEL_15;
            ++v19;
          }
          if ( v18 )
            v21 = v18;
        }
LABEL_15:
        v23 = *(v12 - 6);
        v24 = v21 + 6;
        *((_QWORD *)v21 + 1) = v21 + 6;
        *v21 = v23;
        *((_QWORD *)v21 + 2) = 0x100000000LL;
        v25 = *(v12 - 2);
        if ( v25 && v21 + 2 != v12 - 4 )
        {
          v27 = (_DWORD *)*((_QWORD *)v12 - 2);
          if ( v27 == v12 )
          {
            v28 = v12;
            v29 = 8;
            if ( v25 == 1 )
              goto LABEL_22;
            v32 = *(v12 - 2);
            v35 = v9;
            sub_C8D5F0((__int64)(v21 + 2), v21 + 6, v25, 8u, v25, v9);
            v24 = (void *)*((_QWORD *)v21 + 1);
            v28 = (const void *)*((_QWORD *)v12 - 2);
            v9 = v35;
            v29 = 8LL * (unsigned int)*(v12 - 2);
            v25 = v32;
            if ( v29 )
            {
LABEL_22:
              v31 = v25;
              v34 = v9;
              memcpy(v24, v28, v29);
              v9 = v34;
              v21[4] = v31;
              *(v12 - 2) = 0;
            }
            else
            {
              v21[4] = v32;
              *(v12 - 2) = 0;
            }
          }
          else
          {
            *((_QWORD *)v21 + 1) = v27;
            v21[4] = *(v12 - 2);
            v21[5] = *(v12 - 1);
            *((_QWORD *)v12 - 2) = v12;
            *(v12 - 1) = 0;
            *(v12 - 2) = 0;
          }
        }
        ++*(_DWORD *)(a1 + 16);
        v26 = *((_QWORD *)v12 - 2);
        if ( (_DWORD *)v26 == v12 )
        {
LABEL_10:
          v13 = (char *)(v12 + 8);
          if ( (_DWORD *)v10 == v12 + 2 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        else
        {
          v33 = v9;
          _libc_free(v26);
          v13 = (char *)(v12 + 8);
          v9 = v33;
          if ( (_DWORD *)v10 == v12 + 2 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v12 = v13;
      }
    }
    return (_DWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[8 * *(unsigned int *)(a1 + 24)]; j != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
