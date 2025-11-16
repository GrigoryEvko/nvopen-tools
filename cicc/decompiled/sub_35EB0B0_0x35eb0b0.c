// Function: sub_35EB0B0
// Address: 0x35eb0b0
//
_DWORD *__fastcall sub_35EB0B0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 v10; // r15
  _DWORD *i; // rdx
  _DWORD *v12; // rbx
  char *v13; // rax
  int v14; // eax
  int v15; // edx
  int v16; // edx
  __int64 v17; // rdi
  int *v18; // r10
  int v19; // r11d
  unsigned int v20; // ecx
  int *v21; // r13
  int v22; // esi
  void *v23; // rdi
  unsigned int v24; // r10d
  unsigned __int64 v25; // rdi
  _DWORD *v26; // r11
  const void *v27; // rsi
  __int64 v28; // r8
  _DWORD *j; // rdx
  __int64 v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+8h] [rbp-48h]
  _DWORD *v32; // [rsp+10h] [rbp-40h]
  _DWORD *v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+18h] [rbp-38h]
  unsigned int v35; // [rsp+18h] [rbp-38h]
  unsigned int v36; // [rsp+18h] [rbp-38h]

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
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 72 * v4;
    v10 = v5 + 72 * v4;
    for ( i = &result[18 * v8]; i != result; result += 18 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
    v12 = (_DWORD *)(v5 + 24);
    if ( v10 != v5 )
    {
      while ( 1 )
      {
        v14 = *(v12 - 6);
        if ( (unsigned int)(v14 + 0x7FFFFFFF) > 0xFFFFFFFD )
          goto LABEL_10;
        v15 = *(_DWORD *)(a1 + 24);
        if ( !v15 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 8);
        v18 = 0;
        v19 = 1;
        v20 = v16 & (37 * v14);
        v21 = (int *)(v17 + 72LL * v20);
        v22 = *v21;
        if ( v14 != *v21 )
        {
          while ( v22 != 0x7FFFFFFF )
          {
            if ( !v18 && v22 == 0x80000000 )
              v18 = v21;
            v20 = v16 & (v19 + v20);
            v21 = (int *)(v17 + 72LL * v20);
            v22 = *v21;
            if ( v14 == *v21 )
              goto LABEL_15;
            ++v19;
          }
          if ( v18 )
            v21 = v18;
        }
LABEL_15:
        *v21 = v14;
        v23 = v21 + 6;
        *((_QWORD *)v21 + 1) = v21 + 6;
        *((_QWORD *)v21 + 2) = 0x600000000LL;
        v24 = *(v12 - 2);
        if ( v24 && v21 + 2 != v12 - 4 )
        {
          v26 = (_DWORD *)*((_QWORD *)v12 - 2);
          if ( v12 == v26 )
          {
            v27 = v12;
            v28 = 8LL * v24;
            if ( v24 <= 6 )
              goto LABEL_22;
            v31 = v9;
            v33 = (_DWORD *)*((_QWORD *)v12 - 2);
            v36 = *(v12 - 2);
            sub_C8D5F0((__int64)(v21 + 2), v21 + 6, v24, 8u, v28, v9);
            v23 = (void *)*((_QWORD *)v21 + 1);
            v27 = (const void *)*((_QWORD *)v12 - 2);
            v24 = v36;
            v26 = v33;
            v9 = v31;
            v28 = 8LL * (unsigned int)*(v12 - 2);
            if ( v28 )
            {
LABEL_22:
              v30 = v9;
              v32 = v26;
              v35 = v24;
              memcpy(v23, v27, v28);
              v9 = v30;
              v21[4] = v35;
              *(v32 - 2) = 0;
            }
            else
            {
              v21[4] = v36;
              *(v33 - 2) = 0;
            }
          }
          else
          {
            *((_QWORD *)v21 + 1) = v26;
            v21[4] = *(v12 - 2);
            v21[5] = *(v12 - 1);
            *((_QWORD *)v12 - 2) = v12;
            *(v12 - 1) = 0;
            *(v12 - 2) = 0;
          }
        }
        ++*(_DWORD *)(a1 + 16);
        v25 = *((_QWORD *)v12 - 2);
        if ( (_DWORD *)v25 == v12 )
        {
LABEL_10:
          v13 = (char *)(v12 + 18);
          if ( (_DWORD *)v10 == v12 + 12 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        else
        {
          v34 = v9;
          _libc_free(v25);
          v13 = (char *)(v12 + 18);
          v9 = v34;
          if ( (_DWORD *)v10 == v12 + 12 )
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
    for ( j = &result[18 * *(unsigned int *)(a1 + 24)]; j != result; result += 18 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
  }
  return result;
}
