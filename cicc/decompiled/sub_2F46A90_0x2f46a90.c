// Function: sub_2F46A90
// Address: 0x2f46a90
//
_DWORD *__fastcall sub_2F46A90(__int64 a1, int a2)
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
  _DWORD *v27; // r11
  const void *v28; // rsi
  __int64 v29; // r8
  _DWORD *j; // rdx
  __int64 v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+8h] [rbp-48h]
  _DWORD *v33; // [rsp+10h] [rbp-40h]
  _DWORD *v34; // [rsp+10h] [rbp-40h]
  __int64 v35; // [rsp+18h] [rbp-38h]
  unsigned int v36; // [rsp+18h] [rbp-38h]
  unsigned int v37; // [rsp+18h] [rbp-38h]

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
  result = (_DWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 40 * v4;
    v10 = v5 + 40 * v4;
    for ( i = &result[10 * v8]; i != result; result += 10 )
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
        v21 = (int *)(v17 + 40LL * v20);
        v22 = *v21;
        if ( v14 != *v21 )
        {
          while ( v22 != -1 )
          {
            if ( !v18 && v22 == -2 )
              v18 = v21;
            v20 = v16 & (v19 + v20);
            v21 = (int *)(v17 + 40LL * v20);
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
        *((_QWORD *)v21 + 2) = 0x200000000LL;
        v25 = *(v12 - 2);
        if ( v25 && v21 + 2 != v12 - 4 )
        {
          v27 = (_DWORD *)*((_QWORD *)v12 - 2);
          if ( v12 == v27 )
          {
            v28 = v12;
            v29 = 8LL * v25;
            if ( v25 <= 2 )
              goto LABEL_22;
            v32 = v9;
            v34 = (_DWORD *)*((_QWORD *)v12 - 2);
            v37 = *(v12 - 2);
            sub_C8D5F0((__int64)(v21 + 2), v21 + 6, v25, 8u, v29, v9);
            v24 = (void *)*((_QWORD *)v21 + 1);
            v28 = (const void *)*((_QWORD *)v12 - 2);
            v25 = v37;
            v27 = v34;
            v9 = v32;
            v29 = 8LL * (unsigned int)*(v12 - 2);
            if ( v29 )
            {
LABEL_22:
              v31 = v9;
              v33 = v27;
              v36 = v25;
              memcpy(v24, v28, v29);
              v9 = v31;
              v21[4] = v36;
              *(v33 - 2) = 0;
            }
            else
            {
              v21[4] = v37;
              *(v34 - 2) = 0;
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
        if ( v12 == (_DWORD *)v26 )
        {
LABEL_10:
          v13 = (char *)(v12 + 10);
          if ( (_DWORD *)v10 == v12 + 4 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        else
        {
          v35 = v9;
          _libc_free(v26);
          v13 = (char *)(v12 + 10);
          v9 = v35;
          if ( (_DWORD *)v10 == v12 + 4 )
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
    for ( j = &result[10 * *(unsigned int *)(a1 + 24)]; j != result; result += 10 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
