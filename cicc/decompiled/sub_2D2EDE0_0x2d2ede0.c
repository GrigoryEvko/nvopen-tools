// Function: sub_2D2EDE0
// Address: 0x2d2ede0
//
_DWORD *__fastcall sub_2D2EDE0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  unsigned int v5; // eax
  _DWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // r9
  __int64 v9; // r15
  _DWORD *i; // rdx
  _DWORD *v11; // rbx
  char *v12; // rax
  unsigned int v13; // eax
  int v14; // edx
  int v15; // edx
  __int64 v16; // rdi
  unsigned int *v17; // r10
  int v18; // r11d
  unsigned int v19; // ecx
  unsigned int *v20; // r13
  unsigned int v21; // esi
  void *v22; // rdi
  unsigned int v23; // r10d
  unsigned __int64 v24; // rdi
  _DWORD *v25; // r11
  const void *v26; // rsi
  __int64 v27; // r8
  _DWORD *j; // rdx
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  _DWORD *v31; // [rsp+10h] [rbp-40h]
  _DWORD *v32; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+18h] [rbp-38h]
  unsigned int v34; // [rsp+18h] [rbp-38h]
  unsigned int v35; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_C7D670(72LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 72 * v3;
    v9 = v4 + 72 * v3;
    for ( i = &result[18 * v7]; i != result; result += 18 )
    {
      if ( result )
        *result = -1;
    }
    v11 = (_DWORD *)(v4 + 24);
    if ( v9 != v4 )
    {
      while ( 1 )
      {
        v13 = *(v11 - 6);
        if ( v13 > 0xFFFFFFFD )
          goto LABEL_10;
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 8);
        v17 = 0;
        v18 = 1;
        v19 = v15 & (37 * v13);
        v20 = (unsigned int *)(v16 + 72LL * v19);
        v21 = *v20;
        if ( v13 != *v20 )
        {
          while ( v21 != -1 )
          {
            if ( !v17 && v21 == -2 )
              v17 = v20;
            v19 = v15 & (v18 + v19);
            v20 = (unsigned int *)(v16 + 72LL * v19);
            v21 = *v20;
            if ( v13 == *v20 )
              goto LABEL_15;
            ++v18;
          }
          if ( v17 )
            v20 = v17;
        }
LABEL_15:
        *v20 = v13;
        v22 = v20 + 6;
        *((_QWORD *)v20 + 1) = v20 + 6;
        *((_QWORD *)v20 + 2) = 0xC00000000LL;
        v23 = *(v11 - 2);
        if ( v23 && v20 + 2 != v11 - 4 )
        {
          v25 = (_DWORD *)*((_QWORD *)v11 - 2);
          if ( v11 == v25 )
          {
            v26 = v11;
            v27 = 4LL * v23;
            if ( v23 <= 0xC )
              goto LABEL_22;
            v30 = v8;
            v32 = (_DWORD *)*((_QWORD *)v11 - 2);
            v35 = *(v11 - 2);
            sub_C8D5F0((__int64)(v20 + 2), v20 + 6, v23, 4u, v27, v8);
            v22 = (void *)*((_QWORD *)v20 + 1);
            v26 = (const void *)*((_QWORD *)v11 - 2);
            v23 = v35;
            v25 = v32;
            v8 = v30;
            v27 = 4LL * (unsigned int)*(v11 - 2);
            if ( v27 )
            {
LABEL_22:
              v29 = v8;
              v31 = v25;
              v34 = v23;
              memcpy(v22, v26, v27);
              v8 = v29;
              v20[4] = v34;
              *(v31 - 2) = 0;
            }
            else
            {
              v20[4] = v35;
              *(v32 - 2) = 0;
            }
          }
          else
          {
            *((_QWORD *)v20 + 1) = v25;
            v20[4] = *(v11 - 2);
            v20[5] = *(v11 - 1);
            *((_QWORD *)v11 - 2) = v11;
            *(v11 - 1) = 0;
            *(v11 - 2) = 0;
          }
        }
        ++*(_DWORD *)(a1 + 16);
        v24 = *((_QWORD *)v11 - 2);
        if ( v11 == (_DWORD *)v24 )
        {
LABEL_10:
          v12 = (char *)(v11 + 18);
          if ( (_DWORD *)v9 == v11 + 12 )
            return (_DWORD *)sub_C7D6A0(v4, v8, 8);
        }
        else
        {
          v33 = v8;
          _libc_free(v24);
          v12 = (char *)(v11 + 18);
          v8 = v33;
          if ( (_DWORD *)v9 == v11 + 12 )
            return (_DWORD *)sub_C7D6A0(v4, v8, 8);
        }
        v11 = v12;
      }
    }
    return (_DWORD *)sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[18 * *(unsigned int *)(a1 + 24)]; j != result; result += 18 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
