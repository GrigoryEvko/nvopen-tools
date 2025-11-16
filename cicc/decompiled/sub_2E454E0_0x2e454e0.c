// Function: sub_2E454E0
// Address: 0x2e454e0
//
_DWORD *__fastcall sub_2E454E0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  _DWORD *i; // rdx
  __int64 v11; // rbx
  int v12; // edx
  int v13; // ecx
  int v14; // ecx
  __int64 v15; // r8
  int v16; // r11d
  int *v17; // r10
  unsigned int v18; // esi
  int *v19; // r13
  int v20; // edi
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
  _DWORD *j; // rdx
  __int64 v28; // [rsp+8h] [rbp-38h]

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
  result = (_DWORD *)sub_C7D670((unsigned __int64)v6 << 7, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v28 = v4 << 7;
    v9 = v5 + (v4 << 7);
    for ( i = &result[32 * v8]; i != result; result += 32 )
    {
      if ( result )
        *result = -1;
    }
    if ( v9 != v5 )
    {
      v11 = v5;
      do
      {
        while ( 1 )
        {
          v12 = *(_DWORD *)v11;
          if ( *(_DWORD *)v11 <= 0xFFFFFFFD )
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v14 = v13 - 1;
            v15 = *(_QWORD *)(a1 + 8);
            v16 = 1;
            v17 = 0;
            v18 = v14 & (37 * v12);
            v19 = (int *)(v15 + ((unsigned __int64)v18 << 7));
            v20 = *v19;
            if ( v12 != *v19 )
            {
              while ( v20 != -1 )
              {
                if ( !v17 && v20 == -2 )
                  v17 = v19;
                v18 = v14 & (v16 + v18);
                v19 = (int *)(v15 + ((unsigned __int64)v18 << 7));
                v20 = *v19;
                if ( v12 == *v19 )
                  goto LABEL_14;
                ++v16;
              }
              if ( v17 )
                v19 = v17;
            }
LABEL_14:
            *v19 = v12;
            *((_QWORD *)v19 + 1) = *(_QWORD *)(v11 + 8);
            *((_QWORD *)v19 + 2) = *(_QWORD *)(v11 + 16);
            sub_C8CF70((__int64)(v19 + 6), v19 + 14, 4, v11 + 56, v11 + 24);
            *((_QWORD *)v19 + 11) = v19 + 26;
            *((_QWORD *)v19 + 12) = 0x400000000LL;
            if ( *(_DWORD *)(v11 + 96) )
              sub_2E44AB0((__int64)(v19 + 22), (char **)(v11 + 88), (__int64)(v19 + 26), v21, v22, v23);
            *((_BYTE *)v19 + 120) = *(_BYTE *)(v11 + 120);
            ++*(_DWORD *)(a1 + 16);
            v24 = *(_QWORD *)(v11 + 88);
            if ( v24 != v11 + 104 )
              _libc_free(v24);
            if ( !*(_BYTE *)(v11 + 52) )
              break;
          }
          v11 += 128;
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v5, v28, 8);
        }
        v25 = *(_QWORD *)(v11 + 32);
        v11 += 128;
        _libc_free(v25);
      }
      while ( v9 != v11 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v28, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[32 * v26]; j != result; result += 32 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
