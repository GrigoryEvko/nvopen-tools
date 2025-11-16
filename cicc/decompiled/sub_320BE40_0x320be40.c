// Function: sub_320BE40
// Address: 0x320be40
//
_QWORD *__fastcall sub_320BE40(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r15
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rcx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v10; // rax
  int v11; // esi
  __int64 v12; // rdi
  int v13; // esi
  int v14; // r10d
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 *v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r14
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // r13
  unsigned __int64 v22; // r15
  unsigned __int64 v23; // rdi
  bool v24; // cc
  unsigned __int64 v25; // rdi
  __int64 v26; // rcx
  _QWORD *k; // rdx
  __int64 v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v29 = v4;
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(112LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v28 = 112 * v3;
    v31 = 112 * v3 + v4;
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( i = &result[14 * v7]; i != result; result += 14 )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v4; v31 != j; j += 112 )
    {
      v10 = *(_QWORD *)j;
      if ( *(_QWORD *)j != -8192 && v10 != -4096 )
      {
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v12 = *(_QWORD *)(a1 + 8);
        v13 = v11 - 1;
        v14 = 1;
        v15 = 0;
        v16 = v13 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v17 = (__int64 *)(v12 + 112 * v16);
        v18 = *v17;
        if ( v10 != *v17 )
        {
          while ( v18 != -4096 )
          {
            if ( v18 == -8192 && !v15 )
              v15 = (__int64)v17;
            v16 = v13 & (unsigned int)(v14 + v16);
            v17 = (__int64 *)(v12 + 112LL * (unsigned int)v16);
            v18 = *v17;
            if ( v10 == *v17 )
              goto LABEL_13;
            ++v14;
          }
          if ( v15 )
            v17 = (__int64 *)v15;
        }
LABEL_13:
        *v17 = v10;
        v17[1] = (__int64)(v17 + 3);
        v17[2] = 0x100000000LL;
        if ( *(_DWORD *)(j + 16) )
          sub_320B760((__int64)(v17 + 1), j + 8, (__int64)v17, v16, v18, v15);
        ++*(_DWORD *)(a1 + 16);
        v19 = *(_QWORD *)(j + 8);
        v20 = v19 + 88LL * *(unsigned int *)(j + 16);
        if ( v19 != v20 )
        {
          do
          {
            v20 -= 88LL;
            if ( *(_BYTE *)(v20 + 80) )
            {
              v24 = *(_DWORD *)(v20 + 72) <= 0x40u;
              *(_BYTE *)(v20 + 80) = 0;
              if ( !v24 )
              {
                v25 = *(_QWORD *)(v20 + 64);
                if ( v25 )
                  j_j___libc_free_0_0(v25);
              }
            }
            v21 = *(_QWORD *)(v20 + 40);
            v22 = v21 + 40LL * *(unsigned int *)(v20 + 48);
            if ( v21 != v22 )
            {
              do
              {
                v22 -= 40LL;
                v23 = *(_QWORD *)(v22 + 8);
                if ( v23 != v22 + 24 )
                  _libc_free(v23);
              }
              while ( v21 != v22 );
              v21 = *(_QWORD *)(v20 + 40);
            }
            if ( v21 != v20 + 56 )
              _libc_free(v21);
            sub_C7D6A0(*(_QWORD *)(v20 + 16), 12LL * *(unsigned int *)(v20 + 32), 4);
          }
          while ( v19 != v20 );
          v20 = *(_QWORD *)(j + 8);
        }
        if ( v20 != j + 24 )
          _libc_free(v20);
      }
    }
    return (_QWORD *)sub_C7D6A0(v29, v28, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[14 * v26]; k != result; result += 14 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
