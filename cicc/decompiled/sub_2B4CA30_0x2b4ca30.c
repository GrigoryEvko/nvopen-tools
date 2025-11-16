// Function: sub_2B4CA30
// Address: 0x2b4ca30
//
_DWORD *__fastcall sub_2B4CA30(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // r12
  _DWORD *i; // rdx
  __int64 v10; // rbx
  int v11; // eax
  __int64 v12; // r13
  int v13; // edx
  int v14; // edx
  __int64 v15; // r8
  int v16; // r10d
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // rdi
  int v20; // esi
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rdi
  __int64 v25; // r13
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rcx
  _DWORD *j; // rdx
  __int64 v30; // [rsp+8h] [rbp-38h]

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
  result = (_DWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v30 = 56 * v4;
    v8 = v5 + 56 * v4;
    for ( i = &result[14 * *(unsigned int *)(a1 + 24)]; i != result; result += 14 )
    {
      if ( result )
        *result = -1;
    }
    if ( v8 != v5 )
    {
      v10 = v5;
      do
      {
        while ( 1 )
        {
          v11 = *(_DWORD *)v10;
          v12 = v10 + 56;
          if ( *(_DWORD *)v10 <= 0xFFFFFFFD )
            break;
          v10 += 56;
          if ( v8 == v12 )
            return (_DWORD *)sub_C7D6A0(v5, v30, 8);
        }
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
        v18 = v14 & (unsigned int)(37 * v11);
        v19 = v15 + 56 * v18;
        v20 = *(_DWORD *)v19;
        if ( v11 != *(_DWORD *)v19 )
        {
          while ( v20 != -1 )
          {
            if ( !v17 && v20 == -2 )
              v17 = v19;
            v18 = v14 & (unsigned int)(v16 + v18);
            v19 = v15 + 56LL * (unsigned int)v18;
            v20 = *(_DWORD *)v19;
            if ( v11 == *(_DWORD *)v19 )
              goto LABEL_14;
            ++v16;
          }
          if ( v17 )
            v19 = v17;
        }
LABEL_14:
        *(_QWORD *)(v19 + 24) = 0;
        *(_QWORD *)(v19 + 16) = 0;
        *(_DWORD *)(v19 + 32) = 0;
        *(_DWORD *)v19 = v11;
        *(_QWORD *)(v19 + 8) = 1;
        v21 = *(_QWORD *)(v10 + 16);
        ++*(_QWORD *)(v10 + 8);
        v22 = *(_QWORD *)(v19 + 16);
        *(_QWORD *)(v19 + 16) = v21;
        LODWORD(v21) = *(_DWORD *)(v10 + 24);
        *(_QWORD *)(v10 + 16) = v22;
        LODWORD(v22) = *(_DWORD *)(v19 + 24);
        *(_DWORD *)(v19 + 24) = v21;
        LODWORD(v21) = *(_DWORD *)(v10 + 28);
        *(_DWORD *)(v10 + 24) = v22;
        LODWORD(v22) = *(_DWORD *)(v19 + 28);
        *(_DWORD *)(v19 + 28) = v21;
        v23 = *(unsigned int *)(v10 + 32);
        *(_DWORD *)(v10 + 28) = v22;
        LODWORD(v22) = *(_DWORD *)(v19 + 32);
        *(_DWORD *)(v19 + 32) = v23;
        *(_DWORD *)(v10 + 32) = v22;
        *(_QWORD *)(v19 + 40) = v19 + 56;
        *(_QWORD *)(v19 + 48) = 0;
        if ( *(_DWORD *)(v10 + 48) )
          sub_2B0CE50(v19 + 40, (char **)(v10 + 40), v23, v18, v15, v17);
        ++*(_DWORD *)(a1 + 16);
        v24 = *(_QWORD *)(v10 + 40);
        v25 = v10 + 56;
        if ( v24 != v10 + 56 )
          _libc_free(v24);
        v26 = *(unsigned int *)(v10 + 32);
        v27 = *(_QWORD *)(v10 + 16);
        v10 += 56;
        sub_C7D6A0(v27, 8 * v26, 8);
      }
      while ( v8 != v25 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v30, 8);
  }
  else
  {
    v28 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[14 * v28]; j != result; result += 14 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
