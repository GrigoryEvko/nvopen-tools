// Function: sub_31E45C0
// Address: 0x31e45c0
//
_DWORD *__fastcall sub_31E45C0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // r13d
  __int64 v5; // r12
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rdx
  _DWORD *v9; // r10
  _DWORD *i; // rdx
  _DWORD *v11; // rdx
  int v12; // ecx
  int v13; // r9d
  int v14; // esi
  int v15; // r9d
  __int64 v16; // rdi
  int v17; // r14d
  __int64 v18; // r11
  unsigned int j; // eax
  __int64 v20; // r8
  unsigned int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rdx
  _DWORD *k; // rdx
  int v25; // r15d
  __int64 v26; // [rsp+8h] [rbp-38h]

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
  result = (_DWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v26 = 16LL * v4;
    v9 = (_DWORD *)(v5 + v26);
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = -1;
      }
    }
    if ( v9 != (_DWORD *)v5 )
    {
      v11 = (_DWORD *)v5;
      do
      {
        while ( 1 )
        {
          v12 = *v11;
          if ( *v11 || v11[1] <= 0xFFFFFFFD )
            break;
          v11 += 4;
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v5, v26, 8);
        }
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = *(_QWORD *)v11;
          BUG();
        }
        v14 = v11[1];
        v15 = v13 - 1;
        v17 = 1;
        v18 = 0;
        for ( j = v15
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)(37 * v14) | ((unsigned __int64)(unsigned int)(37 * v12) << 32))) >> 31)
                 ^ (756364221 * v14)); ; j = v15 & v21 )
        {
          v16 = *(_QWORD *)(a1 + 8);
          v20 = v16 + 16LL * j;
          if ( v12 == *(_DWORD *)v20 && v14 == *(_DWORD *)(v20 + 4) )
            break;
          if ( !*(_DWORD *)v20 )
          {
            v25 = *(_DWORD *)(v20 + 4);
            if ( v25 == -1 )
            {
              if ( v18 )
                v20 = v18;
              break;
            }
            if ( !v18 && v25 == -2 )
              v18 = v16 + 16LL * j;
          }
          v21 = v17 + j;
          ++v17;
        }
        v22 = *(_QWORD *)v11;
        v11 += 4;
        *(_QWORD *)v20 = v22;
        *(_QWORD *)(v20 + 8) = *((_QWORD *)v11 - 1);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v9 != v11 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v26, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[4 * v23]; k != result; result += 4 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = -1;
      }
    }
  }
  return result;
}
