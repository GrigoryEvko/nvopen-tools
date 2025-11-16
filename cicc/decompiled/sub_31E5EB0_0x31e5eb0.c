// Function: sub_31E5EB0
// Address: 0x31e5eb0
//
_DWORD *__fastcall sub_31E5EB0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // rdx
  int *v9; // r10
  _DWORD *i; // rdx
  int *v11; // rdx
  int v12; // ecx
  int v13; // r8d
  int v14; // esi
  __int64 v15; // rdi
  int v16; // r14d
  int v17; // r9d
  _DWORD *v18; // r11
  unsigned int j; // eax
  _DWORD *v20; // r8
  unsigned int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rdx
  _DWORD *k; // rdx
  int v25; // r15d

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
  result = (_DWORD *)sub_C7D670(12LL * v6, 4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = (int *)(v5 + 12 * v4);
    for ( i = &result[3 * v8]; i != result; result += 3 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = -1;
      }
    }
    if ( v9 != (int *)v5 )
    {
      v11 = (int *)v5;
      do
      {
        while ( 1 )
        {
          v12 = *v11;
          if ( *v11 || (unsigned int)v11[1] <= 0xFFFFFFFD )
            break;
          v11 += 3;
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v5, 12 * v4, 4);
        }
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = *(_QWORD *)v11;
          BUG();
        }
        v14 = v11[1];
        v16 = 1;
        v17 = v13 - 1;
        v18 = 0;
        for ( j = (v13 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)(37 * v14) | ((unsigned __int64)(unsigned int)(37 * v12) << 32))) >> 31)
                 ^ (756364221 * v14)); ; j = v17 & v21 )
        {
          v15 = *(_QWORD *)(a1 + 8);
          v20 = (_DWORD *)(v15 + 12LL * j);
          if ( v12 == *v20 && v14 == v20[1] )
            break;
          if ( !*v20 )
          {
            v25 = v20[1];
            if ( v25 == -1 )
            {
              if ( v18 )
                v20 = v18;
              break;
            }
            if ( !v18 && v25 == -2 )
              v18 = (_DWORD *)(v15 + 12LL * j);
          }
          v21 = v16 + j;
          ++v16;
        }
        v22 = *(_QWORD *)v11;
        v11 += 3;
        *(_QWORD *)v20 = v22;
        v20[2] = *(v11 - 1);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v9 != v11 );
    }
    return (_DWORD *)sub_C7D6A0(v5, 12 * v4, 4);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[3 * v23]; k != result; result += 3 )
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
