// Function: sub_25CFC20
// Address: 0x25cfc20
//
_QWORD *__fastcall sub_25CFC20(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r15
  __int64 v5; // r13
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r14
  _QWORD *i; // rdx
  __int64 v12; // rbx
  __int64 v13; // rax
  int v14; // esi
  int v15; // esi
  __int64 v16; // r9
  __int64 *v17; // r10
  int v18; // r11d
  unsigned int v19; // ecx
  __int64 *v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  _QWORD *j; // rdx

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
  result = (_QWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 32 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v5 + v9;
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
        *result = -1;
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      do
      {
        while ( 1 )
        {
          v13 = *(_QWORD *)v12;
          if ( *(_QWORD *)v12 <= 0xFFFFFFFFFFFFFFFDLL )
          {
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
            v19 = v15 & (((0xBF58476D1CE4E5B9LL * v13) >> 31) ^ (484763065 * v13));
            v20 = (__int64 *)(v16 + 32LL * v19);
            v21 = *v20;
            if ( v13 != *v20 )
            {
              while ( v21 != -1 )
              {
                if ( !v17 && v21 == -2 )
                  v17 = v20;
                v19 = v15 & (v18 + v19);
                v20 = (__int64 *)(v16 + 32LL * v19);
                v21 = *v20;
                if ( v13 == *v20 )
                  goto LABEL_14;
                ++v18;
              }
              if ( v17 )
                v20 = v17;
            }
LABEL_14:
            *v20 = v13;
            v20[1] = *(_QWORD *)(v12 + 8);
            v22 = *(_QWORD *)(v12 + 16);
            *(_QWORD *)(v12 + 8) = 0;
            v20[2] = v22;
            *((_DWORD *)v20 + 6) = *(_DWORD *)(v12 + 24);
            ++*(_DWORD *)(a1 + 16);
            v23 = *(_QWORD *)(v12 + 8);
            if ( v23 )
              break;
          }
          v12 += 32;
          if ( v10 == v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v12 += 32;
        j_j___libc_free_0(v23);
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * *(unsigned int *)(a1 + 24)]; j != result; result += 4 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
