// Function: sub_2C6FB10
// Address: 0x2c6fb10
//
_QWORD *__fastcall sub_2C6FB10(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r8
  __int64 v9; // r13
  __int64 v10; // r14
  _QWORD *i; // rdx
  __int64 v12; // rbx
  __int64 v13; // rax
  int v14; // ecx
  __int64 v15; // rcx
  __int64 v16; // r9
  int v17; // r11d
  __int64 v18; // r10
  unsigned int v19; // esi
  __int64 v20; // rdx
  __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  _QWORD *j; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(88LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 88 * v4;
    v10 = v5 + 88 * v4;
    for ( i = &result[11 * *(unsigned int *)(a1 + 24)]; i != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      do
      {
        v13 = *(_QWORD *)v12;
        if ( *(_QWORD *)v12 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *(_QWORD *)v12;
            BUG();
          }
          v15 = (unsigned int)(v14 - 1);
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = v16 + 88LL * v19;
          v21 = *(_QWORD *)v20;
          if ( v13 != *(_QWORD *)v20 )
          {
            while ( v21 != -4096 )
            {
              if ( v21 == -8192 && !v18 )
                v18 = v20;
              v8 = (unsigned int)(v17 + 1);
              v19 = v15 & (v17 + v19);
              v20 = v16 + 88LL * v19;
              v21 = *(_QWORD *)v20;
              if ( v13 == *(_QWORD *)v20 )
                goto LABEL_14;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_14:
          *(_QWORD *)v20 = v13;
          *(_DWORD *)(v20 + 8) = *(_DWORD *)(v12 + 8);
          *(_QWORD *)(v20 + 16) = v20 + 32;
          *(_QWORD *)(v20 + 24) = 0x600000000LL;
          if ( *(_DWORD *)(v12 + 24) )
          {
            v25 = v20;
            sub_2C6DFD0(v20 + 16, (char **)(v12 + 16), v20, v15, v8, v16);
            v20 = v25;
          }
          *(_DWORD *)(v20 + 80) = *(_DWORD *)(v12 + 80);
          ++*(_DWORD *)(a1 + 16);
          v22 = *(_QWORD *)(v12 + 16);
          if ( v22 != v12 + 32 )
            _libc_free(v22);
        }
        v12 += 88;
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[11 * v23]; j != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
