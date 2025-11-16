// Function: sub_2E99B90
// Address: 0x2e99b90
//
_QWORD *__fastcall sub_2E99B90(__int64 a1, int a2)
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
  int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // r9
  int v17; // r11d
  _QWORD *v18; // r10
  __int64 v19; // rcx
  _QWORD *v20; // rdi
  __int64 v21; // rsi
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
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
          v19 = (unsigned int)v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = (_QWORD *)(v16 + 88 * v19);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -4096 )
            {
              if ( v21 == -8192 && !v18 )
                v18 = v20;
              v8 = (unsigned int)(v17 + 1);
              v19 = (unsigned int)v15 & (v17 + (_DWORD)v19);
              v20 = (_QWORD *)(v16 + 88LL * (unsigned int)v19);
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_14;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_14:
          *v20 = v13;
          v20[1] = v20 + 3;
          v20[2] = 0x800000000LL;
          if ( *(_DWORD *)(v12 + 16) )
            sub_2E97B00((__int64)(v20 + 1), (char **)(v12 + 8), v15, v19, v8, v16);
          ++*(_DWORD *)(a1 + 16);
          v22 = *(_QWORD *)(v12 + 8);
          if ( v22 != v12 + 24 )
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
