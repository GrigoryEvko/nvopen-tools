// Function: sub_D277E0
// Address: 0xd277e0
//
_QWORD *__fastcall sub_D277E0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // r14
  _QWORD *i; // rdx
  __int64 v13; // rbx
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // r9
  int v18; // r11d
  __int64 *v19; // r10
  __int64 v20; // rcx
  __int64 *v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rdx
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
    v9 = *(unsigned int *)(a1 + 24);
    v10 = 32 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v11 = v5 + v10;
    for ( i = &result[4 * v9]; i != result; result += 4 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v11 != v5 )
    {
      v13 = v5;
      do
      {
        v14 = *(_QWORD *)v13;
        if ( *(_QWORD *)v13 != -8192 && v14 != -4096 )
        {
          v15 = *(_DWORD *)(a1 + 24);
          if ( !v15 )
          {
            MEMORY[0] = *(_QWORD *)v13;
            BUG();
          }
          v16 = (unsigned int)(v15 - 1);
          v17 = *(_QWORD *)(a1 + 8);
          v18 = 1;
          v19 = 0;
          v20 = (unsigned int)v16 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v21 = (__int64 *)(v17 + 32 * v20);
          v22 = *v21;
          if ( v14 != *v21 )
          {
            while ( v22 != -4096 )
            {
              if ( v22 == -8192 && !v19 )
                v19 = v21;
              v8 = (unsigned int)(v18 + 1);
              v20 = (unsigned int)v16 & (v18 + (_DWORD)v20);
              v21 = (__int64 *)(v17 + 32LL * (unsigned int)v20);
              v22 = *v21;
              if ( v14 == *v21 )
                goto LABEL_14;
              ++v18;
            }
            if ( v19 )
              v21 = v19;
          }
LABEL_14:
          *v21 = v14;
          v21[1] = (__int64)(v21 + 3);
          v21[2] = 0x100000000LL;
          if ( *(_DWORD *)(v13 + 16) )
          {
            v22 = v13 + 8;
            sub_D230A0((__int64)(v21 + 1), (char **)(v13 + 8), v16, v20, v8, v17);
          }
          ++*(_DWORD *)(a1 + 16);
          v23 = *(_QWORD *)(v13 + 8);
          if ( v23 != v13 + 24 )
            _libc_free(v23, v22);
        }
        v13 += 32;
      }
      while ( v11 != v13 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v10, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v24]; j != result; result += 4 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
