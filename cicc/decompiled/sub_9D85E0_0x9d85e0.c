// Function: sub_9D85E0
// Address: 0x9d85e0
//
_DWORD *__fastcall sub_9D85E0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r12
  int *v5; // r13
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  int *v10; // r14
  _DWORD *i; // rdx
  char **v12; // rbx
  int v13; // eax
  int v14; // edx
  int v15; // edx
  __int64 v16; // r9
  int v17; // r11d
  int *v18; // r10
  int *v19; // rsi
  int *v20; // rdi
  int v21; // r8d
  int *v22; // rdi
  __int64 v23; // rdx
  _DWORD *j; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(int **)(a1 + 8);
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
  result = (_DWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 32 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = &v5[(unsigned __int64)v9 / 4];
    for ( i = &result[8 * v8]; i != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
    if ( v10 != v5 )
    {
      v12 = (char **)v5;
      do
      {
        while ( 1 )
        {
          v13 = *(_DWORD *)v12;
          if ( *(_DWORD *)v12 <= 0xFFFFFFFD )
          {
            v14 = *(_DWORD *)(a1 + 24);
            if ( !v14 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v15 = v14 - 1;
            v16 = *(_QWORD *)(a1 + 8);
            v17 = 1;
            v18 = 0;
            v19 = (int *)(v15 & (unsigned int)(37 * v13));
            v20 = (int *)(v16 + 32LL * (_QWORD)v19);
            v21 = *v20;
            if ( v13 != *v20 )
            {
              while ( v21 != -1 )
              {
                if ( !v18 && v21 == -2 )
                  v18 = v20;
                v19 = (int *)(v15 & (unsigned int)(v17 + (_DWORD)v19));
                v20 = (int *)(v16 + 32LL * (unsigned int)v19);
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
            *((_QWORD *)v20 + 1) = v20 + 6;
            *((_QWORD *)v20 + 2) = 0x100000000LL;
            if ( *((_DWORD *)v12 + 4) )
            {
              v19 = (int *)(v12 + 1);
              sub_9C31C0((__int64)(v20 + 2), v12 + 1);
            }
            ++*(_DWORD *)(a1 + 16);
            v22 = (int *)v12[1];
            if ( v22 != (int *)(v12 + 3) )
              break;
          }
          v12 += 4;
          if ( v10 == (int *)v12 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        _libc_free(v22, v19);
        v12 += 4;
      }
      while ( v10 != (int *)v12 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[8 * v23]; j != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
