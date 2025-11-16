// Function: sub_1BE8C00
// Address: 0x1be8c00
//
_QWORD *__fastcall sub_1BE8C00(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 *v7; // r15
  _QWORD *i; // rdx
  __int64 *v9; // rbx
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // r8
  int v14; // r10d
  __int64 *v15; // r9
  __int64 v16; // rdx
  __int64 *v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned __int64 v20; // rdi
  __int64 v21; // rdx
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(72LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[9 * v3];
    for ( i = &result[9 * *(unsigned int *)(a1 + 24)]; i != result; result += 9 )
    {
      if ( result )
        *result = -8;
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        v10 = *v9;
        if ( *v9 != -16 && v10 != -8 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = *v9;
            BUG();
          }
          v12 = (unsigned int)(v11 - 1);
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 1;
          v15 = 0;
          v16 = (unsigned int)v12 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v17 = (__int64 *)(v13 + 72 * v16);
          v18 = *v17;
          if ( v10 != *v17 )
          {
            while ( v18 != -8 )
            {
              if ( v18 == -16 && !v15 )
                v15 = v17;
              v16 = (unsigned int)v12 & (v14 + (_DWORD)v16);
              v17 = (__int64 *)(v13 + 72LL * (unsigned int)v16);
              v18 = *v17;
              if ( v10 == *v17 )
                goto LABEL_14;
              ++v14;
            }
            if ( v15 )
              v17 = v15;
          }
LABEL_14:
          *v17 = v10;
          *((_DWORD *)v17 + 2) = *((_DWORD *)v9 + 2);
          *((_DWORD *)v17 + 3) = *((_DWORD *)v9 + 3);
          *((_DWORD *)v17 + 4) = *((_DWORD *)v9 + 4);
          v17[3] = v9[3];
          v19 = v9[4];
          v17[6] = 0x200000000LL;
          v17[4] = v19;
          v17[5] = (__int64)(v17 + 7);
          if ( *((_DWORD *)v9 + 12) )
            sub_1BE2200((__int64)(v17 + 5), (char **)v9 + 5, v16, v12, v13, (int)v15);
          ++*(_DWORD *)(a1 + 16);
          v20 = v9[5];
          if ( (__int64 *)v20 != v9 + 7 )
            _libc_free(v20);
        }
        v9 += 9;
      }
      while ( v7 != v9 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[9 * v21]; j != result; result += 9 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
