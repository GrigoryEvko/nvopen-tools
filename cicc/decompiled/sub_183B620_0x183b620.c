// Function: sub_183B620
// Address: 0x183b620
//
_QWORD *__fastcall sub_183B620(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 *v7; // r14
  _QWORD *i; // rdx
  unsigned __int64 *v9; // rbx
  unsigned __int64 v10; // rax
  int v11; // edx
  int v12; // esi
  __int64 v13; // r8
  __int64 *v14; // r9
  int v15; // r10d
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // rdi
  unsigned __int64 v19; // rdi
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
  result = (_QWORD *)sub_22077B0(40LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[5 * v3];
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
    {
      if ( result )
        *result = -2;
    }
    if ( v7 != v4 )
    {
      v9 = (unsigned __int64 *)v4;
      do
      {
        v10 = *v9;
        if ( *v9 != -16 && v10 != -2 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = *v9;
            BUG();
          }
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 0;
          v15 = 1;
          v16 = (v11 - 1) & (v10 ^ (v10 >> 9));
          v17 = (__int64 *)(v13 + 40LL * v16);
          v18 = *v17;
          if ( v10 != *v17 )
          {
            while ( v18 != -2 )
            {
              if ( v18 == -16 && !v14 )
                v14 = v17;
              v16 = v12 & (v15 + v16);
              v17 = (__int64 *)(v13 + 40LL * v16);
              v18 = *v17;
              if ( v10 == *v17 )
                goto LABEL_14;
              ++v15;
            }
            if ( v14 )
              v17 = v14;
          }
LABEL_14:
          *v17 = *v9;
          *((_DWORD *)v17 + 2) = *((_DWORD *)v9 + 2);
          v17[2] = v9[2];
          v17[3] = v9[3];
          v17[4] = v9[4];
          v9[4] = 0;
          v9[2] = 0;
          v9[3] = 0;
          ++*(_DWORD *)(a1 + 16);
          v19 = v9[2];
          if ( v19 )
            j_j___libc_free_0(v19, v9[4] - v19);
        }
        v9 += 5;
      }
      while ( v7 != (__int64 *)v9 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * *(unsigned int *)(a1 + 24)]; j != result; result += 5 )
    {
      if ( result )
        *result = -2;
    }
  }
  return result;
}
