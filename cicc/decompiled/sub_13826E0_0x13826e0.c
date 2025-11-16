// Function: sub_13826E0
// Address: 0x13826e0
//
_QWORD *__fastcall sub_13826E0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rcx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  char **v10; // rbx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  char *v13; // rax
  int v14; // edx
  int v15; // esi
  __int64 v16; // rdi
  int v17; // r10d
  char **v18; // r9
  unsigned int v19; // ecx
  char **v20; // rdx
  char *v21; // r8
  __int64 v22; // rcx
  _QWORD *j; // rdx
  char **v24; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_22077B0(136LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[17 * v3];
    for ( i = &result[17 * v7]; i != result; result += 17 )
    {
      if ( result )
        *result = -8;
    }
    if ( v8 != v4 )
    {
      v10 = (char **)v4;
      do
      {
        v13 = *v10;
        if ( *v10 != (char *)-16LL && v13 != (char *)-8LL )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *v10;
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = (char **)(v16 + 136LL * v19);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != (char *)-8LL )
            {
              if ( v21 == (char *)-16LL && !v18 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = (char **)(v16 + 136LL * v19);
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_19;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_19:
          *v20 = v13;
          v20[1] = (char *)(v20 + 3);
          v20[2] = (char *)0x400000000LL;
          if ( *((_DWORD *)v10 + 4) )
          {
            v24 = v20;
            sub_1381A30((__int64)(v20 + 1), v10 + 1);
            v20 = v24;
          }
          v20[10] = (char *)0x400000000LL;
          v20[9] = (char *)(v20 + 11);
          if ( *((_DWORD *)v10 + 20) )
            sub_1381A30((__int64)(v20 + 9), v10 + 9);
          ++*(_DWORD *)(a1 + 16);
          v11 = (unsigned __int64)v10[9];
          if ( (char **)v11 != v10 + 11 )
            _libc_free(v11);
          v12 = (unsigned __int64)v10[1];
          if ( (char **)v12 != v10 + 3 )
            _libc_free(v12);
        }
        v10 += 17;
      }
      while ( v8 != (__int64 *)v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[17 * v22]; j != result; result += 17 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
