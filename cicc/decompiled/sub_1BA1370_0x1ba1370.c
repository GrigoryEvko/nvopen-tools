// Function: sub_1BA1370
// Address: 0x1ba1370
//
_QWORD *__fastcall sub_1BA1370(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 *v7; // r15
  _QWORD *i; // rdx
  char **v9; // rbx
  char *v10; // rax
  int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // r8
  int v14; // r10d
  char **v15; // r9
  __int64 v16; // rdx
  char **v17; // rdi
  char *v18; // rsi
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(40LL * v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[5 * v3];
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
    {
      if ( result )
        *result = -8;
    }
    if ( v7 != v4 )
    {
      v9 = (char **)v4;
      do
      {
        v10 = *v9;
        if ( *v9 != (char *)-16LL && v10 != (char *)-8LL )
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
          v17 = (char **)(v13 + 40 * v16);
          v18 = *v17;
          if ( v10 != *v17 )
          {
            while ( v18 != (char *)-8LL )
            {
              if ( !v15 && v18 == (char *)-16LL )
                v15 = v17;
              v16 = (unsigned int)v12 & (v14 + (_DWORD)v16);
              v17 = (char **)(v13 + 40LL * (unsigned int)v16);
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
          v17[1] = (char *)(v17 + 3);
          v17[2] = (char *)0x200000000LL;
          if ( *((_DWORD *)v9 + 4) )
            sub_1B8E3C0((__int64)(v17 + 1), v9 + 1, v16, v12, v13, (int)v15);
          ++*(_DWORD *)(a1 + 16);
          v19 = (unsigned __int64)v9[1];
          if ( (char **)v19 != v9 + 3 )
            _libc_free(v19);
        }
        v9 += 5;
      }
      while ( v7 != (__int64 *)v9 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v20 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * v20]; j != result; result += 5 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
