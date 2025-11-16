// Function: sub_2183150
// Address: 0x2183150
//
_QWORD *__fastcall sub_2183150(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rcx
  __int64 *v8; // r15
  _QWORD *i; // rdx
  char **v10; // rbx
  char *v11; // rax
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // r8
  int v15; // r10d
  char **v16; // r9
  __int64 v17; // rdx
  char **v18; // rdi
  char *v19; // rsi
  unsigned __int64 v20; // rdi
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(56LL * v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[7 * v3];
    for ( i = &result[7 * v7]; i != result; result += 7 )
    {
      if ( result )
        *result = -8;
    }
    if ( v8 != v4 )
    {
      v10 = (char **)v4;
      do
      {
        v11 = *v10;
        if ( *v10 != (char *)-16LL && v11 != (char *)-8LL )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *v10;
            BUG();
          }
          v13 = (unsigned int)(v12 - 1);
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = (unsigned int)v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (char **)(v14 + 56 * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != (char *)-8LL )
            {
              if ( !v16 && v19 == (char *)-16LL )
                v16 = v18;
              v17 = (unsigned int)v13 & (v15 + (_DWORD)v17);
              v18 = (char **)(v14 + 56LL * (unsigned int)v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_14;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_14:
          *v18 = v11;
          v18[1] = (char *)(v18 + 3);
          v18[2] = (char *)0x400000000LL;
          if ( *((_DWORD *)v10 + 4) )
            sub_217D360((__int64)(v18 + 1), v10 + 1, v17, v13, v14, (int)v16);
          ++*(_DWORD *)(a1 + 16);
          v20 = (unsigned __int64)v10[1];
          if ( (char **)v20 != v10 + 3 )
            _libc_free(v20);
        }
        v10 += 7;
      }
      while ( v8 != (__int64 *)v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * *(unsigned int *)(a1 + 24)]; j != result; result += 7 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
