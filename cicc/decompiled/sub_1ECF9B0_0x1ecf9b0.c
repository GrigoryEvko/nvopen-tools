// Function: sub_1ECF9B0
// Address: 0x1ecf9b0
//
_DWORD *__fastcall sub_1ECF9B0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  int *v4; // r12
  unsigned __int64 v5; // rdi
  _DWORD *result; // rax
  __int64 v7; // rdx
  int *v8; // r14
  _DWORD *i; // rdx
  int *v10; // rbx
  int *v11; // rax
  __int64 v12; // rdx
  _DWORD *j; // rdx
  int *v14; // [rsp+8h] [rbp-38h] BYREF

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(int **)(a1 + 8);
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
  result = (_DWORD *)sub_22077B0(8LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[2 * v3];
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      while ( *v10 == -1 )
      {
        if ( v10[1] == -1 )
        {
          v10 += 2;
          if ( v8 == v10 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        else
        {
LABEL_11:
          sub_1ECE260(a1, v10, &v14);
          v11 = v14;
          *v14 = *v10;
          v11[1] = v10[1];
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          v10 += 2;
          if ( v8 == v10 )
            return (_DWORD *)j___libc_free_0(v4);
        }
      }
      if ( *v10 == -2 && v10[1] == -2 )
        goto LABEL_12;
      goto LABEL_11;
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    v12 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v12]; j != result; result += 2 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
  }
  return result;
}
