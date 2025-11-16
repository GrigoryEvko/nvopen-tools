// Function: sub_1ECFB30
// Address: 0x1ecfb30
//
_QWORD *__fastcall sub_1ECFB30(__int64 a1, int a2)
{
  __int64 v3; // r13
  __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r13
  _QWORD *i; // rdx
  __int64 *v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  volatile signed __int32 *v14; // rdi
  __int64 v15; // rdx
  _QWORD *j; // rdx
  __int64 *v17; // [rsp+8h] [rbp-38h] BYREF

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
  result = (_QWORD *)sub_22077B0(32LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[4 * v3];
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
      }
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      while ( *v10 == -8 )
      {
        if ( v10[1] == -8 )
        {
          v10 += 4;
          if ( v8 == v10 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        else
        {
LABEL_11:
          sub_1ECE370(a1, v10, &v17);
          v11 = v17;
          *v17 = *v10;
          v11[1] = v10[1];
          v12 = v10[2];
          v11[3] = 0;
          v11[2] = v12;
          v13 = v10[3];
          v10[3] = 0;
          v11[3] = v13;
          v10[2] = 0;
          ++*(_DWORD *)(a1 + 16);
          v14 = (volatile signed __int32 *)v10[3];
          if ( v14 )
            sub_A191D0(v14);
LABEL_13:
          v10 += 4;
          if ( v8 == v10 )
            return (_QWORD *)j___libc_free_0(v4);
        }
      }
      if ( *v10 == -16 && v10[1] == -16 )
        goto LABEL_13;
      goto LABEL_11;
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v15 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v15]; j != result; result += 4 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
      }
    }
  }
  return result;
}
