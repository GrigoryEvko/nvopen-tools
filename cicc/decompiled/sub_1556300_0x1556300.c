// Function: sub_1556300
// Address: 0x1556300
//
_QWORD *__fastcall sub_1556300(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // rbx
  _QWORD *i; // rdx
  __int64 *v10; // rax
  unsigned __int64 v11; // rdx
  int v12; // ecx
  int v13; // esi
  __int64 v14; // r8
  __int64 *v15; // r10
  int v16; // r14d
  unsigned int v17; // edi
  __int64 *v18; // rcx
  __int64 v19; // r9
  int v20; // edx
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
  result = (_QWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[2 * v3];
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
        *result = -1;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        while ( 1 )
        {
          v11 = *v10;
          if ( (unsigned __int64)*v10 <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          v10 += 2;
          if ( v8 == v10 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 0;
        v16 = 1;
        v17 = (v12 - 1) & (37 * v11);
        v18 = (__int64 *)(v14 + 16LL * v17);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != -1 )
          {
            if ( !v15 && v19 == -2 )
              v15 = v18;
            v17 = v13 & (v16 + v17);
            v18 = (__int64 *)(v14 + 16LL * v17);
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_14;
            ++v16;
          }
          if ( v15 )
            v18 = v15;
        }
LABEL_14:
        *v18 = v11;
        v20 = *((_DWORD *)v10 + 2);
        v10 += 2;
        *((_DWORD *)v18 + 2) = v20;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * *(unsigned int *)(a1 + 24)]; j != result; result += 2 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
