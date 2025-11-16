// Function: sub_138FFA0
// Address: 0x138ffa0
//
_DWORD *__fastcall sub_138FFA0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  int *v4; // r13
  unsigned __int64 v5; // rax
  _DWORD *result; // rax
  int *v7; // rdi
  _DWORD *i; // rdx
  int *v9; // rax
  unsigned int v10; // edx
  int v11; // ecx
  int v12; // esi
  __int64 v13; // r10
  int v14; // ebx
  unsigned int *v15; // r11
  unsigned int v16; // r8d
  unsigned int *v17; // rcx
  unsigned int v18; // r9d
  __int64 v19; // rdx
  _DWORD *j; // rdx

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
  result = (_DWORD *)sub_22077B0(12LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[3 * v3];
    for ( i = &result[3 * *(unsigned int *)(a1 + 24)]; i != result; result += 3 )
    {
      if ( result )
        *result = -1;
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        while ( 1 )
        {
          v10 = *v9;
          if ( (unsigned int)*v9 <= 0xFFFFFFFD )
            break;
          v9 += 3;
          if ( v7 == v9 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a1 + 8);
        v14 = 1;
        v15 = 0;
        v16 = (v11 - 1) & (37 * v10);
        v17 = (unsigned int *)(v13 + 12LL * v16);
        v18 = *v17;
        if ( v10 != *v17 )
        {
          while ( v18 != -1 )
          {
            if ( !v15 && v18 == -2 )
              v15 = v17;
            v16 = v12 & (v14 + v16);
            v17 = (unsigned int *)(v13 + 12LL * v16);
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
        v19 = *(_QWORD *)(v9 + 1);
        v9 += 3;
        *(_QWORD *)(v17 + 1) = v19;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v7 != v9 );
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[3 * *(unsigned int *)(a1 + 24)]; j != result; result += 3 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
