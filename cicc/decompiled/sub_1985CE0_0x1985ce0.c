// Function: sub_1985CE0
// Address: 0x1985ce0
//
_DWORD *__fastcall sub_1985CE0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  int *v4; // r13
  unsigned __int64 v5; // rdi
  _DWORD *result; // rax
  __int64 v7; // rdx
  int *v8; // rsi
  _DWORD *i; // rdx
  int *v10; // rax
  int v11; // edx
  int v12; // ecx
  int v13; // ecx
  __int64 v14; // r10
  int v15; // ebx
  _DWORD *v16; // r11
  unsigned int v17; // edi
  _DWORD *v18; // r8
  int v19; // r9d
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
  result = (_DWORD *)sub_22077B0(4LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    v8 = &v4[v3];
    *(_QWORD *)(a1 + 16) = 0;
    for ( i = &result[v7]; i != result; ++result )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        while ( 1 )
        {
          v11 = *v10;
          if ( (unsigned int)(*v10 + 0x7FFFFFFF) <= 0xFFFFFFFD )
            break;
          if ( v8 == ++v10 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 1;
        v16 = 0;
        v17 = v13 & (37 * v11);
        v18 = (_DWORD *)(v14 + 4LL * v17);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != 0x7FFFFFFF )
          {
            if ( !v16 && v19 == 0x80000000 )
              v16 = v18;
            v17 = v13 & (v15 + v17);
            v18 = (_DWORD *)(v14 + 4LL * v17);
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_14;
            ++v15;
          }
          if ( v16 )
            v18 = v16;
        }
LABEL_14:
        ++v10;
        *v18 = v11;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v8 != v10 );
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[*(unsigned int *)(a1 + 24)]; j != result; ++result )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
  }
  return result;
}
