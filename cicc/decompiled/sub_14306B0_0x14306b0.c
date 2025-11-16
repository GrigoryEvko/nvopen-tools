// Function: sub_14306B0
// Address: 0x14306b0
//
_QWORD *__fastcall sub_14306B0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // rdi
  _QWORD *i; // rdx
  __int64 *v10; // rax
  int v11; // ecx
  int v12; // esi
  unsigned __int64 v13; // rdx
  __int64 v14; // r10
  _QWORD *v15; // r11
  int v16; // ebx
  unsigned int v17; // r9d
  _QWORD *v18; // r8
  unsigned __int64 v19; // rcx
  __int64 v20; // rdx
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
  result = (_QWORD *)sub_22077B0(8LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    v8 = &v4[v3];
    *(_QWORD *)(a1 + 16) = 0;
    for ( i = &result[v7]; i != result; ++result )
    {
      if ( result )
        *result = -8;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        while ( (*v10 & 0xFFFFFFFFFFFFFFF0LL) == 0xFFFFFFFFFFFFFFF0LL )
        {
          if ( v8 == ++v10 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
        {
          MEMORY[0] = *v10;
          BUG();
        }
        v12 = v11 - 1;
        v13 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 0;
        v16 = 1;
        v17 = v13 & (v11 - 1);
        v18 = (_QWORD *)(v14 + 8LL * v17);
        v19 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v13 != v19 )
        {
          while ( v19 != -8 )
          {
            if ( v19 == -16 && !v15 )
              v15 = v18;
            v17 = v12 & (v16 + v17);
            v18 = (_QWORD *)(v14 + 8LL * v17);
            v19 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v13 == v19 )
              goto LABEL_14;
            ++v16;
          }
          if ( v15 )
            v18 = v15;
        }
LABEL_14:
        v20 = *v10++;
        *v18 = v20;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[*(unsigned int *)(a1 + 24)]; j != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
