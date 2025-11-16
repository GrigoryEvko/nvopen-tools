// Function: sub_143E380
// Address: 0x143e380
//
_DWORD *__fastcall sub_143E380(__int64 a1, int a2)
{
  __int64 v3; // rbx
  int *v4; // r12
  unsigned __int64 v5; // rax
  _DWORD *result; // rax
  __int64 v7; // rdx
  int *v8; // r14
  _DWORD *i; // rdx
  int *v10; // rbx
  int v11; // eax
  int v12; // edx
  int v13; // edx
  __int64 v14; // r8
  int v15; // r10d
  _DWORD *v16; // r9
  unsigned int v17; // ecx
  _DWORD *v18; // rsi
  int v19; // edi
  unsigned __int64 v20; // rdi
  __int64 v21; // rdx
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
  result = (_DWORD *)sub_22077B0(80LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[20 * v3];
    for ( i = &result[20 * v7]; i != result; result += 20 )
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
          if ( (unsigned int)*v10 <= 0xFFFFFFFD )
          {
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
            v18 = (_DWORD *)(v14 + 80LL * v17);
            v19 = *v18;
            if ( *v18 != v11 )
            {
              while ( v19 != -1 )
              {
                if ( !v16 && v19 == -2 )
                  v16 = v18;
                v17 = v13 & (v15 + v17);
                v18 = (_DWORD *)(v14 + 80LL * v17);
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
            sub_16CCEE0(v18 + 2, v18 + 12, 4, v10 + 2);
            ++*(_DWORD *)(a1 + 16);
            v20 = *((_QWORD *)v10 + 3);
            if ( v20 != *((_QWORD *)v10 + 2) )
              break;
          }
          v10 += 20;
          if ( v8 == v10 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        _libc_free(v20);
        v10 += 20;
      }
      while ( v8 != v10 );
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[20 * v21]; j != result; result += 20 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
