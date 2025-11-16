// Function: sub_17155A0
// Address: 0x17155a0
//
_DWORD *__fastcall sub_17155A0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  int *v4; // r13
  unsigned __int64 v5; // rdi
  _DWORD *result; // rax
  __int64 v7; // rdx
  int *v8; // r14
  _DWORD *i; // rdx
  int *v10; // rbx
  int v11; // eax
  int v12; // edx
  int v13; // esi
  __int64 v14; // r8
  int *v15; // r9
  int v16; // r10d
  unsigned int v17; // ecx
  int *v18; // rdx
  int v19; // edi
  __int64 v20; // rdi
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
  result = (_DWORD *)sub_22077B0(32LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[8 * v3];
    for ( i = &result[8 * v7]; i != result; result += 8 )
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
            v15 = 0;
            v16 = 1;
            v17 = (v12 - 1) & (37 * v11);
            v18 = (int *)(v14 + 32LL * v17);
            v19 = *v18;
            if ( v11 != *v18 )
            {
              while ( v19 != -1 )
              {
                if ( !v15 && v19 == -2 )
                  v15 = v18;
                v17 = v13 & (v16 + v17);
                v18 = (int *)(v14 + 32LL * v17);
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
            *((_QWORD *)v18 + 1) = *((_QWORD *)v10 + 1);
            *((_QWORD *)v18 + 2) = *((_QWORD *)v10 + 2);
            *((_QWORD *)v18 + 3) = *((_QWORD *)v10 + 3);
            *((_QWORD *)v10 + 3) = 0;
            *((_QWORD *)v10 + 1) = 0;
            *((_QWORD *)v10 + 2) = 0;
            ++*(_DWORD *)(a1 + 16);
            v20 = *((_QWORD *)v10 + 1);
            if ( v20 )
              break;
          }
          v10 += 8;
          if ( v8 == v10 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        j_j___libc_free_0(v20, *((_QWORD *)v10 + 3) - v20);
        v10 += 8;
      }
      while ( v8 != v10 );
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[8 * *(unsigned int *)(a1 + 24)]; j != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
