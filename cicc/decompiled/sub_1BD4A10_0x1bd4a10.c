// Function: sub_1BD4A10
// Address: 0x1bd4a10
//
_QWORD *__fastcall sub_1BD4A10(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 *v7; // rbx
  _QWORD *i; // rdx
  __int64 *j; // r13
  __int64 v10; // rax
  int v11; // edx
  int v12; // esi
  __int64 v13; // r8
  int v14; // r10d
  _QWORD *v15; // r9
  unsigned int v16; // ecx
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdi
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 v22; // rdx
  _QWORD *k; // rdx

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
  result = (_QWORD *)sub_22077B0(88LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[11 * v3];
    for ( i = &result[11 * *(unsigned int *)(a1 + 24)]; i != result; result += 11 )
    {
      if ( result )
        *result = -8;
    }
    for ( j = v4; v7 != j; j += 11 )
    {
      v10 = *j;
      if ( *j != -16 && v10 != -8 )
      {
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a1 + 8);
        v14 = 1;
        v15 = 0;
        v16 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v17 = (_QWORD *)(v13 + 88LL * v16);
        v18 = *v17;
        if ( *v17 != v10 )
        {
          while ( v18 != -8 )
          {
            if ( !v15 && v18 == -16 )
              v15 = v17;
            v16 = v12 & (v14 + v16);
            v17 = (_QWORD *)(v13 + 88LL * v16);
            v18 = *v17;
            if ( v10 == *v17 )
              goto LABEL_13;
            ++v14;
          }
          if ( v15 )
            v17 = v15;
        }
LABEL_13:
        *v17 = v10;
        v19 = (__int64)(v17 + 1);
        v20 = v17 + 3;
        v21 = v17 + 11;
        *(v21 - 10) = 0;
        *(v21 - 9) = 1;
        do
        {
          if ( v20 )
            *v20 = -8;
          v20 += 2;
        }
        while ( v20 != v21 );
        sub_1BD48A0(v19, (__int64)(j + 1));
        ++*(_DWORD *)(a1 + 16);
        if ( (j[2] & 1) == 0 )
          j___libc_free_0(j[3]);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[11 * v22]; k != result; result += 11 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
