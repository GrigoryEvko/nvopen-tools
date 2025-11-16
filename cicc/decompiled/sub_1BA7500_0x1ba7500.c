// Function: sub_1BA7500
// Address: 0x1ba7500
//
_DWORD *__fastcall sub_1BA7500(__int64 a1, int a2)
{
  __int64 v3; // rbx
  int *v4; // r13
  unsigned int v5; // eax
  _DWORD *result; // rax
  int *v7; // r14
  _DWORD *i; // rdx
  int *v9; // rbx
  unsigned int v10; // edx
  int v11; // eax
  int v12; // ecx
  __int64 v13; // r8
  int *v14; // r9
  int v15; // r10d
  unsigned int v16; // esi
  int *v17; // rax
  int v18; // edi
  __int64 v19; // rcx
  __int64 v20; // rdx
  _DWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(int **)(a1 + 8);
  v5 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_22077B0(40LL * v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[10 * v3];
    for ( i = &result[10 * *(unsigned int *)(a1 + 24)]; i != result; result += 10 )
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
          v9 += 10;
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
        v14 = 0;
        v15 = 1;
        v16 = (v11 - 1) & (37 * v10);
        v17 = (int *)(v13 + 40LL * v16);
        v18 = *v17;
        if ( v10 != *v17 )
        {
          while ( v18 != -1 )
          {
            if ( !v14 && v18 == -2 )
              v14 = v17;
            v16 = v12 & (v15 + v16);
            v17 = (int *)(v13 + 40LL * v16);
            v18 = *v17;
            if ( v10 == *v17 )
              goto LABEL_14;
            ++v15;
          }
          if ( v14 )
            v17 = v14;
        }
LABEL_14:
        *((_QWORD *)v17 + 3) = 0;
        *((_QWORD *)v17 + 2) = 0;
        v17[8] = 0;
        *v17 = v10;
        *((_QWORD *)v17 + 1) = 1;
        v19 = *((_QWORD *)v9 + 2);
        ++*((_QWORD *)v9 + 1);
        v20 = *((_QWORD *)v17 + 2);
        v9 += 10;
        *((_QWORD *)v17 + 2) = v19;
        LODWORD(v19) = *(v9 - 4);
        *((_QWORD *)v9 - 3) = v20;
        LODWORD(v20) = v17[6];
        v17[6] = v19;
        LODWORD(v19) = *(v9 - 3);
        *(v9 - 4) = v20;
        LODWORD(v20) = v17[7];
        v17[7] = v19;
        LODWORD(v19) = *(v9 - 2);
        *(v9 - 3) = v20;
        LODWORD(v20) = v17[8];
        v17[8] = v19;
        *(v9 - 2) = v20;
        ++*(_DWORD *)(a1 + 16);
        j___libc_free_0(*((_QWORD *)v9 - 3));
      }
      while ( v7 != v9 );
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[10 * *(unsigned int *)(a1 + 24)]; j != result; result += 10 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
