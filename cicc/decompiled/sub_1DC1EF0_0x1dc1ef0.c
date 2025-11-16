// Function: sub_1DC1EF0
// Address: 0x1dc1ef0
//
unsigned int *__fastcall sub_1DC1EF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned int *result; // rax
  __int64 v7; // rcx
  unsigned int *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r13
  _QWORD *v12; // rax
  __int64 v13; // rax
  unsigned int *v14; // rdx
  __int64 v15; // rax
  int v16; // eax

  result = (unsigned int *)*(unsigned int *)(a1 + 16);
  v7 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)result )
  {
    v9 = *(unsigned int **)(a1 + 8);
    do
    {
      while ( 1 )
      {
        v16 = *(_DWORD *)(*(_QWORD *)(a2 + 24) + 4LL * (*v9 >> 5));
        if ( !_bittest(&v16, *v9) )
          break;
        ++v9;
        result = (unsigned int *)(v7 + 4LL * *(unsigned int *)(a1 + 16));
        if ( v9 == result )
          return result;
      }
      if ( a3 )
      {
        v10 = *(unsigned int *)(a3 + 8);
        v11 = *v9;
        if ( (unsigned int)v10 >= *(_DWORD *)(a3 + 12) )
        {
          sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, a5, a6);
          v10 = *(unsigned int *)(a3 + 8);
        }
        v12 = (_QWORD *)(*(_QWORD *)a3 + 16 * v10);
        *v12 = v11;
        v12[1] = a2;
        ++*(_DWORD *)(a3 + 8);
        v7 = *(_QWORD *)(a1 + 8);
      }
      v13 = *(unsigned int *)(a1 + 16);
      v14 = (unsigned int *)(v7 + 4 * v13 - 4);
      if ( v14 != v9 )
      {
        *v9 = *v14;
        *(_BYTE *)(*(_QWORD *)(a1 + 56) + *(unsigned int *)(*(_QWORD *)(a1 + 8) + 4LL * *(unsigned int *)(a1 + 16) - 4)) = ((__int64)v9 - *(_QWORD *)(a1 + 8)) >> 2;
        LODWORD(v13) = *(_DWORD *)(a1 + 16);
        v7 = *(_QWORD *)(a1 + 8);
      }
      v15 = (unsigned int)(v13 - 1);
      *(_DWORD *)(a1 + 16) = v15;
      result = (unsigned int *)(v7 + 4 * v15);
    }
    while ( v9 != result );
  }
  return result;
}
