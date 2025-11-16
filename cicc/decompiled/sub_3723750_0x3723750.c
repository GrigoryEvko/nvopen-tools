// Function: sub_3723750
// Address: 0x3723750
//
__int64 __fastcall sub_3723750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // r12d
  __int64 result; // rax
  int *v9; // r12
  int *i; // r13
  int v11; // r15d
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  int v14; // r15d

  v6 = *(unsigned int *)(a2 + 8);
  v7 = *(_DWORD *)(a1 + 8);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v6 + 1, 4u, a5, a6);
    v6 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v6) = v7;
  result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = result;
  v9 = *(int **)(a1 + 16);
  for ( i = &v9[2 * *(unsigned int *)(a1 + 24)]; i != v9; *(_DWORD *)(a2 + 8) = result )
  {
    v11 = *v9;
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), result + 1, 4u, a5, a6);
      result = *(unsigned int *)(a2 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v11;
    v12 = *(unsigned int *)(a2 + 12);
    v13 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v13;
    v14 = *((unsigned __int16 *)v9 + 2);
    if ( v13 + 1 > v12 )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v13 + 1, 4u, a5, a6);
      v13 = *(unsigned int *)(a2 + 8);
    }
    v9 += 2;
    *(_DWORD *)(*(_QWORD *)a2 + 4 * v13) = v14;
    result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  }
  return result;
}
