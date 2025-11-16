// Function: sub_33C9670
// Address: 0x33c9670
//
__int64 __fastcall sub_33C9670(__int64 a1, int a2, unsigned __int64 a3, unsigned __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // rax
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  __int64 *v15; // r14
  __int64 result; // rax
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rax
  int v23; // r13d

  v9 = *(unsigned int *)(a1 + 8);
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v9 + 1, 4u, a5, a6);
    v9 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v9) = a2;
  v10 = *(unsigned int *)(a1 + 12);
  v11 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v11;
  if ( v11 + 1 > v10 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v11 + 1, 4u, a5, a6);
    v11 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v11) = a3;
  v12 = HIDWORD(a3);
  v13 = *(unsigned int *)(a1 + 12);
  v14 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v14;
  if ( v14 + 1 > v13 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v14 + 1, 4u, a5, a6);
    v14 = *(unsigned int *)(a1 + 8);
  }
  v15 = (__int64 *)&a4[2 * a5];
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v14) = v12;
  result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  for ( *(_DWORD *)(a1 + 8) = result; v15 != (__int64 *)a4; *(_DWORD *)(a1 + 8) = result )
  {
    v17 = *a4;
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 4u, a5, a6);
      result = *(unsigned int *)(a1 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = v17;
    v18 = HIDWORD(v17);
    v19 = *(unsigned int *)(a1 + 12);
    v20 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v20;
    if ( v20 + 1 > v19 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v20 + 1, 4u, a5, a6);
      v20 = *(unsigned int *)(a1 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a1 + 4 * v20) = v18;
    v21 = *(unsigned int *)(a1 + 12);
    v22 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v22;
    v23 = *((_DWORD *)a4 + 2);
    if ( v22 + 1 > v21 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v22 + 1, 4u, a5, a6);
      v22 = *(unsigned int *)(a1 + 8);
    }
    a4 += 2;
    *(_DWORD *)(*(_QWORD *)a1 + 4 * v22) = v23;
    result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  }
  return result;
}
