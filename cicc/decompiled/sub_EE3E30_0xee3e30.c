// Function: sub_EE3E30
// Address: 0xee3e30
//
__int64 __fastcall sub_EE3E30(__int64 a1, unsigned __int8 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rcx
  __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  unsigned int v18; // edx

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
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v11) = 0;
  v12 = *(unsigned int *)(a1 + 12);
  v13 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v13;
  if ( v13 + 1 > v12 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v13 + 1, 4u, a5, a6);
    v13 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v13) = a3;
  v14 = HIDWORD(a3);
  v15 = *(unsigned int *)(a1 + 12);
  v16 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v16;
  if ( v16 + 1 > v15 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v16 + 1, 4u, a5, a6);
    v16 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v16) = v14;
  ++*(_DWORD *)(a1 + 8);
  if ( a4 )
  {
    v17 = (unsigned __int8 *)a5;
    v18 = a4;
  }
  else
  {
    v17 = 0;
    v18 = 0;
  }
  return sub_C653C0(a1, v17, v18);
}
