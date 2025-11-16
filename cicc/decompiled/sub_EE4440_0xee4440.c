// Function: sub_EE4440
// Address: 0xee4440
//
__int64 __fastcall sub_EE4440(
        __int64 a1,
        unsigned __int8 a2,
        __int64 a3,
        unsigned __int8 *a4,
        unsigned __int64 a5,
        __int64 a6)
{
  __int64 v9; // r8
  int v10; // ebx
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rdx

  v9 = a2;
  v10 = a6;
  v11 = *(unsigned int *)(a1 + 8);
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v11 + 1, 4u, a2, a6);
    v11 = *(unsigned int *)(a1 + 8);
    v9 = a2;
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v11) = v9;
  v12 = *(unsigned int *)(a1 + 12);
  v13 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v13;
  if ( v13 + 1 > v12 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v13 + 1, 4u, v9, a6);
    v13 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v13) = 0;
  ++*(_DWORD *)(a1 + 8);
  if ( a3 )
    sub_C653C0(a1, a4, a3);
  else
    sub_C653C0(a1, 0, 0);
  v16 = *(unsigned int *)(a1 + 8);
  if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v16 + 1, 4u, v14, v15);
    v16 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v16) = a5;
  v17 = HIDWORD(a5);
  v18 = *(unsigned int *)(a1 + 12);
  v19 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v19;
  if ( v19 + 1 > v18 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v19 + 1, 4u, v14, v15);
    v19 = *(unsigned int *)(a1 + 8);
  }
  v20 = *(_QWORD *)a1;
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v19) = v17;
  ++*(_DWORD *)(a1 + 8);
  return sub_D953B0(a1, v10, v20, v18, v14, v15);
}
