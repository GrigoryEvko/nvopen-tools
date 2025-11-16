// Function: sub_EE40D0
// Address: 0xee40d0
//
__int64 __fastcall sub_EE40D0(
        __int64 a1,
        unsigned __int8 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rax
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rcx
  __int64 result; // rax

  v8 = *(unsigned int *)(a1 + 8);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v8 + 1, 4u, a5, a6);
    v8 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v8) = a2;
  v9 = *(unsigned int *)(a1 + 12);
  v10 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v10;
  if ( v10 + 1 > v9 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v10 + 1, 4u, a5, a6);
    v10 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v10) = 0;
  v11 = *(unsigned int *)(a1 + 12);
  v12 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v12;
  if ( v12 + 1 > v11 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v12 + 1, 4u, a5, a6);
    v12 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v12) = a3;
  v13 = HIDWORD(a3);
  v14 = *(unsigned int *)(a1 + 12);
  v15 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v15;
  if ( v15 + 1 > v14 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v15 + 1, 4u, a5, a6);
    v15 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v15) = v13;
  v16 = *(unsigned int *)(a1 + 12);
  v17 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v17;
  if ( v17 + 1 > v16 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v17 + 1, 4u, a5, a6);
    v17 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v17) = a4;
  v18 = HIDWORD(a4);
  v19 = *(unsigned int *)(a1 + 12);
  result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = result;
  if ( result + 1 > v19 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 4u, a5, a6);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = v18;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
