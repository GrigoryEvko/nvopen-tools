// Function: sub_EE3C10
// Address: 0xee3c10
//
__int64 __fastcall sub_EE3C10(__int64 a1, unsigned __int8 a2, __int64 a3, unsigned __int8 *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rax

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
  ++*(_DWORD *)(a1 + 8);
  if ( a3 )
    return sub_C653C0(a1, a4, a3);
  else
    return sub_C653C0(a1, 0, 0);
}
