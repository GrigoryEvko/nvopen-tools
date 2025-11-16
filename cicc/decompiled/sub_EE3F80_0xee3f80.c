// Function: sub_EE3F80
// Address: 0xee3f80
//
__int64 __fastcall sub_EE3F80(
        __int64 a1,
        unsigned __int8 a2,
        __int64 a3,
        unsigned __int8 *a4,
        unsigned __int64 a5,
        __int64 a6)
{
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rcx
  __int64 result; // rax

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
  ++*(_DWORD *)(a1 + 8);
  if ( a3 )
  {
    sub_C653C0(a1, a4, a3);
    v14 = *(unsigned int *)(a1 + 8);
    v15 = v14 + 1;
    if ( v14 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
      goto LABEL_7;
  }
  else
  {
    sub_C653C0(a1, 0, 0);
    v14 = *(unsigned int *)(a1 + 8);
    v15 = v14 + 1;
    if ( v14 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
      goto LABEL_7;
  }
  sub_C8D5F0(a1, (const void *)(a1 + 16), v15, 4u, v12, v13);
  v14 = *(unsigned int *)(a1 + 8);
LABEL_7:
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v14) = a5;
  v16 = HIDWORD(a5);
  v17 = *(unsigned int *)(a1 + 12);
  result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = result;
  if ( result + 1 > v17 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 4u, v12, v13);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = v16;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
