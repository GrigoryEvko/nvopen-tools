// Function: sub_3214BB0
// Address: 0x3214bb0
//
__int64 __fastcall sub_3214BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // ebx
  unsigned __int64 v8; // rcx
  __int64 result; // rax
  int v10; // ebx
  __int64 v11; // rdx
  unsigned __int16 *v12; // rbx
  unsigned __int16 *i; // r13
  unsigned __int16 *v14; // rdi

  v6 = *(unsigned int *)(a2 + 8);
  v7 = *(unsigned __int16 *)(a1 + 12);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v6 + 1, 4u, a5, a6);
    v6 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v6) = v7;
  v8 = *(unsigned int *)(a2 + 12);
  result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = result;
  v10 = *(unsigned __int8 *)(a1 + 14);
  if ( result + 1 > v8 )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), result + 1, 4u, a5, a6);
    result = *(unsigned int *)(a2 + 8);
  }
  v11 = *(_QWORD *)a2;
  *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v10;
  ++*(_DWORD *)(a2 + 8);
  v12 = *(unsigned __int16 **)(a1 + 16);
  for ( i = &v12[8 * *(unsigned int *)(a1 + 24)]; i != v12; result = sub_3214A70(v14, a2, v11, v8, a5, a6) )
  {
    v14 = v12;
    v12 += 8;
  }
  return result;
}
