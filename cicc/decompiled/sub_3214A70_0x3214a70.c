// Function: sub_3214A70
// Address: 0x3214a70
//
__int64 __fastcall sub_3214A70(unsigned __int16 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // r13d
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  int v10; // r13d
  __int64 result; // rax
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // rcx

  v6 = *(unsigned int *)(a2 + 8);
  v7 = *a1;
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v6 + 1, 4u, a5, a6);
    v6 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v6) = v7;
  v8 = *(unsigned int *)(a2 + 12);
  v9 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = v9;
  v10 = a1[1];
  if ( v9 + 1 > v8 )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v9 + 1, 4u, a5, a6);
    v9 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v9) = v10;
  result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = result;
  if ( a1[1] == 33 )
  {
    v12 = *((_QWORD *)a1 + 1);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), result + 1, 4u, a5, a6);
      result = *(unsigned int *)(a2 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v12;
    v13 = HIDWORD(v12);
    v14 = *(unsigned int *)(a2 + 12);
    result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = result;
    if ( result + 1 > v14 )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), result + 1, 4u, a5, a6);
      result = *(unsigned int *)(a2 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v13;
    ++*(_DWORD *)(a2 + 8);
  }
  return result;
}
