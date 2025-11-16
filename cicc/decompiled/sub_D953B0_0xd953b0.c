// Function: sub_D953B0
// Address: 0xd953b0
//
__int64 __fastcall sub_D953B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rcx
  __int64 result; // rax

  v6 = *(unsigned int *)(a1 + 8);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v6 + 1, 4u, a5, a6);
    v6 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v6) = a2;
  v7 = *(unsigned int *)(a1 + 12);
  result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = result;
  if ( result + 1 > v7 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 4u, a5, a6);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = HIDWORD(a2);
  ++*(_DWORD *)(a1 + 8);
  return result;
}
