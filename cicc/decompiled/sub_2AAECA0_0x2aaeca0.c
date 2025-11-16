// Function: sub_2AAECA0
// Address: 0x2aaeca0
//
__int64 __fastcall sub_2AAECA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 result; // rax

  v6 = *(unsigned int *)(a1 + 16);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
  {
    sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v6 + 1, 8u, a5, a6);
    v6 = *(unsigned int *)(a1 + 16);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v6) = a2;
  ++*(_DWORD *)(a1 + 16);
  result = *(unsigned int *)(a2 + 24);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 28) )
  {
    sub_C8D5F0(a2 + 16, (const void *)(a2 + 32), result + 1, 8u, a5, a6);
    result = *(unsigned int *)(a2 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * result) = a1;
  ++*(_DWORD *)(a2 + 24);
  return result;
}
