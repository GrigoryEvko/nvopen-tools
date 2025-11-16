// Function: sub_A17060
// Address: 0xa17060
//
__int64 __fastcall sub_A17060(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  int v5; // r12d
  __int64 result; // rax

  v5 = a3 - sub_A3F3B0(a1 + 24);
  result = *(unsigned int *)(a4 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    sub_C8D5F0(a4, a4 + 16, result + 1, 4);
    result = *(unsigned int *)(a4 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a4 + 4 * result) = v5;
  ++*(_DWORD *)(a4 + 8);
  return result;
}
