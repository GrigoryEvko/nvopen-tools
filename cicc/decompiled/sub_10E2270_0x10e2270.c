// Function: sub_10E2270
// Address: 0x10e2270
//
__int64 __fastcall sub_10E2270(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 4u, a5, a6);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
