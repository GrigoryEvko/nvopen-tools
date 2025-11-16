// Function: sub_2E32160
// Address: 0x2e32160
//
__int64 __fastcall sub_2E32160(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 72);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
  {
    sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), result + 1, 8u, a5, a6);
    result = *(unsigned int *)(a1 + 72);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * result) = a2;
  ++*(_DWORD *)(a1 + 72);
  return result;
}
