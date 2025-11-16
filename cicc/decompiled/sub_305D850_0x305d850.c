// Function: sub_305D850
// Address: 0x305d850
//
__int64 __fastcall sub_305D850(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  result = *(unsigned int *)(a2 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), result + 1, 8u, a5, a6);
    result = *(unsigned int *)(a2 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = sub_30608C0;
  ++*(_DWORD *)(a2 + 8);
  return result;
}
