// Function: sub_B1A4E0
// Address: 0xb1a4e0
//
__int64 __fastcall sub_B1A4E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, a1 + 16, result + 1, 8);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
