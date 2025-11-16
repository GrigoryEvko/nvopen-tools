// Function: sub_16BD3E0
// Address: 0x16bd3e0
//
__int64 __fastcall sub_16BD3E0(__int64 a1, int a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 12) )
  {
    sub_16CD150(a1, a1 + 16, 0, 4);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
