// Function: sub_1525B90
// Address: 0x1525b90
//
__int64 __fastcall sub_1525B90(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)result )
  {
    sub_16CD150(a1, a1 + 16, 0, 4);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = *a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
