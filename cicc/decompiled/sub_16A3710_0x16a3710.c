// Function: sub_16A3710
// Address: 0x16a3710
//
__int64 __fastcall sub_16A3710(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)result )
  {
    sub_16CD150(a1, a1 + 16, 0, 1);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_BYTE *)(*(_QWORD *)a1 + result) = *a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
