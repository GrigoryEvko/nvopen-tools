// Function: sub_16B55E0
// Address: 0x16b55e0
//
__int64 __fastcall sub_16B55E0(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)result )
  {
    sub_16CD150(a1, a1 + 16, 0, 8);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = *a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
