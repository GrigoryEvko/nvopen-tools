// Function: sub_1D02420
// Address: 0x1d02420
//
__int64 __fastcall sub_1D02420(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_DWORD *)(*(_QWORD *)(a1 + 96) + 4LL * *(unsigned int *)(a2 + 192)) = 0;
  result = *(unsigned int *)(*(_QWORD *)(a1 + 96) + 4LL * *(unsigned int *)(a2 + 192));
  if ( !(_DWORD)result )
    return sub_1D01FD0(a2, (__int64 *)(a1 + 96));
  return result;
}
