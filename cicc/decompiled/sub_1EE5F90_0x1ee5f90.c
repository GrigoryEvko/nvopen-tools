// Function: sub_1EE5F90
// Address: 0x1ee5f90
//
__int64 __fastcall sub_1EE5F90(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_DWORD *)((*(_QWORD *)(a1 + 184) & 0xFFFFFFFFFFFFFFF8LL) + 24)
         | (unsigned int)(*(__int64 *)(a1 + 184) >> 1) & 3;
  if ( (unsigned int)result > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) )
  {
    *(_QWORD *)(a1 + 184) = 0;
    *(_DWORD *)(a1 + 32) = 0;
  }
  return result;
}
