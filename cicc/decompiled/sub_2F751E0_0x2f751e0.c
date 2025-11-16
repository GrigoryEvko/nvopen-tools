// Function: sub_2F751E0
// Address: 0x2f751e0
//
__int64 __fastcall sub_2F751E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_DWORD *)((*(_QWORD *)(a1 + 448) & 0xFFFFFFFFFFFFFFF8LL) + 24)
         | (unsigned int)(*(__int64 *)(a1 + 448) >> 1) & 3;
  if ( (unsigned int)result <= (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) )
  {
    *(_QWORD *)(a1 + 448) = 0;
    *(_DWORD *)(a1 + 32) = 0;
  }
  return result;
}
