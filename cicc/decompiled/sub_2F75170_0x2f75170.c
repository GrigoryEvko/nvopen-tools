// Function: sub_2F75170
// Address: 0x2f75170
//
__int64 __fastcall sub_2F75170(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_DWORD *)((*(_QWORD *)(a1 + 440) & 0xFFFFFFFFFFFFFFF8LL) + 24)
         | (unsigned int)(*(__int64 *)(a1 + 440) >> 1) & 3;
  if ( (unsigned int)result > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) )
  {
    *(_QWORD *)(a1 + 440) = 0;
    *(_DWORD *)(a1 + 32) = 0;
  }
  return result;
}
