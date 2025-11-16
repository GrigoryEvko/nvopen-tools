// Function: sub_1E16450
// Address: 0x1e16450
//
__int64 __fastcall sub_1E16450(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( **(_WORD **)(a1 + 16) == 1 )
    return ((unsigned int)*(_QWORD *)(*(_QWORD *)(a1 + 32) + 64LL) >> 1) & 1;
  return result;
}
