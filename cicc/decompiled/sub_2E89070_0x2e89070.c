// Function: sub_2E89070
// Address: 0x2e89070
//
__int64 __fastcall sub_2E89070(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 <= 1 )
    return ((unsigned int)*(_QWORD *)(*(_QWORD *)(a1 + 32) + 64LL) >> 1) & 1;
  return result;
}
