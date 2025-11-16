// Function: sub_892920
// Address: 0x892920
//
__int64 __fastcall sub_892920(__int64 a1)
{
  __int64 result; // rax

  if ( !a1 )
    return 0;
  result = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 152LL);
  if ( !result )
    return a1;
  return result;
}
