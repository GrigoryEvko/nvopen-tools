// Function: sub_34A0140
// Address: 0x34a0140
//
__int64 __fastcall sub_34A0140(__int64 a1)
{
  if ( *(_WORD *)(a1 + 68) == 14 )
    return *(_QWORD *)(a1 + 32);
  else
    return *(_QWORD *)(a1 + 32) + 80LL;
}
