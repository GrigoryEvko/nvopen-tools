// Function: sub_15A0F90
// Address: 0x15a0f90
//
__int64 __fastcall sub_15A0F90(__int64 a1, __int64 a2)
{
  if ( *(_BYTE *)(a2 + 16) != 13 )
    return 0;
  if ( *(_DWORD *)(a2 + 32) <= 0x40u )
    return sub_15A0A60(a1, *(_QWORD *)(a2 + 24));
  return sub_15A0A60(a1, **(_QWORD **)(a2 + 24));
}
