// Function: sub_240D590
// Address: 0x240d590
//
bool __fastcall sub_240D590(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx

  v2 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v2 == 5 )
    return *(_WORD *)(v2 + 2) != 53;
  else
    return *(_BYTE *)v2 != 82;
}
