// Function: sub_D97090
// Address: 0xd97090
//
__int64 __fastcall sub_D97090(__int64 a1, __int64 a2)
{
  if ( *(_BYTE *)(a2 + 8) == 12 )
    return a2;
  else
    return sub_AE4570(*(_QWORD *)(a1 + 8), a2);
}
