// Function: sub_1441E30
// Address: 0x1441e30
//
_BOOL8 __fastcall sub_1441E30(__int64 a1)
{
  if ( *(_BYTE *)(a1 + 24) )
    return *(_QWORD *)(a1 + 16) != 0;
  sub_1441BF0(a1);
  return *(_BYTE *)(a1 + 24) && *(_QWORD *)(a1 + 16) != 0;
}
