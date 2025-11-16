// Function: sub_6E1A60
// Address: 0x6e1a60
//
__int64 __fastcall sub_6E1A60(__int64 a1)
{
  if ( *(_BYTE *)(a1 + 8) == 1 )
    return a1 + 40;
  else
    return *(_QWORD *)(a1 + 24) + 84LL;
}
