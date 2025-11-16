// Function: sub_E5B9B0
// Address: 0xe5b9b0
//
__int64 __fastcall sub_E5B9B0(__int64 a1)
{
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    return *(_QWORD *)(a1 - 8) + 24LL;
  else
    return 0;
}
