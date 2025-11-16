// Function: sub_3913870
// Address: 0x3913870
//
__int64 __fastcall sub_3913870(_BYTE *a1)
{
  if ( (*a1 & 4) != 0 )
    return *((_QWORD *)a1 - 1) + 16LL;
  else
    return 0;
}
