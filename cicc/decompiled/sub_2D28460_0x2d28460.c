// Function: sub_2D28460
// Address: 0x2d28460
//
unsigned __int64 __fastcall sub_2D28460(__int64 a1)
{
  bool v1; // zf
  unsigned __int64 v2; // rdi

  v1 = (a1 & 4) == 0;
  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 )
    return sub_2D283E0(v2);
  else
    return sub_2D283A0(v2);
}
