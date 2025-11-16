// Function: sub_B2FBD0
// Address: 0xb2fbd0
//
__int64 __fastcall sub_B2FBD0(__int64 a1)
{
  if ( *(_BYTE *)a1 || a1 + 72 == (*(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  else
    return sub_B2D610(a1, 23);
}
