// Function: sub_8670F0
// Address: 0x8670f0
//
__int64 sub_8670F0()
{
  unsigned int v0; // r8d

  v0 = 0;
  if ( !qword_4F04C18 )
    return v0;
  if ( *((_BYTE *)qword_4F04C18 + 42) )
    return v0;
  v0 = 1;
  if ( *(_QWORD *)(qword_4F04C18[1] + 24LL) )
    return v0;
  else
    return *((_BYTE *)qword_4F04C18 + 52) != 0;
}
