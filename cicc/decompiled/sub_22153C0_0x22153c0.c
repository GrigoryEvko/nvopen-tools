// Function: sub_22153C0
// Address: 0x22153c0
//
__int64 __fastcall sub_22153C0(_QWORD *a1, char a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // rcx

  v3 = *(_QWORD *)(*a1 - 24LL);
  if ( a3 >= v3 )
    return -1;
  while ( *(_BYTE *)(*a1 + a3) == a2 )
  {
    if ( ++a3 == v3 )
      return -1;
  }
  return a3;
}
