// Function: sub_C93580
// Address: 0xc93580
//
__int64 __fastcall sub_C93580(_QWORD *a1, char a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // rax

  v3 = a1[1];
  if ( a3 >= v3 )
    return -1;
  while ( a2 == *(_BYTE *)(*a1 + a3) )
  {
    if ( ++a3 >= v3 )
      return -1;
  }
  return a3;
}
