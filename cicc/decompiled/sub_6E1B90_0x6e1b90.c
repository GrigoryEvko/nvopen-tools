// Function: sub_6E1B90
// Address: 0x6e1b90
//
__int64 __fastcall sub_6E1B90(_QWORD *a1)
{
  _QWORD *v1; // rbx

  if ( !a1 )
    return 0;
  v1 = a1;
  do
  {
    if ( sub_6E1B40((__int64)v1) )
      return 1;
    v1 = (_QWORD *)*v1;
  }
  while ( v1 );
  return 0;
}
