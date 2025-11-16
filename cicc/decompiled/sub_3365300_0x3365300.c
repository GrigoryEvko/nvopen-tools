// Function: sub_3365300
// Address: 0x3365300
//
__int64 __fastcall sub_3365300(_QWORD *a1, _QWORD *a2, int a3)
{
  if ( a3 == 1 )
  {
    *a1 = a2;
    return 0;
  }
  else
  {
    if ( a3 == 2 )
      *a1 = *a2;
    return 0;
  }
}
