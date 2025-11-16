// Function: sub_D76090
// Address: 0xd76090
//
__int64 __fastcall sub_D76090(_QWORD *a1, _QWORD *a2, int a3)
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
