// Function: sub_3101C60
// Address: 0x3101c60
//
__int64 __fastcall sub_3101C60(_QWORD *a1, _QWORD *a2, int a3)
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
