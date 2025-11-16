// Function: sub_2BDB880
// Address: 0x2bdb880
//
__int64 __fastcall sub_2BDB880(_QWORD *a1, _QWORD *a2, int a3)
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
