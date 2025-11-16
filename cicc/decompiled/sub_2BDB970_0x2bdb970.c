// Function: sub_2BDB970
// Address: 0x2bdb970
//
__int64 __fastcall sub_2BDB970(_QWORD *a1, _WORD *a2, int a3)
{
  if ( a3 == 1 )
  {
    *a1 = a2;
    return 0;
  }
  else
  {
    if ( a3 == 2 )
      *(_WORD *)a1 = *a2;
    return 0;
  }
}
