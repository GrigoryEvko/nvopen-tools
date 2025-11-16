// Function: sub_AA4320
// Address: 0xaa4320
//
__int64 __fastcall sub_AA4320(_QWORD *a1, _BYTE *a2, int a3)
{
  if ( a3 == 1 )
  {
    *a1 = a2;
    return 0;
  }
  else
  {
    if ( a3 == 2 )
      *(_BYTE *)a1 = *a2;
    return 0;
  }
}
