// Function: sub_AA4350
// Address: 0xaa4350
//
__int64 __fastcall sub_AA4350(_QWORD *a1, _BYTE *a2, int a3)
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
