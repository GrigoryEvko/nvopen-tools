// Function: sub_325DB00
// Address: 0x325db00
//
__int64 __fastcall sub_325DB00(_QWORD *a1, _DWORD *a2, int a3)
{
  if ( a3 == 1 )
  {
    *a1 = a2;
    return 0;
  }
  else
  {
    if ( a3 == 2 )
      *(_DWORD *)a1 = *a2;
    return 0;
  }
}
