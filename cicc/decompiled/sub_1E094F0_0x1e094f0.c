// Function: sub_1E094F0
// Address: 0x1e094f0
//
__int64 __fastcall sub_1E094F0(_QWORD *a1, _DWORD *a2, int a3)
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
