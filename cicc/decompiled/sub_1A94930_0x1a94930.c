// Function: sub_1A94930
// Address: 0x1a94930
//
__int64 __fastcall sub_1A94930(_QWORD *a1, _QWORD *a2)
{
  if ( *(_DWORD *)a1 == 1 )
  {
    if ( !*(_DWORD *)a2 || *(_DWORD *)a2 == 1 && a2[1] == a1[1] )
      return *a1;
    return 2;
  }
  else
  {
    if ( *(_DWORD *)a1 == 2 )
      return *a1;
    return *a2;
  }
}
