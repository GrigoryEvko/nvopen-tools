// Function: sub_1F6C240
// Address: 0x1f6c240
//
__int64 __fastcall sub_1F6C240(_QWORD *a1, _DWORD *a2, int a3)
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
