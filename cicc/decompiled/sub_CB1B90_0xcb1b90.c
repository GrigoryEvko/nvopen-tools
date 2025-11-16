// Function: sub_CB1B90
// Address: 0xcb1b90
//
__int64 __fastcall sub_CB1B90(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // eax
  int v6; // eax
  int i; // ebx

  if ( *(_BYTE *)(a1 + 93) )
    sub_CB1B10(a1, ", ", 2u);
  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 && v4 < *(_DWORD *)(a1 + 80) )
  {
    sub_CB1B10(a1, "\n", 1u);
    v6 = *(_DWORD *)(a1 + 84);
    if ( v6 > 0 )
    {
      for ( i = 0; i < v6; ++i )
      {
        sub_CB1B10(a1, " ", 1u);
        v6 = *(_DWORD *)(a1 + 84);
      }
    }
    *(_DWORD *)(a1 + 80) = v6;
    sub_CB1B10(a1, "  ", 2u);
  }
  *a3 = 0;
  return 1;
}
