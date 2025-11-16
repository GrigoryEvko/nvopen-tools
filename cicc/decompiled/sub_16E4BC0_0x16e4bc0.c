// Function: sub_16E4BC0
// Address: 0x16e4bc0
//
__int64 __fastcall sub_16E4BC0(__int64 a1)
{
  int v1; // eax
  int v3; // eax
  int i; // ebx

  if ( *(_BYTE *)(a1 + 93) )
    sub_16E4B40(a1, ", ", 2u);
  v1 = *(_DWORD *)(a1 + 24);
  if ( !v1 || v1 >= *(_DWORD *)(a1 + 80) )
    return 1;
  sub_16E4B40(a1, "\n", 1u);
  v3 = *(_DWORD *)(a1 + 84);
  if ( v3 > 0 )
  {
    for ( i = 0; i < v3; ++i )
    {
      sub_16E4B40(a1, " ", 1u);
      v3 = *(_DWORD *)(a1 + 84);
    }
  }
  *(_DWORD *)(a1 + 80) = v3;
  sub_16E4B40(a1, "  ", 2u);
  return 1;
}
