// Function: sub_168E630
// Address: 0x168e630
//
__int64 __fastcall sub_168E630(__int64 a1, _BYTE *a2, int a3)
{
  if ( a3 == 8 || a3 == 20 )
  {
    sub_168E230(a1, a2, a3);
    if ( a3 != 12 )
      return 1;
  }
  else if ( a3 != 12 )
  {
    return 1;
  }
  sub_168E440(a1, a2);
  return 1;
}
