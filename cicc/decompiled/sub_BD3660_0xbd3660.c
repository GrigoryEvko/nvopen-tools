// Function: sub_BD3660
// Address: 0xbd3660
//
__int64 __fastcall sub_BD3660(__int64 a1, int a2)
{
  __int64 v2; // rdx
  unsigned __int8 v3; // al
  __int64 v4; // rdx

  v2 = *(_QWORD *)(a1 + 16);
  if ( !a2 )
    return 1;
  if ( !v2 )
    return 0;
  do
  {
    v3 = sub_BD3070(v2);
    v2 = *(_QWORD *)(v4 + 8);
    a2 -= v3;
    if ( !a2 )
      return 1;
  }
  while ( v2 );
  return 0;
}
