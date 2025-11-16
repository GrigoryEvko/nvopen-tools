// Function: sub_6ECFC0
// Address: 0x6ecfc0
//
__int64 __fastcall sub_6ECFC0(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 result; // rax

  *a3 = 0;
  *a2 = 0;
  if ( *(_BYTE *)(a1 + 24) != 1 )
    return a1;
  result = a1;
  if ( *(_BYTE *)(a1 + 56) == 9 )
  {
    result = *(_QWORD *)(a1 + 72);
    if ( *(_BYTE *)(result + 24) != 1 )
      return a1;
  }
  do
  {
    if ( (*(_BYTE *)(result + 27) & 2) == 0 )
      break;
    if ( *(_BYTE *)(result + 56) != 14 )
      break;
    *a2 = a1;
    *a3 = result;
    result = *(_QWORD *)(result + 72);
  }
  while ( *(_BYTE *)(result + 24) == 1 );
  if ( !*a2 )
    return a1;
  return result;
}
