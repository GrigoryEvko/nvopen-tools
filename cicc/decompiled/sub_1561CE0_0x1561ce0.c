// Function: sub_1561CE0
// Address: 0x1561ce0
//
__int64 __fastcall sub_1561CE0(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // r12

  if ( (*a2 & *a1) != 0 )
    return 1;
  v2 = a1[4];
  if ( a1 + 2 == (_QWORD *)v2 )
    return 0;
  while ( !(unsigned __int8)sub_1561C30((__int64)a2, *(_BYTE **)(v2 + 32), *(_QWORD *)(v2 + 40)) )
  {
    v2 = sub_220EF30(v2);
    if ( a1 + 2 == (_QWORD *)v2 )
      return 0;
  }
  return 1;
}
