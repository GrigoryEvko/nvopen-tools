// Function: sub_37EAB50
// Address: 0x37eab50
//
__int64 __fastcall sub_37EAB50(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 i; // r12

  v2 = *(_QWORD *)(a1 + 544);
  if ( v2 != *(_QWORD *)(a1 + 552) )
    *(_QWORD *)(a1 + 552) = v2;
  v3 = *(_QWORD *)(a2 + 56);
  for ( i = a2 + 48; i != v3; v3 = *(_QWORD *)(v3 + 8) )
  {
    while ( 1 )
    {
      if ( (unsigned __int16)(*(_WORD *)(v3 + 68) - 14) > 4u )
        sub_37EA930(a1, v3);
      if ( (*(_BYTE *)v3 & 4) == 0 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( i == v3 )
        return sub_37EA550(a1, a2);
    }
    while ( (*(_BYTE *)(v3 + 44) & 8) != 0 )
      v3 = *(_QWORD *)(v3 + 8);
  }
  return sub_37EA550(a1, a2);
}
