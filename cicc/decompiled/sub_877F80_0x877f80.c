// Function: sub_877F80
// Address: 0x877f80
//
__int64 __fastcall sub_877F80(__int64 a1)
{
  unsigned __int8 v1; // al
  unsigned int v2; // r8d

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 80);
    if ( v1 != 17 )
      break;
    a1 = *(_QWORD *)(a1 + 88);
  }
  if ( v1 <= 0x11u )
  {
    v2 = 0;
    if ( (unsigned __int8)(v1 - 10) <= 1u )
      return *(unsigned __int8 *)(*(_QWORD *)(a1 + 88) + 174LL);
    return v2;
  }
  v2 = 0;
  if ( v1 != 20 )
    return v2;
  return *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 176LL) + 174LL);
}
