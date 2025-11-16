// Function: sub_35108D0
// Address: 0x35108d0
//
__int64 __fastcall sub_35108D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 i; // r8

  v1 = *(_QWORD *)(a1 + 56);
  v2 = a1 + 48;
  for ( i = 0; v2 != v1; v1 = *(_QWORD *)(v1 + 8) )
  {
    while ( 1 )
    {
      if ( *(_WORD *)(v1 + 68) != 68 && *(_WORD *)(v1 + 68) )
        i += (*(_QWORD *)(*(_QWORD *)(v1 + 16) + 24LL) & 0x10LL) == 0;
      if ( (*(_BYTE *)v1 & 4) == 0 )
        break;
      v1 = *(_QWORD *)(v1 + 8);
      if ( v2 == v1 )
        return i;
    }
    while ( (*(_BYTE *)(v1 + 44) & 8) != 0 )
      v1 = *(_QWORD *)(v1 + 8);
  }
  return i;
}
