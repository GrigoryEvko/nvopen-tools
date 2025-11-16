// Function: sub_1A94A00
// Address: 0x1a94a00
//
__int64 __fastcall sub_1A94A00(__int64 a1)
{
  __int64 **v1; // r8
  unsigned __int8 v2; // al

  while ( 1 )
  {
    v1 = *(__int64 ***)a1;
    v2 = *(_BYTE *)(a1 + 16);
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
      break;
LABEL_6:
    if ( v2 == 17 )
      return a1;
    if ( v2 <= 0x10u )
      return sub_1598F00(v1);
    switch ( v2 )
    {
      case '6':
      case 'T':
      case 'U':
        return a1;
      case '8':
LABEL_23:
        a1 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
        break;
      case 'G':
        a1 = *(_QWORD *)(a1 - 24);
        break;
      default:
        return a1;
    }
  }
  while ( 1 )
  {
    if ( v2 == 17 )
      return a1;
    if ( v2 <= 0x10u )
      return sub_1599A20(v1);
    if ( (unsigned __int8)(v2 - 60) > 0xCu )
    {
      if ( v2 != 54 )
      {
        if ( v2 == 56 )
          goto LABEL_23;
        if ( v2 <= 0x17u || v2 != 78 && v2 != 29 && v2 != 58 && v2 != 86 )
          return a1;
      }
      return a1;
    }
    a1 = sub_1649C60(a1);
    v1 = *(__int64 ***)a1;
    v2 = *(_BYTE *)(a1 + 16);
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
      goto LABEL_6;
  }
}
