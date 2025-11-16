// Function: sub_3215500
// Address: 0x3215500
//
__int64 __fastcall sub_3215500(__int64 a1, __int64 a2, __int16 a3)
{
  char v3; // al

  switch ( a3 )
  {
    case 1:
      return *(unsigned __int8 *)(a2 + 2);
    case 6:
      return 4;
    case 7:
      return 8;
    case 14:
    case 23:
      v3 = *(_BYTE *)(a2 + 3);
      if ( !v3 )
        return 4;
      if ( v3 != 1 )
LABEL_7:
        BUG();
      return 8;
    default:
      goto LABEL_7;
  }
}
