// Function: sub_1699540
// Address: 0x1699540
//
__int64 __fastcall sub_1699540(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // di
  __int64 result; // rax

  v3 = *(_BYTE *)(a1 + 18);
  switch ( (*(_BYTE *)(a2 + 18) & 7) + 4 * (v3 & 7) )
  {
    case 0:
      result = 1;
      if ( ((v3 ^ *(_BYTE *)(a2 + 18)) & 8) != 0 )
        goto LABEL_5;
      return result;
    case 1:
    case 4:
    case 5:
    case 6:
    case 7:
    case 9:
    case 0xD:
      return 3;
    case 2:
    case 3:
    case 0xB:
      goto LABEL_5;
    case 8:
    case 0xC:
    case 0xE:
      if ( (*(_BYTE *)(a2 + 18) & 8) != 0 )
        return 2;
      return 0;
    case 0xA:
      if ( ((v3 ^ *(_BYTE *)(a2 + 18)) & 8) != 0 )
      {
LABEL_5:
        if ( (v3 & 8) != 0 )
          return 0;
        return 2;
      }
      result = sub_1698CF0(a1, a2);
      if ( (*(_BYTE *)(a1 + 18) & 8) == 0 )
        return result;
      if ( !(_DWORD)result )
        return 2;
      if ( (_DWORD)result == 2 )
        return 0;
      return result;
    case 0xF:
      return 1;
  }
}
