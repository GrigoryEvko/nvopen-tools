// Function: sub_731540
// Address: 0x731540
//
__int64 __fastcall sub_731540(__int64 a1)
{
  __int64 result; // rax
  char v3; // al
  __int64 v4; // rdx

  while ( 2 )
  {
    switch ( *(_BYTE *)(a1 + 24) )
    {
      case 0:
      case 3:
      case 5:
      case 0xB:
      case 0x14:
        return 1;
      case 1:
        if ( (*(_BYTE *)(a1 + 58) & 1) != 0 )
          goto LABEL_3;
        switch ( *(_BYTE *)(a1 + 56) )
        {
          case 3:
          case 4:
          case 7:
          case 8:
          case 0xD:
          case 0x13:
          case 0x21:
          case 0x22:
          case 0x25:
          case 0x26:
          case 0x47:
          case 0x48:
          case 0x49:
          case 0x4A:
          case 0x4B:
          case 0x4C:
          case 0x4D:
          case 0x4E:
          case 0x4F:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x53:
          case 0x54:
          case 0x55:
          case 0x5B:
          case 0x5C:
          case 0x5D:
          case 0x5F:
          case 0x61:
          case 0x64:
          case 0x65:
          case 0x67:
          case 0x69:
          case 0x6A:
          case 0x6B:
          case 0x6C:
          case 0x6D:
          case 0x6E:
          case 0x70:
          case 0x74:
            return 1;
          case 0xE:
          case 0xF:
            result = sub_8D3A70(*(_QWORD *)a1);
            if ( (_DWORD)result )
              result = (*(_BYTE *)(*(_QWORD *)(a1 + 72) + 25LL) & 3) != 0;
            break;
          case 0x19:
            a1 = *(_QWORD *)(a1 + 72);
            continue;
          case 0x5E:
          case 0x60:
            result = 1;
            if ( !dword_4D04410 )
            {
              v4 = *(_QWORD *)(a1 + 72);
              if ( (*(_BYTE *)(v4 + 25) & 3) == 0 )
                result = *(_BYTE *)(v4 + 24) == 0;
            }
            break;
          default:
            goto LABEL_3;
        }
        break;
      case 2:
        return *(_BYTE *)(*(_QWORD *)(a1 + 56) + 173LL) == 12;
      case 0x18:
        return *(_DWORD *)(a1 + 56) != 0;
      case 0x1B:
        v3 = *(_BYTE *)(a1 + 64);
        a1 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL);
        if ( (v3 & 1) == 0 )
          a1 = *(_QWORD *)(a1 + 16);
        continue;
      default:
LABEL_3:
        result = 0;
        break;
    }
    return result;
  }
}
