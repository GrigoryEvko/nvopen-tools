// Function: sub_2C1ABE0
// Address: 0x2c1abe0
//
__int64 __fastcall sub_2C1ABE0(__int64 a1)
{
  int v1; // eax
  __int64 result; // rax
  unsigned __int64 v3; // rax

  switch ( *(_BYTE *)(a1 + 8) )
  {
    case 0:
    case 6:
    case 7:
    case 0xB:
    case 0xC:
    case 0xF:
    case 0x10:
    case 0x11:
    case 0x15:
    case 0x16:
    case 0x17:
    case 0x18:
    case 0x19:
    case 0x1B:
    case 0x1C:
    case 0x21:
      result = 0;
      break;
    case 4:
      v1 = *(unsigned __int8 *)(a1 + 160);
      if ( (unsigned int)(v1 - 13) > 0x11 )
      {
        switch ( (char)v1 )
        {
          case '5':
          case '9':
          case 'E':
          case 'F':
          case 'L':
          case 'M':
          case 'R':
          case 'S':
          case 'T':
          case 'U':
          case 'V':
            JUMPOUT(0x2C1AA70);
          default:
            JUMPOUT(0x2C1AA60);
        }
      }
      JUMPOUT(0x2C1AA66);
    case 9:
      v3 = *(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v3 = **(_QWORD **)v3;
      result = sub_B46420(*(_QWORD *)(v3 + 40));
      break;
    case 0xE:
      result = (unsigned int)sub_B2DD10(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48)
                                                              + 8LL * (unsigned int)(*(_DWORD *)(a1 + 56) - 1))
                                                  + 40LL))
             ^ 1;
      break;
    case 0x12:
      result = *(unsigned __int8 *)(a1 + 176);
      break;
    default:
      result = 1;
      break;
  }
  return result;
}
