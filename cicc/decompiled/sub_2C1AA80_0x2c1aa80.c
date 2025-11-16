// Function: sub_2C1AA80
// Address: 0x2c1aa80
//
char __fastcall sub_2C1AA80(__int64 a1)
{
  char v1; // r8
  int v2; // eax
  int v3; // eax
  unsigned __int64 v4; // rdx

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
    case 0x13:
    case 0x14:
    case 0x17:
    case 0x18:
    case 0x19:
    case 0x1B:
    case 0x1C:
    case 0x21:
      LOBYTE(v3) = 0;
      break;
    case 4:
      v1 = 0;
      v2 = *(unsigned __int8 *)(a1 + 160);
      if ( (unsigned int)(v2 - 13) <= 0x11 )
      {
LABEL_13:
        LOBYTE(v3) = v1;
      }
      else
      {
        switch ( (char)v2 )
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
            LOBYTE(v3) = 0;
            break;
          default:
            v1 = 1;
            goto LABEL_13;
        }
      }
      break;
    case 5:
      LOBYTE(v3) = 1 - ((*(_BYTE *)(a1 + 104) == 0) - 1) != *(_DWORD *)(a1 + 56);
      break;
    case 9:
      v4 = *(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v4 = **(_QWORD **)v4;
      LOBYTE(v3) = sub_B46490(*(_QWORD *)(v4 + 40));
      break;
    case 0xE:
      v3 = sub_B2DCE0(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 56) - 1))
                                + 40LL))
         ^ 1;
      break;
    case 0x12:
      LOBYTE(v3) = *(_BYTE *)(a1 + 177);
      break;
    default:
      LOBYTE(v3) = 1;
      break;
  }
  return v3;
}
