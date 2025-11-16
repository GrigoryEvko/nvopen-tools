// Function: sub_2C1AB20
// Address: 0x2c1ab20
//
char __fastcall sub_2C1AB20(__int64 a1)
{
  int v1; // eax
  __int64 v2; // r12

  switch ( *(_BYTE *)(a1 + 8) )
  {
    case 1:
    case 6:
    case 7:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
    case 0xF:
    case 0x10:
    case 0x11:
    case 0x17:
    case 0x18:
    case 0x19:
    case 0x1B:
    case 0x1C:
    case 0x21:
    case 0x22:
      LOBYTE(v1) = 0;
      break;
    case 4:
    case 5:
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      LOBYTE(v1) = sub_2C1AA80(a1);
      break;
    case 9:
      LOBYTE(v1) = sub_B46970(*(unsigned __int8 **)(a1 + 136));
      break;
    case 0xE:
      v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 56) - 1)) + 40LL);
      if ( sub_2C1AA80(a1) || !(unsigned __int8)sub_B2D610(v2, 41) )
        LOBYTE(v1) = 1;
      else
        v1 = sub_B2D610(v2, 76) ^ 1;
      break;
    case 0x12:
      LOBYTE(v1) = *(_BYTE *)(a1 + 178);
      break;
    default:
      LOBYTE(v1) = 1;
      break;
  }
  return v1;
}
