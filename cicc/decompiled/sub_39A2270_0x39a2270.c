// Function: sub_39A2270
// Address: 0x39a2270
//
__int64 __fastcall sub_39A2270(__int64 a1)
{
  __int64 result; // rax

  switch ( *(_WORD *)(*(_QWORD *)(a1 + 80) + 24LL) )
  {
    case 0:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
      goto LABEL_3;
    case 1:
    case 2:
    case 4:
    case 0x34:
    case 0x35:
    case 0x36:
      return 0;
    case 3:
    case 5:
    case 6:
    case 9:
    case 0xA:
    case 0xD:
    case 0xF:
      if ( (unsigned __int16)sub_398C0A0(*(_QWORD *)(a1 + 200)) > 3u )
        goto LABEL_6;
      goto LABEL_3;
    case 7:
    case 8:
      goto LABEL_6;
    case 0xB:
    case 0x12:
    case 0x13:
    case 0x14:
      return -(__int64)((unsigned __int16)sub_398C0A0(*(_QWORD *)(a1 + 200)) < 4u);
    case 0xC:
    case 0x10:
    case 0x11:
      return -(__int64)((unsigned __int16)sub_398C0A0(*(_QWORD *)(a1 + 200)) < 3u);
    case 0xE:
      if ( (unsigned __int16)sub_398C0A0(*(_QWORD *)(a1 + 200)) <= 2u )
        goto LABEL_3;
      goto LABEL_6;
    case 0x15:
    case 0x16:
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x1D:
    case 0x1E:
    case 0x20:
    case 0x21:
    case 0x24:
    case 0x25:
      return -(__int64)((unsigned __int16)sub_398C0A0(*(_QWORD *)(a1 + 200)) < 5u);
    case 0x17:
    case 0x1F:
    case 0x22:
    case 0x23:
      if ( (unsigned __int16)sub_398C0A0(*(_QWORD *)(a1 + 200)) <= 4u )
LABEL_3:
        result = -1;
      else
LABEL_6:
        result = 1;
      break;
    default:
      result = -1;
      break;
  }
  return result;
}
