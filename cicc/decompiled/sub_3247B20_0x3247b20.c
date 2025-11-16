// Function: sub_3247B20
// Address: 0x3247b20
//
__int64 __fastcall sub_3247B20(__int64 a1)
{
  __int64 result; // rax

  switch ( *(_WORD *)(*(_QWORD *)(a1 + 80) + 16LL) )
  {
    case 0:
      goto LABEL_3;
    case 1:
    case 2:
    case 4:
      return 0;
    case 3:
    case 5:
    case 6:
    case 9:
    case 0xA:
    case 0xD:
    case 0xF:
      if ( (unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) > 3u )
        goto LABEL_6;
      goto LABEL_3;
    case 7:
    case 8:
      goto LABEL_6;
    case 0xB:
    case 0x12:
    case 0x13:
    case 0x14:
      return -(__int64)((unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) < 4u);
    case 0xC:
    case 0x10:
    case 0x11:
      return -(__int64)((unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) < 3u);
    case 0xE:
      if ( (unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) <= 2u )
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
      return -(__int64)((unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) < 5u);
    case 0x17:
    case 0x1F:
    case 0x22:
    case 0x23:
      if ( (unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) <= 4u )
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
