// Function: sub_853DF0
// Address: 0x853df0
//
__int64 __fastcall sub_853DF0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 result; // rax

  v1 = unk_4D03D28;
  if ( unk_4D03D28 )
    unk_4D03D28 = *unk_4D03D28;
  else
    v1 = sub_823970(104);
  *(_QWORD *)v1 = 0;
  sub_7ADF70(v1 + 16, 1);
  *(_QWORD *)(v1 + 8) = a1;
  *(_QWORD *)(v1 + 80) = 0;
  *(_QWORD *)(v1 + 88) = 0;
  v2 = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(v1 + 64) = 0;
  *(_QWORD *)(v1 + 48) = v2;
  *(_QWORD *)(v1 + 56) = v2;
  *(_BYTE *)(v1 + 72) = *(_BYTE *)(v1 + 72) & 0xF0 | 1;
  switch ( *(_BYTE *)(a1 + 8) )
  {
    case 1:
    case 2:
    case 3:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
    case 0xE:
    case 0xF:
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17:
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
      goto LABEL_5;
    case 4:
      *(_WORD *)(v1 + 96) = 0;
LABEL_5:
      result = v1;
      break;
    case 0x1C:
    case 0x1D:
      sub_726C20((_BYTE *)(v1 + 96));
      result = v1;
      break;
    default:
      sub_721090();
  }
  return result;
}
