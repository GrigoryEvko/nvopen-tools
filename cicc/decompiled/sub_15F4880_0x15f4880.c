// Function: sub_15F4880
// Address: 0x15f4880
//
__int64 __fastcall sub_15F4880(__int64 a1)
{
  __int64 v1; // r13

  switch ( *(_BYTE *)(a1 + 16) )
  {
    case 0x18:
    case 0x58:
      v1 = sub_1601680();
      break;
    case 0x19:
      v1 = sub_16016C0();
      break;
    case 0x1A:
      v1 = sub_1601700();
      break;
    case 0x1B:
      v1 = sub_1601740();
      break;
    case 0x1C:
      v1 = sub_1601780();
      break;
    case 0x1D:
      v1 = sub_16017C0();
      break;
    case 0x1E:
      v1 = sub_16018B0();
      break;
    case 0x1F:
      v1 = sub_16019F0();
      break;
    case 0x20:
      v1 = sub_16018F0();
      break;
    case 0x21:
      v1 = sub_1601930();
      break;
    case 0x22:
      v1 = sub_1601970();
      break;
    case 0x23:
    case 0x24:
    case 0x25:
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
    case 0x34:
      v1 = sub_1600640();
      break;
    case 0x35:
      v1 = sub_1600850();
      break;
    case 0x36:
      v1 = sub_1600930();
      break;
    case 0x37:
      v1 = sub_16009D0();
      break;
    case 0x38:
      v1 = sub_1600600();
      break;
    case 0x39:
      v1 = sub_1600BF0();
      break;
    case 0x3A:
      v1 = sub_1600A70();
      break;
    case 0x3B:
      v1 = sub_1600B40();
      break;
    case 0x3C:
      v1 = sub_1600C60();
      break;
    case 0x3D:
      v1 = sub_1600CC0();
      break;
    case 0x3E:
      v1 = sub_1600D20();
      break;
    case 0x3F:
      v1 = sub_1600F00();
      break;
    case 0x40:
      v1 = sub_1600F60();
      break;
    case 0x41:
      v1 = sub_1600E40();
      break;
    case 0x42:
      v1 = sub_1600EA0();
      break;
    case 0x43:
      v1 = sub_1600D80();
      break;
    case 0x44:
      v1 = sub_1600DE0();
      break;
    case 0x45:
      v1 = sub_1600FC0();
      break;
    case 0x46:
      v1 = sub_1601020();
      break;
    case 0x47:
      v1 = sub_1601080();
      break;
    case 0x48:
      v1 = sub_16010E0();
      break;
    case 0x49:
    case 0x4A:
      v1 = sub_16019B0();
      break;
    case 0x4B:
      v1 = sub_1600720();
      break;
    case 0x4C:
      v1 = sub_1600670();
      break;
    case 0x4D:
      v1 = sub_1601640();
      break;
    case 0x4E:
      v1 = sub_1601140();
      break;
    case 0x4F:
      v1 = sub_1601230();
      break;
    case 0x50:
    case 0x51:
      sub_41A0AE();
    case 0x52:
      v1 = sub_1601410();
      break;
    case 0x53:
      v1 = sub_1601500();
      break;
    case 0x54:
      v1 = sub_1601560();
      break;
    case 0x55:
      v1 = sub_16015D0();
      break;
    case 0x56:
      v1 = sub_16007D0();
      break;
    case 0x57:
      v1 = sub_1600810();
      break;
  }
  *(_BYTE *)(v1 + 17) = *(_BYTE *)(a1 + 17) & 0xFE | *(_BYTE *)(v1 + 17) & 1;
  sub_15F4370(v1, a1, 0, 0);
  return v1;
}
