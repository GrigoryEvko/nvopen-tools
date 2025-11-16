// Function: sub_B47F80
// Address: 0xb47f80
//
__int64 __fastcall sub_B47F80(_BYTE *a1)
{
  __int64 v1; // r13

  switch ( *a1 )
  {
    case 0x1E:
      v1 = sub_B55CC0();
      break;
    case 0x1F:
      v1 = sub_B55D10();
      break;
    case 0x20:
      v1 = sub_B55D60();
      break;
    case 0x21:
      v1 = sub_B55DA0();
      break;
    case 0x22:
      v1 = sub_B55DE0();
      break;
    case 0x23:
      v1 = sub_B56060();
      break;
    case 0x24:
      v1 = sub_B561C0();
      break;
    case 0x25:
      v1 = sub_B560A0();
      break;
    case 0x26:
      v1 = sub_B560F0();
      break;
    case 0x27:
      v1 = sub_B56130();
      break;
    case 0x28:
      v1 = sub_B55F20();
      break;
    case 0x29:
      v1 = sub_B54AE0();
      break;
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
    case 0x35:
    case 0x36:
    case 0x37:
    case 0x38:
    case 0x39:
    case 0x3A:
    case 0x3B:
      v1 = sub_B54B10();
      break;
    case 0x3C:
      v1 = sub_B54D40();
      break;
    case 0x3D:
      v1 = sub_B54E10();
      break;
    case 0x3E:
      v1 = sub_B54EC0();
      break;
    case 0x3F:
      v1 = sub_B54A90();
      break;
    case 0x40:
      v1 = sub_B55130();
      break;
    case 0x41:
      v1 = sub_B54F70();
      break;
    case 0x42:
      v1 = sub_B55060();
      break;
    case 0x43:
      v1 = sub_B551A0();
      break;
    case 0x44:
      v1 = sub_B55210();
      break;
    case 0x45:
      v1 = sub_B55280();
      break;
    case 0x46:
      v1 = sub_B554B0();
      break;
    case 0x47:
      v1 = sub_B55520();
      break;
    case 0x48:
      v1 = sub_B553D0();
      break;
    case 0x49:
      v1 = sub_B55440();
      break;
    case 0x4A:
      v1 = sub_B552F0();
      break;
    case 0x4B:
      v1 = sub_B55360();
      break;
    case 0x4C:
      v1 = sub_B55590();
      break;
    case 0x4D:
      v1 = sub_B55600();
      break;
    case 0x4E:
      v1 = sub_B55670();
      break;
    case 0x4F:
      v1 = sub_B556E0();
      break;
    case 0x50:
    case 0x51:
      v1 = sub_B56170();
      break;
    case 0x52:
      v1 = sub_B54C00();
      break;
    case 0x53:
      v1 = sub_B54B40();
      break;
    case 0x54:
      v1 = sub_B55C40();
      break;
    case 0x55:
      v1 = sub_B55750();
      break;
    case 0x56:
      v1 = sub_B55890();
      break;
    case 0x57:
    case 0x58:
      sub_408AE4();
    case 0x59:
      v1 = sub_B55A10();
      break;
    case 0x5A:
      v1 = sub_B55AE0();
      break;
    case 0x5B:
      v1 = sub_B55B40();
      break;
    case 0x5C:
      v1 = sub_B55BC0();
      break;
    case 0x5D:
      v1 = sub_B54CC0();
      break;
    case 0x5E:
      v1 = sub_B54D00();
      break;
    case 0x5F:
      v1 = sub_B55C80();
      break;
    case 0x60:
      v1 = sub_B56210();
      break;
    default:
      BUG();
  }
  *(_BYTE *)(v1 + 1) = a1[1] & 0xFE | *(_BYTE *)(v1 + 1) & 1;
  sub_B47C00(v1, (__int64)a1, 0, 0);
  return v1;
}
