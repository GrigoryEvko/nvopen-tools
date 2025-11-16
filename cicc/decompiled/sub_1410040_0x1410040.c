// Function: sub_1410040
// Address: 0x1410040
//
__int64 __fastcall sub_1410040(__int64 *a1, __int64 a2)
{
  __int64 result; // rax

  switch ( *(_BYTE *)(a2 + 16) )
  {
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1C:
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
    case 0x37:
    case 0x38:
    case 0x39:
    case 0x3A:
    case 0x3B:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x47:
    case 0x48:
    case 0x49:
    case 0x4A:
    case 0x4B:
    case 0x4C:
    case 0x52:
    case 0x54:
    case 0x55:
    case 0x57:
    case 0x58:
      result = sub_140D0A0();
      break;
    case 0x1D:
      result = (__int64)sub_140D4F0((__int64)a1, a2 & 0xFFFFFFFFFFFFFFFBLL);
      break;
    case 0x35:
      result = (__int64)sub_140D220(a1, a2);
      break;
    case 0x36:
      result = sub_140D090();
      break;
    case 0x46:
      result = sub_140D080();
      break;
    case 0x4D:
      result = sub_140F4E0((__int64)a1, a2);
      break;
    case 0x4E:
      result = (__int64)sub_140F1A0((__int64)a1, a2);
      break;
    case 0x4F:
      result = sub_1410D30();
      break;
    case 0x50:
    case 0x51:
      result = sub_140D0A0();
      break;
    case 0x53:
      result = sub_140D060();
      break;
    case 0x56:
      result = sub_140D070();
      break;
  }
  return result;
}
