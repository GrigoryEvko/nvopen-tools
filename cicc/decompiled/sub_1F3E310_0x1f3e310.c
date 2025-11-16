// Function: sub_1F3E310
// Address: 0x1f3e310
//
__int64 __fastcall sub_1F3E310(_BYTE *a1)
{
  __int64 result; // rax

  switch ( *a1 )
  {
    case 0:
    case 1:
    case 0x6F:
    case 0x70:
    case 0x71:
    case 0x72:
      result = 0;
      break;
    case 2:
    case 0xE:
    case 0x38:
      result = 1;
      break;
    case 3:
    case 0x11:
    case 0x18:
    case 0x3B:
    case 0x3E:
      result = 8;
      break;
    case 4:
    case 8:
    case 0x12:
    case 0x19:
    case 0x21:
    case 0x3C:
    case 0x3F:
    case 0x44:
      result = 16;
      break;
    case 5:
    case 9:
    case 0x13:
    case 0x1A:
    case 0x22:
    case 0x29:
    case 0x3D:
    case 0x40:
    case 0x45:
    case 0x4A:
    case 0x56:
    case 0x59:
    case 0x62:
    case 0x65:
      result = 32;
      break;
    case 6:
    case 0xA:
    case 0x14:
    case 0x1B:
    case 0x23:
    case 0x2A:
    case 0x31:
    case 0x41:
    case 0x46:
    case 0x4B:
    case 0x50:
    case 0x57:
    case 0x5A:
    case 0x5E:
    case 0x63:
    case 0x66:
    case 0x6A:
    case 0x6E:
      result = 64;
      break;
    case 7:
    case 0xC:
    case 0xD:
    case 0x15:
    case 0x1C:
    case 0x24:
    case 0x2B:
    case 0x32:
    case 0x37:
    case 0x42:
    case 0x47:
    case 0x4C:
    case 0x51:
    case 0x58:
    case 0x5B:
    case 0x5F:
    case 0x64:
    case 0x67:
    case 0x6B:
      result = 128;
      break;
    case 0xB:
      result = 80;
      break;
    case 0xF:
    case 0x39:
      result = 2;
      break;
    case 0x10:
    case 0x3A:
      result = 4;
      break;
    case 0x16:
    case 0x1E:
    case 0x26:
    case 0x2D:
    case 0x34:
    case 0x49:
    case 0x4E:
    case 0x53:
    case 0x5D:
    case 0x61:
    case 0x69:
    case 0x6D:
      result = 512;
      break;
    case 0x17:
    case 0x1F:
    case 0x27:
    case 0x2E:
    case 0x35:
    case 0x4F:
    case 0x54:
      result = 1024;
      break;
    case 0x1D:
    case 0x25:
    case 0x2C:
    case 0x33:
    case 0x43:
    case 0x48:
    case 0x4D:
    case 0x52:
    case 0x5C:
    case 0x60:
    case 0x68:
    case 0x6C:
      result = 256;
      break;
    case 0x20:
    case 0x28:
    case 0x2F:
    case 0x36:
    case 0x55:
      result = 2048;
      break;
    case 0x30:
      result = 4096;
      break;
  }
  return result;
}
