// Function: sub_1F7DF20
// Address: 0x1f7df20
//
__int64 __fastcall sub_1F7DF20(_BYTE *a1)
{
  __int64 result; // rax
  char v2; // di
  unsigned int v3; // eax
  char v4; // cl
  char v5; // di
  char v6; // r8
  int v7; // esi

  if ( !*a1 )
    return sub_1F5A910((__int64)a1);
  switch ( *a1 )
  {
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
    case 0x38:
    case 0x39:
    case 0x3A:
    case 0x3B:
    case 0x3C:
    case 0x3D:
      v2 = 2;
      goto LABEL_5;
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x1D:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
      v2 = 3;
      goto LABEL_5;
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x48:
    case 0x49:
      v2 = 4;
      goto LABEL_5;
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x4A:
    case 0x4B:
    case 0x4C:
    case 0x4D:
    case 0x4E:
    case 0x4F:
      v2 = 5;
      goto LABEL_5;
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x35:
    case 0x36:
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x53:
    case 0x54:
    case 0x55:
      v2 = 6;
      goto LABEL_5;
    case 0x37:
      v3 = sub_1F6C8D0(7);
      if ( v3 != 32 )
        goto LABEL_6;
      v6 = 5;
      v7 = 1;
      goto LABEL_12;
    case 0x56:
    case 0x57:
    case 0x58:
    case 0x62:
    case 0x63:
    case 0x64:
      v2 = 8;
      goto LABEL_5;
    case 0x59:
    case 0x5A:
    case 0x5B:
    case 0x5C:
    case 0x5D:
    case 0x65:
    case 0x66:
    case 0x67:
    case 0x68:
    case 0x69:
      v2 = 9;
      goto LABEL_5;
    case 0x5E:
    case 0x5F:
    case 0x60:
    case 0x61:
    case 0x6A:
    case 0x6B:
    case 0x6C:
    case 0x6D:
      v2 = 10;
LABEL_5:
      v3 = sub_1F6C8D0(v2);
      if ( v3 == 32 )
      {
        v5 = 5;
      }
      else
      {
LABEL_6:
        if ( v3 > 0x20 )
        {
          v5 = 6;
          if ( v3 != 64 )
          {
            v5 = 0;
            if ( v3 == 128 )
              v5 = 7;
          }
        }
        else
        {
          v5 = 3;
          if ( v3 != 8 )
          {
            v5 = 4;
            if ( v3 != 16 )
              v5 = 2 * (v3 == 1);
          }
        }
      }
      v6 = v5;
      v7 = word_42FA680[(unsigned __int8)(v4 - 14)];
      if ( (unsigned __int8)(v4 - 56) <= 0x1Du || (unsigned __int8)(v4 - 98) <= 0xBu )
        result = (unsigned __int8)sub_1D154A0(v5, v7);
      else
LABEL_12:
        result = (unsigned __int8)sub_1D15020(v6, v7);
      break;
  }
  return result;
}
