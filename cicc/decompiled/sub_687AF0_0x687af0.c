// Function: sub_687AF0
// Address: 0x687af0
//
__int16 __fastcall sub_687AF0(__int64 a1, _DWORD *a2, _DWORD *a3, int *a4)
{
  char v5; // al
  int v6; // edx
  int v7; // eax
  __int64 v8; // rax
  bool v9; // cf
  char v10; // al
  char v11; // al
  char v12; // al
  __int64 v13; // rax
  char v14; // al
  char v15; // al
  __int64 v16; // rax

  *a2 = 0;
  *a3 = 0;
  v5 = *(_BYTE *)(a1 + 24);
  switch ( v5 )
  {
    case 1:
      switch ( *(_BYTE *)(a1 + 56) )
      {
        case 0:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 33;
          break;
        case 3:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 34;
          break;
        case 5:
        case 6:
        case 7:
        case 0xA:
        case 0xC:
        case 0xE:
        case 0xF:
        case 0x10:
        case 0x11:
        case 0x14:
          *a2 = 1;
          if ( (*(_BYTE *)(a1 + 25) & 0x40) != 0 )
          {
            v6 = 1;
            LOWORD(v7) = 177;
          }
          else
          {
            v10 = *(_BYTE *)(a1 + 58);
            v6 = 1;
            if ( (v10 & 8) != 0 )
              LOWORD(v7) = 166;
            else
              LOWORD(v7) = (v10 & 2) == 0 ? 183 : 176;
          }
          break;
        case 0x12:
        case 0x13:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 167;
          break;
        case 0x16:
        case 0x64:
          v6 = 1;
          LOWORD(v7) = 29;
          break;
        case 0x17:
        case 0x5F:
        case 0x65:
          v6 = 1;
          LOWORD(v7) = 30;
          break;
        case 0x18:
LABEL_40:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 243;
          break;
        case 0x1A:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 36;
          break;
        case 0x1B:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 35;
          break;
        case 0x1C:
        case 0x20:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 37;
          break;
        case 0x1D:
        case 0x1E:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 38;
          break;
        case 0x21:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 144;
          break;
        case 0x22:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 145;
          break;
        case 0x23:
          *a2 = 1;
          LOWORD(v7) = 31;
          *a3 = 1;
          v6 = 1;
          break;
        case 0x24:
          *a2 = 1;
          LOWORD(v7) = 32;
          *a3 = 1;
          v6 = 1;
          break;
        case 0x25:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 31;
          break;
        case 0x26:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 32;
          break;
        case 0x27:
        case 0x2E:
        case 0x2F:
        case 0x32:
          v6 = 1;
          LOWORD(v7) = 35;
          break;
        case 0x28:
        case 0x30:
        case 0x31:
        case 0x33:
        case 0x34:
          v6 = 1;
          LOWORD(v7) = 36;
          break;
        case 0x29:
        case 0x2C:
          v6 = 1;
          LOWORD(v7) = 34;
          break;
        case 0x2A:
        case 0x2D:
          v6 = 1;
          LOWORD(v7) = 39;
          break;
        case 0x2B:
          v6 = 1;
          LOWORD(v7) = 40;
          break;
        case 0x35:
          v6 = 1;
          LOWORD(v7) = 41;
          break;
        case 0x36:
          v6 = 1;
          LOWORD(v7) = 42;
          break;
        case 0x37:
          v6 = 1;
          LOWORD(v7) = 33;
          break;
        case 0x38:
          v6 = 1;
          LOWORD(v7) = 51;
          break;
        case 0x39:
          v6 = 1;
          LOWORD(v7) = 50;
          break;
        case 0x3A:
        case 0x41:
          v6 = 1;
          LOWORD(v7) = 47;
          break;
        case 0x3B:
        case 0x42:
          v6 = 1;
          LOWORD(v7) = 48;
          break;
        case 0x3C:
        case 0x43:
          v6 = 1;
          LOWORD(v7) = 44;
          break;
        case 0x3D:
        case 0x44:
          v6 = 1;
          LOWORD(v7) = 43;
          break;
        case 0x3E:
        case 0x45:
          v6 = 1;
          LOWORD(v7) = 46;
          break;
        case 0x3F:
        case 0x46:
          v6 = 1;
          LOWORD(v7) = 45;
          break;
        case 0x40:
          v6 = 1;
          LOWORD(v7) = 49;
          break;
        case 0x47:
          v6 = 1;
          LOWORD(v7) = 70;
          break;
        case 0x48:
          v6 = 1;
          LOWORD(v7) = 71;
          break;
        case 0x49:
          v6 = 1;
          LOWORD(v7) = 56;
          break;
        case 0x4A:
        case 0x54:
          v6 = 1;
          LOWORD(v7) = 60;
          break;
        case 0x4B:
        case 0x55:
          v6 = 1;
          LOWORD(v7) = 61;
          break;
        case 0x4C:
          v6 = 1;
          LOWORD(v7) = 57;
          break;
        case 0x4D:
          v6 = 1;
          LOWORD(v7) = 58;
          break;
        case 0x4E:
          v6 = 1;
          LOWORD(v7) = 59;
          break;
        case 0x4F:
          v6 = 1;
          LOWORD(v7) = 62;
          break;
        case 0x50:
          v6 = 1;
          LOWORD(v7) = 63;
          break;
        case 0x51:
          v6 = 1;
          LOWORD(v7) = 64;
          break;
        case 0x52:
          v6 = 1;
          LOWORD(v7) = 66;
          break;
        case 0x53:
          v6 = 1;
          LOWORD(v7) = 65;
          break;
        case 0x57:
        case 0x59:
          v6 = 1;
          LOWORD(v7) = 52;
          break;
        case 0x58:
        case 0x5A:
          v6 = 1;
          LOWORD(v7) = 53;
          break;
        case 0x5B:
          v6 = 1;
          LOWORD(v7) = 67;
          break;
        case 0x5C:
        case 0x6E:
          v6 = 1;
          LOWORD(v7) = 25;
          break;
        case 0x5E:
          if ( (*(_BYTE *)(a1 + 27) & 2) != 0 )
          {
            v13 = *(_QWORD *)(a1 + 72);
            v6 = 1;
            if ( *(_BYTE *)(v13 + 24) == 3 )
              v7 = (*(_BYTE *)(*(_QWORD *)(v13 + 56) + 172LL) & 2) == 0 ? 29 : 1;
            else
              LOWORD(v7) = 29;
          }
          else
          {
            v6 = 1;
            LOWORD(v7) = 29;
          }
          break;
        case 0x60:
        case 0x62:
          v6 = 1;
          LOWORD(v7) = 147;
          break;
        case 0x61:
        case 0x63:
          v6 = 1;
          LOWORD(v7) = 148;
          break;
        case 0x67:
        case 0x68:
          v6 = 1;
          LOWORD(v7) = 54;
          break;
        case 0x69:
        case 0x6A:
        case 0x6B:
        case 0x6C:
        case 0x6D:
          v6 = 1;
          LOWORD(v7) = 27;
          break;
        default:
          v6 = 0;
          LOWORD(v7) = 0;
          break;
      }
      break;
    case 2:
      v8 = *(_QWORD *)(a1 + 56);
      if ( *(_BYTE *)(v8 + 173) == 12 )
      {
        switch ( *(_BYTE *)(v8 + 176) )
        {
          case 2:
          case 3:
          case 4:
          case 0xB:
          case 0xD:
            v6 = 1;
            LOWORD(v7) = 1;
            break;
          case 5:
            *a2 = 1;
            v6 = 1;
            LOWORD(v7) = 99;
            break;
          case 6:
            *a2 = 1;
            v6 = 1;
            LOWORD(v7) = 284;
            break;
          case 7:
            v12 = *(_BYTE *)(v8 + 200);
            v6 = 1;
            *a2 = 1;
            v7 = -((v12 & 1) == 0);
            LOBYTE(v7) = v7 & 0x78;
            LOWORD(v7) = v7 + 247;
            break;
          case 9:
            *a2 = 1;
            v6 = 1;
            LOWORD(v7) = 178;
            break;
          case 0xA:
            goto LABEL_40;
          default:
            v6 = 0;
            LOWORD(v7) = 0;
            break;
        }
      }
      else
      {
        if ( !*(_QWORD *)(a1 + 80) || (unsigned int)sub_8DBE70(*(_QWORD *)(v8 + 128)) )
          goto LABEL_21;
        v6 = 1;
        LOWORD(v7) = 4;
      }
      break;
    case 12:
      goto LABEL_28;
    case 15:
      *a2 = 1;
      v6 = 1;
      LOWORD(v7) = 284;
      break;
    case 13:
LABEL_28:
      *a2 = 1;
      v6 = 1;
      LOWORD(v7) = 99;
      break;
    case 14:
      v9 = *(_BYTE *)(a1 + 57) == 0;
      v6 = 1;
      *a2 = 1;
      v7 = -v9;
      LOBYTE(v7) = v9 ? 0x78 : 0;
      LOWORD(v7) = v7 + 247;
      break;
    case 11:
      *a2 = 1;
      v6 = 1;
      LOWORD(v7) = 178;
      break;
    case 5:
      v14 = *(_BYTE *)(*(_QWORD *)(a1 + 56) + 50LL);
      if ( (v14 & 0x20) != 0 )
      {
        *a2 = 1;
        v6 = 1;
        LOWORD(v7) = 183;
      }
      else
      {
        if ( (v14 & 0x10) == 0 )
          goto LABEL_21;
        v15 = *(_BYTE *)(a1 + 25);
        v6 = 1;
        *a2 = 1;
        LOWORD(v7) = (v15 & 0x40) == 0 ? 183 : 177;
      }
      break;
    case 7:
      v6 = 1;
      v11 = **(_BYTE **)(a1 + 56);
      *a2 = 1;
      LOWORD(v7) = (v11 & 1) == 0 ? 152 : 155;
      break;
    case 8:
      *a2 = 1;
      v6 = 1;
      LOWORD(v7) = 162;
      break;
    case 23:
      switch ( *(_BYTE *)(a1 + 56) )
      {
        case 0:
          v6 = 1;
          LOWORD(v7) = 117;
          break;
        case 0xD:
          v6 = 1;
          LOWORD(v7) = 207;
          break;
        case 0xF:
          v6 = 1;
          LOWORD(v7) = 209;
          break;
        case 0x15:
          sub_721090(a1);
        case 0x16:
          v6 = 1;
          LOWORD(v7) = 112;
          break;
        case 0x1E:
          v6 = 1;
          LOWORD(v7) = 227;
          break;
        case 0x1F:
          v6 = 1;
          LOWORD(v7) = 228;
          break;
        case 0x2D:
          v6 = 1;
          LOWORD(v7) = 233;
          break;
        case 0x2E:
          v6 = 1;
          LOWORD(v7) = 234;
          break;
        case 0x2F:
          v6 = 1;
          LOWORD(v7) = 256;
          break;
        case 0x30:
          v6 = 1;
          LOWORD(v7) = 251;
          break;
        case 0x31:
          v6 = 1;
          LOWORD(v7) = 252;
          break;
        case 0x32:
          v6 = 1;
          LOWORD(v7) = 253;
          break;
        case 0x33:
          v6 = 1;
          LOWORD(v7) = 254;
          break;
        case 0x34:
          v6 = 1;
          LOWORD(v7) = 255;
          break;
        case 0x35:
          v6 = 1;
          LOWORD(v7) = 257;
          break;
        case 0x36:
          v6 = 1;
          LOWORD(v7) = 261;
          break;
        case 0x3A:
          v6 = 1;
          LOWORD(v7) = 258;
          break;
        case 0x3B:
          v6 = 1;
          LOWORD(v7) = 259;
          break;
        case 0x3C:
          v6 = 1;
          LOWORD(v7) = 270;
          break;
        case 0x42:
          v6 = 1;
          LOWORD(v7) = 288;
          break;
        case 0x43:
          v6 = 1;
          LOWORD(v7) = 291;
          break;
        case 0x44:
          v6 = 1;
          LOWORD(v7) = 292;
          break;
        case 0x46:
          v6 = 1;
          LOWORD(v7) = 296;
          break;
        case 0x47:
          v6 = 1;
          LOWORD(v7) = 297;
          break;
        case 0x48:
          v6 = 1;
          LOWORD(v7) = 298;
          break;
        case 0x49:
          v6 = 1;
          LOWORD(v7) = 299;
          break;
        case 0x4A:
          v6 = 1;
          LOWORD(v7) = 300;
          break;
        case 0x4B:
          v6 = 1;
          LOWORD(v7) = 301;
          break;
        case 0x4C:
          v6 = 1;
          LOWORD(v7) = 302;
          break;
        case 0x4D:
          v6 = 1;
          LOWORD(v7) = 303;
          break;
        case 0x4E:
          v6 = 1;
          LOWORD(v7) = 304;
          break;
        case 0x6C:
          v6 = 1;
          LOWORD(v7) = 211;
          break;
        case 0x6D:
          v6 = 1;
          LOWORD(v7) = 289;
          break;
        case 0x6E:
          v6 = 1;
          LOWORD(v7) = 290;
          break;
        case 0x6F:
          v6 = 1;
          LOWORD(v7) = 210;
          break;
        case 0x70:
          v6 = 1;
          LOWORD(v7) = 225;
          break;
        case 0x71:
          v6 = 1;
          LOWORD(v7) = 226;
          break;
        default:
          *a2 = 1;
          v6 = 1;
          LOWORD(v7) = 195;
          break;
      }
      break;
    case 24:
      v6 = 1;
      LOWORD(v7) = 1;
      break;
    case 25:
      v6 = 1;
      LOWORD(v7) = 73;
      break;
    case 3:
      v16 = *(_QWORD *)(a1 + 56);
      if ( (*(_BYTE *)(v16 + 170) & 0x10) != 0 )
      {
        v6 = 1;
        LOWORD(v7) = 1;
      }
      else
      {
        if ( (*(_BYTE *)(v16 + 89) & 1) == 0 )
          goto LABEL_21;
        v6 = 1;
        LOWORD(v7) = 1;
        if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) == 14 )
          goto LABEL_21;
      }
      break;
    case 20:
      v6 = 1;
      LOWORD(v7) = 1;
      if ( !*(_QWORD *)(*(_QWORD *)(a1 + 56) + 248LL) )
      {
LABEL_21:
        v6 = 0;
        LOWORD(v7) = 0;
      }
      break;
    case 30:
      v6 = 1;
      LOWORD(v7) = 76;
      break;
    default:
      goto LABEL_21;
  }
  if ( a4 )
    *a4 = v6;
  return v7;
}
