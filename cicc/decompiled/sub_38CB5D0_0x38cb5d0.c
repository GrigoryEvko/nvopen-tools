// Function: sub_38CB5D0
// Address: 0x38cb5d0
//
char *__fastcall sub_38CB5D0(__int16 a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "<<none>>";
      break;
    case 1:
      result = "<<invalid>>";
      break;
    case 2:
      result = "GOT";
      break;
    case 3:
      result = "GOTOFF";
      break;
    case 4:
      result = "GOTREL";
      break;
    case 5:
      result = "GOTPCREL";
      break;
    case 6:
      result = "GOTTPOFF";
      break;
    case 7:
      result = "INDNTPOFF";
      break;
    case 8:
      result = "NTPOFF";
      break;
    case 9:
      result = "GOTNTPOFF";
      break;
    case 10:
      result = "PLT";
      break;
    case 11:
      result = "TLSGD";
      break;
    case 12:
      result = "TLSLD";
      break;
    case 13:
      result = "TLSLDM";
      break;
    case 14:
      result = "TPOFF";
      break;
    case 15:
      result = "DTPOFF";
      break;
    case 16:
      result = "tlscall";
      break;
    case 17:
      result = "tlsdesc";
      break;
    case 18:
      result = "TLVP";
      break;
    case 19:
      result = "TLVPPAGE";
      break;
    case 20:
      result = "TLVPPAGEOFF";
      break;
    case 21:
      result = "PAGE";
      break;
    case 22:
      result = "PAGEOFF";
      break;
    case 23:
      result = "GOTPAGE";
      break;
    case 24:
      result = "GOTPAGEOFF";
      break;
    case 25:
      result = "SECREL32";
      break;
    case 26:
      result = "SIZE";
      break;
    case 27:
      result = "WEAKREF";
      break;
    case 28:
      result = "ABS8";
      break;
    case 29:
    case 37:
      result = "none";
      break;
    case 30:
      result = "GOT_PREL";
      break;
    case 31:
      result = "target1";
      break;
    case 32:
      result = "target2";
      break;
    case 33:
      result = "prel31";
      break;
    case 34:
      result = "sbrel";
      break;
    case 35:
      result = "tlsldo";
      break;
    case 36:
      result = "tlsdescseq";
      break;
    case 38:
      result = "lo8";
      break;
    case 39:
      result = "hi8";
      break;
    case 40:
      result = "hlo8";
      break;
    case 41:
      result = "diff8";
      break;
    case 42:
      result = "diff16";
      break;
    case 43:
      result = "diff32";
      break;
    case 44:
      result = "l";
      break;
    case 45:
      result = "h";
      break;
    case 46:
      result = "ha";
      break;
    case 47:
      result = "high";
      break;
    case 48:
      result = "higha";
      break;
    case 49:
      result = "higher";
      break;
    case 50:
      result = "highera";
      break;
    case 51:
      result = "highest";
      break;
    case 52:
      result = "highesta";
      break;
    case 53:
      result = "got@l";
      break;
    case 54:
      result = "got@h";
      break;
    case 55:
      result = "got@ha";
      break;
    case 56:
      result = "tocbase";
      break;
    case 57:
      result = "toc";
      break;
    case 58:
      result = "toc@l";
      break;
    case 59:
      result = "toc@h";
      break;
    case 60:
      result = "toc@ha";
      break;
    case 61:
      result = "dtpmod";
      break;
    case 62:
      result = "tprel@l";
      break;
    case 63:
      result = "tprel@h";
      break;
    case 64:
      result = "tprel@ha";
      break;
    case 65:
      result = "tprel@high";
      break;
    case 66:
      result = "tprel@higha";
      break;
    case 67:
      result = "tprel@higher";
      break;
    case 68:
      result = "tprel@highera";
      break;
    case 69:
      result = "tprel@highest";
      break;
    case 70:
      result = "tprel@highesta";
      break;
    case 71:
      result = "dtprel@l";
      break;
    case 72:
      result = "dtprel@h";
      break;
    case 73:
      result = "dtprel@ha";
      break;
    case 74:
      result = "dtprel@high";
      break;
    case 75:
      result = "dtprel@higha";
      break;
    case 76:
      result = "dtprel@higher";
      break;
    case 77:
      result = "dtprel@highera";
      break;
    case 78:
      result = "dtprel@highest";
      break;
    case 79:
      result = "dtprel@highesta";
      break;
    case 80:
      result = "got@tprel";
      break;
    case 81:
      result = "got@tprel@l";
      break;
    case 82:
      result = "got@tprel@h";
      break;
    case 83:
      result = "got@tprel@ha";
      break;
    case 84:
      result = "got@dtprel";
      break;
    case 85:
      result = "got@dtprel@l";
      break;
    case 86:
      result = "got@dtprel@h";
      break;
    case 87:
      result = "got@dtprel@ha";
      break;
    case 88:
      result = "tls";
      break;
    case 89:
      result = "got@tlsgd";
      break;
    case 90:
      result = "got@tlsgd@l";
      break;
    case 91:
      result = "got@tlsgd@h";
      break;
    case 92:
      result = "got@tlsgd@ha";
      break;
    case 93:
      result = "tlsgd";
      break;
    case 94:
      result = "got@tlsld";
      break;
    case 95:
      result = "got@tlsld@l";
      break;
    case 96:
      result = "got@tlsld@h";
      break;
    case 97:
      result = "got@tlsld@ha";
      break;
    case 98:
      result = "tlsld";
      break;
    case 99:
      result = "local";
      break;
    case 100:
      result = "IMGREL";
      break;
    case 101:
      result = "PCREL";
      break;
    case 102:
      result = "LO16";
      break;
    case 103:
      result = "HI16";
      break;
    case 104:
      result = "GPREL";
      break;
    case 105:
      result = "GDGOT";
      break;
    case 106:
      result = "LDGOT";
      break;
    case 107:
      result = "GDPLT";
      break;
    case 108:
      result = "LDPLT";
      break;
    case 109:
      result = "IE";
      break;
    case 110:
      result = "IEGOT";
      break;
    case 111:
      result = "FUNCTION";
      break;
    case 112:
      result = "TYPEINDEX";
      break;
    case 113:
      result = "gotpcrel32@lo";
      break;
    case 114:
      result = "gotpcrel32@hi";
      break;
    case 115:
      result = "rel32@lo";
      break;
    case 116:
      result = "rel32@hi";
      break;
    case 117:
      result = "rel64";
      break;
    case 118:
      result = "TPREL";
      break;
    case 119:
      result = "DTPREL";
      break;
  }
  return result;
}
