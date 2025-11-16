// Function: sub_8094C0
// Address: 0x8094c0
//
char *__fastcall sub_8094C0(char a1, int a2)
{
  char *result; // rax

  switch ( a1 )
  {
    case 1:
      result = "nw";
      break;
    case 2:
      result = "dl";
      break;
    case 3:
      result = "na";
      break;
    case 4:
      result = "da";
      break;
    case 5:
      result = "ps";
      if ( a2 != 1 )
        result = "pl";
      break;
    case 6:
      result = "ng";
      if ( a2 != 1 )
        result = "mi";
      break;
    case 7:
      result = "de";
      if ( a2 != 1 )
        result = "ml";
      break;
    case 8:
      result = (char *)&unk_3C1B694;
      break;
    case 9:
      result = "rm";
      break;
    case 10:
      result = (char *)&unk_3C1B697;
      break;
    case 11:
      result = "ad";
      if ( a2 != 1 )
        result = "an";
      break;
    case 12:
      result = "or";
      break;
    case 13:
      result = "co";
      break;
    case 14:
      result = "nt";
      break;
    case 15:
      result = (char *)&unk_3C1B69A;
      break;
    case 16:
      result = (char *)&unk_432C6B5;
      break;
    case 17:
      result = (char *)&unk_432C6B9;
      break;
    case 18:
      result = (char *)&unk_3C1B69D;
      break;
    case 19:
      result = (char *)&unk_3C1B6A0;
      break;
    case 20:
      result = (char *)&unk_3C1B6A3;
      break;
    case 21:
      result = (char *)&unk_3C1B6A6;
      break;
    case 22:
      result = (char *)&unk_3C1B6A9;
      break;
    case 23:
      result = (char *)&unk_3C1B6AC;
      break;
    case 24:
      result = "aN";
      break;
    case 25:
      result = (char *)&unk_3C1B6AF;
      break;
    case 26:
      result = (char *)&unk_432C6C5;
      break;
    case 27:
      result = "rs";
      break;
    case 28:
      result = (char *)&unk_3C1B6B2;
      break;
    case 29:
      result = (char *)&unk_3C1B6B5;
      break;
    case 30:
      result = "eq";
      break;
    case 31:
      result = (char *)&unk_432C6B1;
      break;
    case 32:
      result = "le";
      break;
    case 33:
      result = "ge";
      break;
    case 34:
      result = "ss";
      break;
    case 35:
      result = "aa";
      break;
    case 36:
      result = "oo";
      break;
    case 37:
      result = "pp";
      break;
    case 38:
      result = "mm";
      break;
    case 39:
      result = "cm";
      break;
    case 40:
      result = "pm";
      break;
    case 41:
      result = "pt";
      break;
    case 42:
      result = "cl";
      break;
    case 43:
      result = "ix";
      break;
    case 44:
      result = "qu";
      break;
    case 45:
      result = "v23min";
      break;
    case 46:
      result = "v23max";
      break;
    case 47:
      result = "aw";
      break;
    default:
      sub_721090();
  }
  return result;
}
