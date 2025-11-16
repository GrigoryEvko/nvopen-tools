// Function: sub_E22B50
// Address: 0xe22b50
//
__int64 __fastcall sub_E22B50(__int64 a1, __int64 *a2)
{
  char *v2; // rdx
  __int64 v3; // r10
  char v4; // al
  _BYTE *v5; // r9
  __int64 v6; // r8
  __int64 result; // rax
  char v8; // cl
  int v9; // eax

  v2 = (char *)a2[1];
  v3 = *a2;
  v4 = *v2;
  v5 = v2 + 1;
  v6 = *a2 - 1;
  a2[1] = (__int64)(v2 + 1);
  *a2 = v6;
  switch ( v4 )
  {
    case '$':
      if ( !v6 )
        goto LABEL_2;
      v8 = v2[1];
      v9 = 512;
      if ( v8 == 82 )
      {
        v5 = v2 + 2;
        v6 = v3 - 2;
        a2[1] = (__int64)(v2 + 2);
        *a2 = v3 - 2;
        if ( v3 == 2 )
          goto LABEL_2;
        v8 = v2[2];
        v9 = 1536;
      }
      a2[1] = (__int64)(v5 + 1);
      *a2 = v6 - 1;
      switch ( v8 )
      {
        case '0':
          result = v9 | 0x24u;
          break;
        case '1':
          result = v9 | 0x64u;
          break;
        case '2':
          result = v9 | 0x22u;
          break;
        case '3':
          result = v9 | 0x62u;
          break;
        case '4':
          result = v9 | 0x21u;
          break;
        case '5':
          result = v9 | 0x61u;
          break;
        default:
          goto LABEL_2;
      }
      break;
    case '9':
      result = 384;
      break;
    case 'A':
      result = 4;
      break;
    case 'B':
      result = 68;
      break;
    case 'C':
      result = 20;
      break;
    case 'D':
      result = 84;
      break;
    case 'E':
      result = 36;
      break;
    case 'F':
      result = 100;
      break;
    case 'G':
      result = 2052;
      break;
    case 'H':
      result = 2116;
      break;
    case 'I':
      result = 2;
      break;
    case 'J':
      result = 66;
      break;
    case 'K':
      result = 18;
      break;
    case 'L':
      result = 82;
      break;
    case 'M':
      result = 34;
      break;
    case 'N':
      result = 98;
      break;
    case 'O':
      result = 2082;
      break;
    case 'P':
      result = 2146;
      break;
    case 'Q':
      goto LABEL_3;
    case 'R':
      result = 65;
      break;
    case 'S':
      result = 17;
      break;
    case 'T':
      result = 81;
      break;
    case 'U':
      result = 33;
      break;
    case 'V':
      result = 97;
      break;
    case 'W':
      result = 2081;
      break;
    case 'X':
      result = 2145;
      break;
    case 'Y':
      result = 8;
      break;
    case 'Z':
      result = 72;
      break;
    default:
LABEL_2:
      *(_BYTE *)(a1 + 8) = 1;
LABEL_3:
      result = 1;
      break;
  }
  return result;
}
