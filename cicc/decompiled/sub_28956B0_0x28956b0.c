// Function: sub_28956B0
// Address: 0x28956b0
//
__int64 __fastcall sub_28956B0(__int64 a1)
{
  __int64 v1; // rbx
  int v2; // edx
  unsigned int v3; // ecx
  unsigned __int8 v4; // al
  __int64 *v5; // rax
  __int64 result; // rax

  if ( *(_BYTE *)a1 <= 0x1Cu )
    return 32 * (unsigned int)(unsigned __int8)byte_5004008;
  switch ( *(_BYTE *)a1 )
  {
    case ')':
    case '+':
    case '-':
    case '/':
    case '2':
    case '5':
    case 'J':
    case 'K':
    case 'S':
      goto LABEL_20;
    case '*':
    case ',':
    case '.':
    case '0':
    case '1':
    case '3':
    case '4':
    case '6':
    case '7':
    case '8':
    case '9':
    case ':':
    case ';':
    case '<':
    case '=':
    case '>':
    case '?':
    case '@':
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
    case 'Q':
    case 'R':
      return 32 * (unsigned int)(unsigned __int8)byte_5004008;
    case 'T':
    case 'U':
    case 'V':
      v1 = *(_QWORD *)(a1 + 8);
      v2 = *(unsigned __int8 *)(v1 + 8);
      v3 = v2 - 17;
      v4 = *(_BYTE *)(v1 + 8);
      if ( (unsigned int)(v2 - 17) <= 1 )
        v4 = *(_BYTE *)(**(_QWORD **)(v1 + 16) + 8LL);
      if ( v4 <= 3u || v4 == 5 || (v4 & 0xFD) == 4 )
        goto LABEL_20;
      if ( (_BYTE)v2 == 15 )
      {
        if ( (*(_BYTE *)(v1 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(a1 + 8)) )
          return 32 * (unsigned int)(unsigned __int8)byte_5004008;
        v5 = *(__int64 **)(v1 + 16);
        v1 = *v5;
        v2 = *(unsigned __int8 *)(*v5 + 8);
        v3 = v2 - 17;
      }
      else if ( (_BYTE)v2 == 16 )
      {
        do
        {
          v1 = *(_QWORD *)(v1 + 24);
          LOBYTE(v2) = *(_BYTE *)(v1 + 8);
        }
        while ( (_BYTE)v2 == 16 );
        v3 = (unsigned __int8)v2 - 17;
      }
      if ( v3 <= 1 )
        LOBYTE(v2) = *(_BYTE *)(**(_QWORD **)(v1 + 16) + 8LL);
      if ( (unsigned __int8)v2 > 3u && (_BYTE)v2 != 5 && (v2 & 0xFD) != 4 )
        return 32 * (unsigned int)(unsigned __int8)byte_5004008;
LABEL_20:
      result = sub_B45210(a1);
      if ( byte_5004008 )
        result = (unsigned int)result | 0x20;
      break;
    default:
      return 32 * (unsigned int)(unsigned __int8)byte_5004008;
  }
  return result;
}
