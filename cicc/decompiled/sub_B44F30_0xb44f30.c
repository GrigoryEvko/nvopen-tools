// Function: sub_B44F30
// Address: 0xb44f30
//
__int64 __fastcall sub_B44F30(unsigned __int8 *a1)
{
  int v1; // edx
  __int64 result; // rax
  __int64 v3; // rbx
  int v4; // edx
  unsigned int v5; // ecx
  int v6; // eax

  v1 = *a1;
  result = (unsigned int)(v1 - 42);
  switch ( *a1 )
  {
    case '*':
    case ',':
    case '.':
    case '6':
      a1[1] &= 0xF9u;
      if ( (unsigned __int8)v1 > 0x1Cu )
        goto LABEL_4;
      return result;
    case '0':
    case '1':
    case '7':
    case '8':
      a1[1] &= ~2u;
      goto LABEL_3;
    case ':':
    case 'R':
      a1[1] &= ~2u;
      return result;
    case '?':
      result = sub_B4DDE0(a1, 0);
      v1 = *a1;
      if ( (unsigned __int8)v1 <= 0x1Cu )
        return result;
      goto LABEL_4;
    case 'C':
      a1[1] &= 0xF9u;
      return result;
    case 'D':
    case 'H':
      result = sub_B448D0((__int64)a1, 0);
      v1 = *a1;
      if ( (unsigned __int8)v1 <= 0x1Cu )
        return result;
      goto LABEL_4;
    default:
LABEL_3:
      if ( (unsigned __int8)v1 > 0x1Cu )
      {
LABEL_4:
        switch ( v1 )
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
            goto LABEL_10;
          case 'T':
          case 'U':
          case 'V':
            v3 = *((_QWORD *)a1 + 1);
            v4 = *(unsigned __int8 *)(v3 + 8);
            v5 = v4 - 17;
            v6 = v4;
            if ( (unsigned int)(v4 - 17) <= 1 )
              v6 = *(unsigned __int8 *)(**(_QWORD **)(v3 + 16) + 8LL);
            if ( (unsigned __int8)v6 <= 3u )
              goto LABEL_10;
            if ( (_BYTE)v6 == 5 )
              goto LABEL_10;
            result = v6 & 0xFFFFFFFD;
            if ( (_BYTE)result == 4 )
              goto LABEL_10;
            if ( (_BYTE)v4 == 15 )
            {
              if ( (*(_BYTE *)(v3 + 9) & 4) == 0 )
                return result;
              result = sub_BCB420(*((_QWORD *)a1 + 1));
              if ( !(_BYTE)result )
                return result;
              result = *(_QWORD *)(v3 + 16);
              v3 = *(_QWORD *)result;
              v4 = *(unsigned __int8 *)(*(_QWORD *)result + 8LL);
              v5 = v4 - 17;
            }
            else if ( (_BYTE)v4 == 16 )
            {
              do
              {
                v3 = *(_QWORD *)(v3 + 24);
                LOBYTE(v4) = *(_BYTE *)(v3 + 8);
              }
              while ( (_BYTE)v4 == 16 );
              v5 = (unsigned __int8)v4 - 17;
            }
            if ( v5 <= 1 )
            {
              result = **(_QWORD **)(v3 + 16);
              LOBYTE(v4) = *(_BYTE *)(result + 8);
            }
            if ( (unsigned __int8)v4 <= 3u || (_BYTE)v4 == 5 || (v4 & 0xFD) == 4 )
            {
LABEL_10:
              sub_B44EF0((__int64)a1, 0);
              result = sub_B44F10((__int64)a1, 0);
            }
            break;
          default:
            return result;
        }
      }
      return result;
  }
}
