// Function: sub_2484E50
// Address: 0x2484e50
//
char __fastcall sub_2484E50(__int64 a1, __int64 a2)
{
  char result; // al
  unsigned int v3; // [rsp+Ch] [rbp-4h] BYREF

  result = sub_981210(a2, a1, &v3);
  if ( result )
  {
    switch ( v3 )
    {
      case '*':
      case ',':
      case '.':
      case '0':
      case '6':
      case '8':
      case ':':
      case '<':
      case '>':
      case '@':
        return result;
      case '+':
      case '-':
      case '/':
      case '1':
      case '7':
      case '9':
      case ';':
      case '=':
      case '?':
      case 'A':
        result = byte_4FE94A8;
        break;
      default:
        result = 0;
        break;
    }
  }
  return result;
}
