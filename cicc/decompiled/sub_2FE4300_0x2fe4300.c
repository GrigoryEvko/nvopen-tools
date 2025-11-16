// Function: sub_2FE4300
// Address: 0x2fe4300
//
__int64 __fastcall sub_2FE4300(__int64 a1, unsigned int a2)
{
  char (__fastcall *v2)(__int64, unsigned int); // rax
  __int64 result; // rax

  v2 = *(char (__fastcall **)(__int64, unsigned int))(*(_QWORD *)a1 + 1360LL);
  if ( v2 == sub_2FE3400 )
  {
    if ( a2 <= 0x62 )
    {
      if ( a2 > 0x37 )
      {
        switch ( a2 )
        {
          case '8':
          case ':':
          case '?':
          case '@':
          case 'D':
          case 'F':
          case 'L':
          case 'M':
          case 'R':
          case 'S':
          case '`':
          case 'b':
            return 1;
          default:
            goto LABEL_4;
        }
      }
      goto LABEL_4;
    }
    if ( a2 > 0xBC )
    {
      if ( a2 - 279 <= 7 )
        return 1;
    }
    else
    {
      if ( a2 > 0xB9 || a2 - 172 <= 0xB )
        return 1;
      if ( a2 <= 0x64 )
        goto LABEL_5;
    }
    return a2 - 190 <= 4;
  }
  result = ((__int64 (*)(void))v2)();
  if ( (_BYTE)result )
    return result;
  if ( a2 > 0x64 )
    return a2 - 190 <= 4;
LABEL_4:
  result = 0;
  if ( a2 > 0x38 )
  {
LABEL_5:
    switch ( a2 )
    {
      case '9':
      case ';':
      case '<':
      case '=':
      case '>':
      case 'T':
      case 'U':
      case 'a':
      case 'c':
      case 'd':
        return 1;
      default:
        return 0;
    }
  }
  return result;
}
