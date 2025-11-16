// Function: sub_8274A0
// Address: 0x8274a0
//
__int64 __fastcall sub_8274A0(char a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 'A':
    case 'a':
      result = 67;
      break;
    case 'B':
      result = 64;
      break;
    case 'C':
      result = 0;
      break;
    case 'D':
      result = 1024;
      break;
    case 'E':
      result = 128;
      break;
    case 'F':
      result = 16;
      break;
    case 'I':
    case 'i':
      result = 65;
      break;
    case 'M':
      result = 32;
      break;
    case 'N':
      result = 0x4000;
      break;
    case 'O':
      result = 8;
      break;
    case 'P':
      result = 4;
      break;
    case 'S':
      result = 512;
      break;
    case 'b':
      result = 231;
      break;
    case 'n':
      result = 3;
      break;
    default:
      sub_721090();
  }
  return result;
}
