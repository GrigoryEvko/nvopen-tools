// Function: sub_827390
// Address: 0x827390
//
char *__fastcall sub_827390(char a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 'A':
    case 'a':
      result = "arithmetic";
      break;
    case 'B':
      result = "bool";
      break;
    case 'C':
      result = "class";
      break;
    case 'D':
    case 'I':
    case 'i':
      result = "integer";
      break;
    case 'E':
      result = "enum";
      break;
    case 'F':
      result = "pointer-to-function";
      break;
    case 'H':
      result = "handle";
      break;
    case 'M':
      result = "pointer-to-member";
      break;
    case 'N':
      result = "nullptr type";
      break;
    case 'O':
      result = "pointer-to-object";
      break;
    case 'P':
      result = "pointer";
      break;
    case 'S':
      result = "scoped enum";
      break;
    case 'b':
      result = "bool-equivalent";
      break;
    case 'h':
      result = "handle-to-CLI-array";
      break;
    case 'n':
      result = "non-bool arithmetic";
      break;
    default:
      sub_721090();
  }
  return result;
}
