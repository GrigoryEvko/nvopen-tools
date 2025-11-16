// Function: sub_E0CAC0
// Address: 0xe0cac0
//
char *__fastcall sub_E0CAC0(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "NONE";
      break;
    case 1:
      result = "TYPE";
      break;
    case 2:
      result = "VARIABLE";
      break;
    case 3:
      result = "FUNCTION";
      break;
    case 4:
      result = "OTHER";
      break;
    case 5:
      result = "UNUSED5";
      break;
    case 6:
      result = "UNUSED6";
      break;
    case 7:
      result = "UNUSED7";
      break;
    default:
      BUG();
  }
  return result;
}
