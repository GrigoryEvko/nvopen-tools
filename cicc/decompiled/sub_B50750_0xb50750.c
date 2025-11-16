// Function: sub_B50750
// Address: 0xb50750
//
bool __fastcall sub_B50750(int a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __int64 v6; // rax
  int v7; // r12d
  __int64 v8; // rax
  int v9; // ebx

  switch ( a1 )
  {
    case '&':
    case '\'':
    case '(':
    case ')':
    case '*':
    case '+':
    case ',':
    case '-':
    case '.':
    case '2':
      result = 0;
      break;
    case '/':
      v8 = sub_AE4450(a4, a2);
      v9 = sub_BCB060(v8);
      result = v9 == (unsigned int)sub_BCB060(a3);
      break;
    case '0':
      v6 = sub_AE4450(a4, a3);
      v7 = sub_BCB060(v6);
      result = v7 == (unsigned int)sub_BCB060(a2);
      break;
    case '1':
      result = 1;
      break;
    default:
      BUG();
  }
  return result;
}
