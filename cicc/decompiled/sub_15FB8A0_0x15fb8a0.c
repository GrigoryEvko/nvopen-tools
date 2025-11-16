// Function: sub_15FB8A0
// Address: 0x15fb8a0
//
bool __fastcall sub_15FB8A0(int a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __int64 v6; // rax
  int v7; // r12d
  __int64 v8; // rax
  int v9; // ebx

  switch ( a1 )
  {
    case '$':
    case '%':
    case '&':
    case '\'':
    case '(':
    case ')':
    case '*':
    case '+':
    case ',':
    case '0':
      result = 0;
      break;
    case '-':
      v8 = sub_15A9650(a4, a2);
      v9 = sub_16431D0(v8);
      result = v9 == (unsigned int)sub_16431D0(a3);
      break;
    case '.':
      v6 = sub_15A9650(a4, a3);
      v7 = sub_16431D0(v6);
      result = v7 == (unsigned int)sub_16431D0(a2);
      break;
    case '/':
      result = 1;
      break;
  }
  return result;
}
