// Function: sub_B534E0
// Address: 0xb534e0
//
__int64 __fastcall sub_B534E0(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax

  switch ( a3 )
  {
    case ' ':
      result = sub_C76FF0(a1, a2);
      break;
    case '!':
      result = sub_C771C0();
      break;
    case '"':
      result = sub_C771E0();
      break;
    case '#':
      result = sub_C77420(a1, a2);
      break;
    case '$':
      result = sub_C77450();
      break;
    case '%':
      result = sub_C77460();
      break;
    case '&':
      result = sub_C77470(a1, a2);
      break;
    case '\'':
      result = sub_C77860();
      break;
    case '(':
      result = sub_C77890();
      break;
    case ')':
      result = sub_C778A0();
      break;
    default:
      BUG();
  }
  return result;
}
