// Function: sub_15A46C0
// Address: 0x15a46c0
//
__int64 __fastcall sub_15A46C0(int a1, __int64 ***a2, __int64 **a3, char a4)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case '$':
      result = sub_15A43B0((unsigned __int64)a2, a3, a4);
      break;
    case '%':
      result = sub_15A3CB0((unsigned __int64)a2, a3, a4);
      break;
    case '&':
      result = sub_15A4460((unsigned __int64)a2, a3, a4);
      break;
    case '\'':
      result = sub_15A4020((unsigned __int64)a2, a3, a4);
      break;
    case '(':
      result = sub_15A40D0((unsigned __int64)a2, a3, a4);
      break;
    case ')':
      result = sub_15A3EC0((unsigned __int64)a2, a3, a4);
      break;
    case '*':
      result = sub_15A3F70((unsigned __int64)a2, a3, a4);
      break;
    case '+':
      result = sub_15A3D60((unsigned __int64)a2, a3, a4);
      break;
    case ',':
      result = sub_15A3E10((unsigned __int64)a2, a3, a4);
      break;
    case '-':
      result = sub_15A4180((unsigned __int64)a2, a3, a4);
      break;
    case '.':
      result = sub_15A3BA0((unsigned __int64)a2, a3, a4);
      break;
    case '/':
      result = sub_15A4510(a2, a3, a4);
      break;
    case '0':
      result = sub_15A3300((unsigned __int64)a2, (__int64)a3, a4);
      break;
  }
  return result;
}
