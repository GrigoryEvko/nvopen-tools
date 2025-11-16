// Function: sub_DDAF10
// Address: 0xddaf10
//
char __fastcall sub_DDAF10(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v10; // al
  __int64 v11; // r9
  char result; // al
  char v13; // al
  __int64 v14; // rsi
  char v15; // al
  char v16; // al
  char v17; // al

  switch ( (int)a2 )
  {
    case ' ':
    case '!':
      v16 = sub_D90F00(a3, a5);
      v11 = a6;
      if ( v16 )
      {
        result = sub_D90F00(a4, a6);
        v11 = a6;
        if ( result )
          return result;
      }
      return sub_DDA790(a1, a2, a3, a4, a5, v11, 0);
    case '"':
    case '#':
      v15 = sub_DCD020(a1, 35, a3, a5);
      v11 = a6;
      if ( !v15 )
        return sub_DDA790(a1, a2, a3, a4, a5, v11, 0);
      v14 = 37;
      break;
    case '$':
    case '%':
      v13 = sub_DCD020(a1, 37, a3, a5);
      v11 = a6;
      if ( !v13 )
        return sub_DDA790(a1, a2, a3, a4, a5, v11, 0);
      v14 = 35;
      break;
    case '&':
    case '\'':
      v10 = sub_DCD020(a1, 39, a3, a5);
      v11 = a6;
      if ( !v10 )
        return sub_DDA790(a1, a2, a3, a4, a5, v11, 0);
      v14 = 41;
      break;
    case '(':
    case ')':
      v17 = sub_DCD020(a1, 41, a3, a5);
      v11 = a6;
      if ( !v17 )
        return sub_DDA790(a1, a2, a3, a4, a5, v11, 0);
      v14 = 39;
      break;
    default:
      BUG();
  }
  result = sub_DCD020(a1, v14, a4, v11);
  v11 = a6;
  if ( !result )
    return sub_DDA790(a1, a2, a3, a4, a5, v11, 0);
  return result;
}
