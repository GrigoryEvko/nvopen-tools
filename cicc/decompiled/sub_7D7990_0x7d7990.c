// Function: sub_7D7990
// Address: 0x7d7990
//
__int64 __fastcall sub_7D7990(char a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 10:
LABEL_18:
      result = qword_4F187E0;
      if ( !qword_4F187E0 )
      {
        result = sub_7D7530(0, "_Complex_float16");
        qword_4F187E0 = result;
      }
      break;
    case 11:
LABEL_16:
      result = qword_4F187D0;
      if ( !qword_4F187D0 )
      {
        result = sub_7D7530(2u, "_Complex_float");
        qword_4F187D0 = result;
      }
      break;
    case 12:
LABEL_6:
      result = qword_4F187C8;
      if ( !qword_4F187C8 )
      {
        result = sub_7D7530(4u, "_Complex_double");
        qword_4F187C8 = result;
      }
      break;
    case 13:
LABEL_14:
      result = qword_4F187B0;
      if ( !qword_4F187B0 )
      {
        result = sub_7D7530(8u, "_Complex_float128");
        qword_4F187B0 = result;
      }
      break;
    default:
      switch ( a1 )
      {
        case 0:
          goto LABEL_18;
        case 2:
          goto LABEL_16;
        case 3:
        case 4:
          goto LABEL_6;
        case 5:
        case 6:
          result = qword_4F187C0;
          if ( !qword_4F187C0 )
          {
            result = sub_7D7530(6u, "_Complex_long_double");
            qword_4F187C0 = result;
          }
          return result;
        case 7:
          result = qword_4F187B8;
          if ( !qword_4F187B8 )
          {
            result = sub_7D7530(7u, "_Complex_float80");
            qword_4F187B8 = result;
          }
          return result;
        case 8:
          goto LABEL_14;
        case 9:
          result = qword_4F187D8;
          if ( !qword_4F187D8 )
          {
            result = sub_7D7530(9u, "_Complex_bfloat16");
            qword_4F187D8 = result;
          }
          return result;
        default:
          sub_721090();
      }
  }
  return result;
}
