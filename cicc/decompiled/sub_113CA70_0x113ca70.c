// Function: sub_113CA70
// Address: 0x113ca70
//
__int64 __fastcall sub_113CA70(const __m128i *a1, __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  __int64 result; // rax

  switch ( *a3 )
  {
    case '*':
      result = (__int64)sub_1128290((__int64)a1, a2, a3, a4);
      if ( !result )
        goto LABEL_4;
      break;
    case ',':
      result = (__int64)sub_1119FB0((__int64)a1, a2, (__int64)a3, a4);
      if ( !result )
        goto LABEL_4;
      break;
    case '.':
      result = (__int64)sub_1115510((__int64)a1, a2, (__int64)a3, a4);
      if ( !result )
        goto LABEL_4;
      break;
    case '0':
      result = (__int64)sub_11164F0((__int64)a1, a2, (__int64)a3, a4);
      if ( !result )
        goto LABEL_8;
      break;
    case '1':
LABEL_8:
      result = sub_1122A30((__int64)a1, a2, (__int64)a3, a4);
      if ( !result )
        goto LABEL_4;
      break;
    case '4':
      result = (__int64)sub_1115C10((__int64)a1, a2, (__int64)a3, a4);
      if ( !result )
        goto LABEL_4;
      break;
    case '6':
      result = (__int64)sub_1120680((__int64)a1, a2, a3, a4);
      if ( !result )
        goto LABEL_4;
      break;
    case '7':
    case '8':
      result = (__int64)sub_1126B10((__int64)a1, a2, (__int64)a3, a4);
      if ( !result )
        goto LABEL_4;
      break;
    case '9':
      result = (__int64)sub_112C930(a1, a2, (__int64)a3, a4);
      if ( !result )
      {
        result = (__int64)sub_112E9E0((__int64)a1, a2, (char *)a3, a4);
        if ( !result )
          goto LABEL_4;
      }
      break;
    case ':':
      result = (__int64)sub_1133500((__int64)a1, a2, (__int64)a3, a4);
      if ( !result )
        goto LABEL_4;
      break;
    case ';':
      result = (__int64)sub_111CED0((__int64)a1, a2, (__int64)a3, a4);
      if ( !result )
      {
        result = (__int64)sub_113BFE0((__int64)a1, a2, (__int64)a3, a4);
        if ( !result )
          goto LABEL_4;
      }
      break;
    default:
LABEL_4:
      result = sub_1119340(a1, a2, a3, a4);
      break;
  }
  return result;
}
