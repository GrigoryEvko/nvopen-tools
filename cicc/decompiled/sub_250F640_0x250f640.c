// Function: sub_250F640
// Address: 0x250f640
//
__int64 __fastcall sub_250F640(__int64 a1, char a2)
{
  __int64 result; // rax

  switch ( a2 )
  {
    case 0:
      result = sub_904010(a1, "inv");
      break;
    case 1:
      result = sub_904010(a1, "flt");
      break;
    case 2:
      result = sub_904010(a1, "fn_ret");
      break;
    case 3:
      result = sub_904010(a1, "cs_ret");
      break;
    case 4:
      result = sub_904010(a1, "fn");
      break;
    case 5:
      result = sub_904010(a1, "cs");
      break;
    case 6:
      result = sub_904010(a1, "arg");
      break;
    case 7:
      result = sub_904010(a1, "cs_arg");
      break;
    default:
      BUG();
  }
  return result;
}
