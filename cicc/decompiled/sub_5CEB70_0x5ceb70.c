// Function: sub_5CEB70
// Address: 0x5ceb70
//
__int64 __fastcall sub_5CEB70(__int64 a1, char a2)
{
  __int64 result; // rax

  switch ( a2 )
  {
    case 2:
    case 6:
    case 7:
    case 8:
    case 11:
    case 12:
    case 28:
      result = a1 + 104;
      break;
    case 3:
      result = a1 + 64;
      break;
    case 21:
    case 29:
    case 37:
      result = a1 + 32;
      break;
    case 86:
      result = a1 + 24;
      break;
    default:
      sub_721090(a1);
  }
  return result;
}
