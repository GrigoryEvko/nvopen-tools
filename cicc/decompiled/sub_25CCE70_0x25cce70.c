// Function: sub_25CCE70
// Address: 0x25cce70
//
char *__fastcall sub_25CCE70(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "None";
      break;
    case 1:
      result = "GlobalVar";
      break;
    case 2:
      result = "NotLive";
      break;
    case 3:
      result = "TooLarge";
      break;
    case 4:
      result = "InterposableLinkage";
      break;
    case 5:
      result = "LocalLinkageNotInModule";
      break;
    case 6:
      result = "NotEligible";
      break;
    case 7:
      result = "NoInline";
      break;
    default:
      BUG();
  }
  return result;
}
