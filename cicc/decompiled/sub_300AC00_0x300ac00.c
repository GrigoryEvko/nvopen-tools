// Function: sub_300AC00
// Address: 0x300ac00
//
void *__fastcall sub_300AC00(unsigned __int16 *a1)
{
  int v1; // eax
  void *result; // rax

  v1 = *a1;
  if ( (unsigned __int16)(v1 - 17) <= 0xD3u )
    LOWORD(v1) = word_4456580[v1 - 1];
  switch ( (__int16)v1 )
  {
    case 10:
      result = sub_C33300();
      break;
    case 11:
      result = sub_C332F0();
      break;
    case 12:
      result = sub_C33310();
      break;
    case 13:
      result = sub_C33320();
      break;
    case 14:
      result = sub_C33420();
      break;
    case 15:
      result = sub_C33330();
      break;
    case 16:
      result = sub_C33340();
      break;
    default:
      BUG();
  }
  return result;
}
