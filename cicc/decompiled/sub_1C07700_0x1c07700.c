// Function: sub_1C07700
// Address: 0x1c07700
//
__int64 __fastcall sub_1C07700(unsigned int a1)
{
  __int64 result; // rax

  if ( a1 == 4345 )
    return 2;
  if ( a1 > 0x10F9 )
    return 4 * (unsigned int)(a1 == 4346);
  if ( a1 > 0xE82 )
    return a1 == 4344;
  result = 0;
  if ( a1 > 0xE54 )
  {
    switch ( a1 )
    {
      case 0xE55u:
      case 0xE56u:
      case 0xE57u:
      case 0xE58u:
      case 0xE5Bu:
      case 0xE5Cu:
      case 0xE5Du:
      case 0xE5Eu:
      case 0xE5Fu:
      case 0xE60u:
      case 0xE67u:
      case 0xE68u:
      case 0xE69u:
      case 0xE6Au:
      case 0xE71u:
      case 0xE72u:
      case 0xE73u:
      case 0xE74u:
      case 0xE75u:
      case 0xE76u:
      case 0xE79u:
      case 0xE7Au:
      case 0xE7Bu:
      case 0xE7Cu:
      case 0xE7Du:
      case 0xE7Eu:
      case 0xE7Fu:
      case 0xE80u:
      case 0xE81u:
      case 0xE82u:
        result = 7;
        break;
      default:
        result = 0;
        break;
    }
  }
  return result;
}
