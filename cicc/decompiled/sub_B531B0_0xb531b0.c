// Function: sub_B531B0
// Address: 0xb531b0
//
__int64 __fastcall sub_B531B0(unsigned int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 2u:
      result = 3;
      break;
    case 4u:
      result = 5;
      break;
    case 0xAu:
      result = 11;
      break;
    case 0xCu:
      result = 13;
      break;
    case 0x22u:
      result = 35;
      break;
    case 0x24u:
      result = 37;
      break;
    case 0x26u:
      result = 39;
      break;
    case 0x28u:
      result = 41;
      break;
    default:
      result = a1;
      break;
  }
  return result;
}
