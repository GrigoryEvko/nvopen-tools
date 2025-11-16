// Function: sub_240D4F0
// Address: 0x240d4f0
//
__int64 __fastcall sub_240D4F0(unsigned int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 0u:
    case 7u:
      result = a1;
      break;
    case 1u:
    case 2u:
    case 5u:
      result = 5;
      break;
    case 4u:
    case 6u:
      result = 6;
      break;
    default:
      BUG();
  }
  return result;
}
