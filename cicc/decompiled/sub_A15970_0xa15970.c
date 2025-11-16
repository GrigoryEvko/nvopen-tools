// Function: sub_A15970
// Address: 0xa15970
//
__int64 __fastcall sub_A15970(unsigned int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 0u:
    case 1u:
    case 2u:
      result = a1;
      break;
    case 4u:
      result = 3;
      break;
    case 5u:
      result = 4;
      break;
    case 6u:
      result = 5;
      break;
    case 7u:
      result = 6;
      break;
    default:
      BUG();
  }
  return result;
}
