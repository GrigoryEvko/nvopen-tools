// Function: sub_2461F90
// Address: 0x2461f90
//
__int64 __fastcall sub_2461F90(unsigned int a1)
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
