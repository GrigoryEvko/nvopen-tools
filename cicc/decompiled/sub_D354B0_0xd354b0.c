// Function: sub_D354B0
// Address: 0xd354b0
//
__int64 __fastcall sub_D354B0(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 0:
    case 3:
    case 6:
      result = 0;
      break;
    case 1:
      result = 1;
      break;
    case 2:
    case 4:
    case 5:
    case 7:
      result = 2;
      break;
    default:
      BUG();
  }
  return result;
}
