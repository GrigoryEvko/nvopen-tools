// Function: sub_385F800
// Address: 0x385f800
//
__int64 __fastcall sub_385F800(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 0:
    case 2:
    case 5:
      result = 1;
      break;
    case 1:
    case 3:
    case 4:
    case 6:
      result = 0;
      break;
  }
  return result;
}
