// Function: sub_2FE4C50
// Address: 0x2fe4c50
//
__int64 __fastcall sub_2FE4C50(__int64 a1, int a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  switch ( a2 )
  {
    case 1:
      v3 = 2;
      result = 2;
      break;
    case 2:
      v3 = 3;
      result = 3;
      break;
    case 4:
      v3 = 4;
      result = 4;
      break;
    case 8:
      v3 = 5;
      result = 5;
      break;
    case 16:
      v3 = 6;
      result = 6;
      break;
    case 32:
      v3 = 7;
      result = 7;
      break;
    case 64:
      v3 = 8;
      result = 8;
      break;
    case 128:
      v3 = 9;
      result = 9;
      break;
    default:
      return 0;
  }
  if ( !*(_QWORD *)(a1 + 8 * v3 + 112) )
    return 0;
  return result;
}
