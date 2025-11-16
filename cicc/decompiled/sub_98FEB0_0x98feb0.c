// Function: sub_98FEB0
// Address: 0x98feb0
//
__int64 __fastcall sub_98FEB0(int a1, int a2)
{
  __int64 result; // rax
  __int64 v3; // [rsp+0h] [rbp-Ch]
  __int64 v4; // [rsp+0h] [rbp-Ch]

  switch ( a1 )
  {
    case 2:
    case 3:
    case 10:
    case 11:
      LODWORD(v3) = 6;
      HIDWORD(v3) = a2;
      result = v3;
      break;
    case 4:
    case 5:
    case 12:
    case 13:
      LODWORD(v4) = 5;
      HIDWORD(v4) = a2;
      result = v4;
      break;
    case 34:
    case 35:
      result = 4;
      break;
    case 36:
    case 37:
      result = 2;
      break;
    case 38:
    case 39:
      result = 3;
      break;
    case 40:
    case 41:
      result = 1;
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
