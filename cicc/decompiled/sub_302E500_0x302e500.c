// Function: sub_302E500
// Address: 0x302e500
//
__int64 __fastcall sub_302E500(__int16 a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 18:
    case 20:
    case 35:
    case 37:
    case 38:
    case 39:
    case 40:
    case 47:
    case 49:
    case 50:
    case 51:
    case 58:
    case 60:
    case 64:
    case 116:
    case 118:
    case 127:
    case 129:
    case 130:
    case 131:
    case 138:
    case 140:
    case 141:
    case 142:
    case 147:
    case 149:
    case 153:
    case 167:
    case 169:
      result = 1;
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
