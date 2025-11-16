// Function: sub_37361D0
// Address: 0x37361d0
//
__int64 __fastcall sub_37361D0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  if ( !sub_3736140(a1) )
    return a2;
  switch ( (__int16)a2 )
  {
    case 122:
      result = 8471;
      break;
    case 125:
      result = 17;
      break;
    case 126:
      result = 8465;
      break;
    case 127:
      result = 49;
      break;
    case 130:
      result = 8469;
      break;
    case 131:
      result = 8467;
      break;
    default:
      BUG();
  }
  return result;
}
