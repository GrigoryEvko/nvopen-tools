// Function: sub_1D16F90
// Address: 0x1d16f90
//
__int64 __fastcall sub_1D16F90(unsigned int a1, unsigned int a2, char a3)
{
  __int64 result; // rax
  unsigned int v4; // edi

  result = a2 & a1;
  if ( !a3 )
    return result;
  if ( a1 != 17 )
  {
    if ( a1 <= 0x11 )
    {
      if ( a2 >= 0x12 && a2 <= 0x15 )
        return 24;
    }
    else if ( a1 <= 0x15 && a2 != 17 )
    {
      result = 24;
      if ( a2 <= 0x11 )
        return result;
      if ( a2 <= 0x15 )
        return a2 & a1;
    }
  }
  v4 = a2 & a1;
  switch ( v4 )
  {
    case 1u:
    case 9u:
      result = 17;
      break;
    case 2u:
      result = 10;
      break;
    case 4u:
      result = 12;
      break;
    case 8u:
      result = 0;
      break;
    default:
      return v4;
  }
  return result;
}
