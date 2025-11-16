// Function: sub_2FE5770
// Address: 0x2fe5770
//
__int64 __fastcall sub_2FE5770(__int16 a1, __int64 a2, __int16 a3)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 11:
      result = 346;
      if ( a3 != 12 )
      {
        result = 345;
        if ( a3 != 13 )
        {
          result = 343;
          if ( a3 != 14 )
          {
            result = 729;
            if ( a3 == 15 )
              return 342;
          }
        }
      }
      break;
    case 12:
      result = 344;
      if ( a3 != 13 )
      {
        result = 341;
        if ( a3 != 15 )
        {
          result = 729;
          if ( a3 == 16 )
            return 337;
        }
      }
      break;
    case 13:
      result = 340;
      if ( a3 != 15 )
      {
        result = 729;
        if ( a3 == 16 )
          return 338;
      }
      break;
    case 14:
      result = 729;
      if ( a3 == 15 )
        return 339;
      break;
    default:
      if ( a3 == 12 && a1 == 10 )
        return 336;
      else
        return 729;
  }
  return result;
}
