// Function: sub_2FE5990
// Address: 0x2fe5990
//
__int64 __fastcall sub_2FE5990(__int16 a1, __int64 a2, __int16 a3)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 11:
      result = 364;
      if ( a3 != 7 )
      {
        result = 365;
        if ( a3 != 8 )
        {
          result = 729;
          if ( a3 == 9 )
            return 366;
        }
      }
      break;
    case 12:
      result = 367;
      if ( a3 != 7 )
      {
        result = 368;
        if ( a3 != 8 )
        {
          result = 729;
          if ( a3 == 9 )
            return 369;
        }
      }
      break;
    case 13:
      result = 370;
      if ( a3 != 7 )
      {
        result = 371;
        if ( a3 != 8 )
        {
          result = 729;
          if ( a3 == 9 )
            return 372;
        }
      }
      break;
    case 14:
      result = 373;
      if ( a3 != 7 )
      {
        result = 374;
        if ( a3 != 8 )
        {
          result = 729;
          if ( a3 == 9 )
            return 375;
        }
      }
      break;
    case 15:
      result = 376;
      if ( a3 != 7 )
      {
        result = 377;
        if ( a3 != 8 )
        {
          result = 729;
          if ( a3 == 9 )
            return 378;
        }
      }
      break;
    default:
      result = 729;
      if ( a1 == 16 )
      {
        result = 379;
        if ( a3 != 7 )
        {
          result = 380;
          if ( a3 != 8 )
          {
            result = 729;
            if ( a3 == 9 )
              return 381;
          }
        }
      }
      break;
  }
  return result;
}
