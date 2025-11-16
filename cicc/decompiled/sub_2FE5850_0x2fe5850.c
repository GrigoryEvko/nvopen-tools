// Function: sub_2FE5850
// Address: 0x2fe5850
//
__int64 __fastcall sub_2FE5850(__int16 a1, __int64 a2, __int16 a3)
{
  __int64 result; // rax

  switch ( a3 )
  {
    case 11:
      result = 347;
      if ( a1 != 12 )
      {
        result = 348;
        if ( a1 != 13 )
        {
          result = 349;
          if ( a1 != 14 )
          {
            result = 350;
            if ( a1 != 15 )
            {
              result = 729;
              if ( a1 == 16 )
                return 351;
            }
          }
        }
      }
      break;
    case 10:
      result = 352;
      if ( a1 != 12 )
      {
        result = 353;
        if ( a1 != 13 )
        {
          result = 354;
          if ( a1 != 14 )
          {
            result = 729;
            if ( a1 == 15 )
              return 355;
          }
        }
      }
      break;
    case 12:
      result = 356;
      if ( a1 != 13 )
      {
        result = 357;
        if ( a1 != 14 )
        {
          result = 358;
          if ( a1 != 15 )
          {
            result = 729;
            if ( a1 == 16 )
              return 359;
          }
        }
      }
      break;
    case 13:
      result = 360;
      if ( a1 != 14 )
      {
        result = 361;
        if ( a1 != 15 )
        {
          result = 729;
          if ( a1 == 16 )
            return 362;
        }
      }
      break;
    default:
      if ( a1 == 15 && a3 == 14 )
        return 363;
      else
        return 729;
  }
  return result;
}
