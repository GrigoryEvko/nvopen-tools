// Function: sub_82E970
// Address: 0x82e970
//
char *__fastcall sub_82E970(int a1)
{
  char *result; // rax

  result = "integral";
  if ( a1 != 65 )
  {
    if ( a1 > 65 )
    {
      result = "scoped enum";
      if ( a1 != 512 )
      {
        if ( a1 <= 512 )
        {
          result = "integral or enum";
          if ( a1 != 193 )
          {
            if ( a1 <= 193 )
            {
              result = "enum";
              if ( a1 != 128 )
              {
                result = "built-in";
                if ( a1 == 129 )
                  return "integral or enum";
              }
            }
            else
            {
              result = "built-in";
              if ( a1 == 256 )
                return "unscoped enum";
            }
          }
        }
        else
        {
          result = "size_t";
          if ( a1 != 2048 )
          {
            if ( a1 == 0x4000 )
            {
              return "nullptr_t";
            }
            else
            {
              result = "built-in";
              if ( a1 == 1024 )
                return "ptrdiff_t";
            }
          }
        }
      }
    }
    else if ( a1 > 32 )
    {
      result = "built-in";
      if ( a1 == 64 )
        return "bool";
    }
    else
    {
      result = "built-in";
      if ( a1 > 0 )
      {
        switch ( a1 )
        {
          case 1:
            result = "non-bool integral";
            break;
          case 2:
            result = "floating";
            break;
          case 4:
            result = "pointer";
            break;
          case 8:
            result = "pointer-to-object";
            break;
          case 16:
            result = "pointer-to-function";
            break;
          case 32:
            result = "pointer-to-member";
            break;
          default:
            result = "built-in";
            break;
        }
      }
    }
  }
  return result;
}
