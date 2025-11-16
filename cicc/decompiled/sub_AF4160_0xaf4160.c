// Function: sub_AF4160
// Address: 0xaf4160
//
__int64 __fastcall sub_AF4160(unsigned __int64 **a1)
{
  unsigned __int64 v1; // rdx
  __int64 result; // rax
  _BOOL4 v3; // eax

  v1 = **a1;
  result = 2;
  if ( v1 - 112 > 0x1F )
  {
    if ( v1 <= 0x95 )
    {
      if ( v1 <= 0x92 && v1 != 144 )
      {
        if ( v1 <= 0x90 )
        {
          if ( v1 <= 0x11 )
            v3 = v1 > 0xF;
          else
            v3 = v1 == 35;
          return (unsigned int)(v3 + 1);
        }
        else
        {
          return 2 * (unsigned int)(v1 == 146) + 1;
        }
      }
    }
    else if ( v1 != 4101 )
    {
      if ( v1 > 0x1005 )
      {
        return v1 - 4102 < 2 ? 3 : 1;
      }
      else if ( v1 > 0x1001 )
      {
        return (unsigned int)(v1 - 4098 < 2) + 1;
      }
      else
      {
        return v1 < 0x1000 ? 1 : 3;
      }
    }
  }
  return result;
}
