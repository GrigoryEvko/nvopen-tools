// Function: sub_21C5950
// Address: 0x21c5950
//
__int64 __fastcall sub_21C5950(char a1, unsigned __int8 a2, char a3)
{
  int v3; // eax
  int v5; // eax
  int v6; // eax

  if ( a2 == 5 )
  {
    v6 = a3 == 0 ? 0x30 : 0;
    if ( a1 == 4 )
    {
      return (unsigned int)(v6 + 244);
    }
    else if ( a1 == 6 )
    {
      return (unsigned int)(v6 + 266);
    }
    else
    {
      return (unsigned int)(v6 + 277);
    }
  }
  else if ( a2 > 5u )
  {
    v5 = a3 == 0 ? 0x30 : 0;
    if ( a1 == 4 )
    {
      return (unsigned int)(v5 + 245);
    }
    else if ( a1 == 5 )
    {
      return (unsigned int)(v5 + 256);
    }
    else
    {
      return (unsigned int)(v5 + 278);
    }
  }
  else
  {
    v3 = a3 == 0 ? 0x30 : 0;
    if ( a2 == 3 )
    {
      if ( a1 == 5 )
      {
        return (unsigned int)(v3 + 257);
      }
      else if ( a1 == 6 )
      {
        return (unsigned int)(v3 + 268);
      }
      else
      {
        return (unsigned int)(v3 + 246);
      }
    }
    else if ( a1 == 5 )
    {
      return (unsigned int)(v3 + 254);
    }
    else if ( a1 == 6 )
    {
      return (unsigned int)(v3 + 265);
    }
    else
    {
      return (unsigned int)(v3 + 276);
    }
  }
}
