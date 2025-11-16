// Function: sub_98AB90
// Address: 0x98ab90
//
char __fastcall sub_98AB90(__int64 a1, char a2)
{
  unsigned int v2; // eax
  unsigned int v3; // edx
  int v4; // eax

  v2 = sub_B49240();
  if ( v2 == 353 )
  {
    return (unsigned int)sub_B2D610(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 72LL), 49) ^ 1;
  }
  else
  {
    v3 = v2;
    if ( v2 > 0x161 )
    {
      LOBYTE(v4) = 1;
      if ( v3 != 8170 )
      {
        if ( v3 <= 0x1FEA )
        {
          if ( v3 != 2070 && v3 != 3012 )
            LOBYTE(v4) = v3 == 558;
        }
        else if ( v3 != 8923 )
        {
          LOBYTE(v4) = v3 == 9250;
        }
      }
    }
    else
    {
      LOBYTE(v4) = a2 ^ 1;
      if ( v3 != 299 )
      {
        LOBYTE(v4) = 1;
        if ( v3 != 346 )
          LOBYTE(v4) = v3 == 208;
      }
    }
  }
  return v4;
}
