// Function: sub_B5A570
// Address: 0xb5a570
//
__int64 __fastcall sub_B5A570(unsigned int a1)
{
  __int64 v2; // [rsp+0h] [rbp-8h]

  if ( a1 != 438 )
  {
    if ( a1 > 0x1B6 )
    {
      if ( a1 != 470 && a1 != 481 )
        goto LABEL_6;
    }
    else if ( a1 != 168 )
    {
      if ( a1 != 433 && a1 != 167 )
      {
LABEL_6:
        BYTE4(v2) = 0;
        return v2;
      }
      goto LABEL_10;
    }
    LODWORD(v2) = 1;
    BYTE4(v2) = 1;
    return v2;
  }
LABEL_10:
  LODWORD(v2) = 0;
  BYTE4(v2) = 1;
  return v2;
}
