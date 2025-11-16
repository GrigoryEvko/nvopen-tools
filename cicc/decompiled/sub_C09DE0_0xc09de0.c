// Function: sub_C09DE0
// Address: 0xc09de0
//
__int64 __fastcall sub_C09DE0(__int64 a1)
{
  __int64 v2; // [rsp+14h] [rbp-1Ch]

  if ( sub_BCAC40(a1, 32) || *(_BYTE *)(a1 + 8) == 2 )
  {
    LODWORD(v2) = 4;
    BYTE4(v2) = 1;
    return v2;
  }
  if ( !sub_BCAC40(a1, 16) && *(_BYTE *)(a1 + 8) > 1u )
  {
    if ( sub_BCAC40(a1, 8) )
    {
      LODWORD(v2) = 16;
      BYTE4(v2) = 1;
    }
    return v2;
  }
  LODWORD(v2) = 8;
  BYTE4(v2) = 1;
  return v2;
}
