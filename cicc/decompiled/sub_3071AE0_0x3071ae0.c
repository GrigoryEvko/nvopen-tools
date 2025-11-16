// Function: sub_3071AE0
// Address: 0x3071ae0
//
__int64 __fastcall sub_3071AE0(unsigned int a1)
{
  char v2; // cl
  bool v3; // dl
  __int64 v4; // [rsp+8h] [rbp-8h]

  if ( a1 == 8170 || a1 - 8927 <= 5 )
  {
    LODWORD(v4) = 0;
    BYTE4(v4) = 1;
    return v4;
  }
  v2 = a1 + 11;
  if ( a1 - 8181 > 0x23 )
    goto LABEL_7;
  if ( ((1LL << v2) & 0xCFC00040FLL) == 0 )
  {
    v3 = 1;
    if ( ((1LL << v2) & 0xCFC0C000FLL) != 0 )
    {
LABEL_9:
      LODWORD(v4) = 0;
      BYTE4(v4) = v3;
      return v4;
    }
LABEL_7:
    v3 = 0;
    if ( byte_502CE28 )
      v3 = sub_3071900(a1);
    goto LABEL_9;
  }
  LODWORD(v4) = 1;
  BYTE4(v4) = 1;
  return v4;
}
