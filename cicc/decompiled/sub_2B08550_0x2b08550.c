// Function: sub_2B08550
// Address: 0x2b08550
//
bool __fastcall sub_2B08550(unsigned __int8 **a1, __int64 a2)
{
  unsigned __int8 **v2; // rcx
  unsigned __int8 *v3; // rsi
  unsigned __int8 *v4; // rdx

  v2 = &a1[a2];
  if ( v2 == a1 )
    return 0;
  v3 = 0;
  do
  {
    while ( 1 )
    {
      v4 = *a1;
      if ( (unsigned int)**a1 - 12 > 1 )
        break;
LABEL_3:
      if ( v2 == ++a1 )
        return v3 != 0;
    }
    if ( v3 )
    {
      if ( v4 != v3 )
        return 0;
      goto LABEL_3;
    }
    ++a1;
    v3 = v4;
  }
  while ( v2 != a1 );
  return v3 != 0;
}
