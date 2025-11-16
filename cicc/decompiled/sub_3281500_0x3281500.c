// Function: sub_3281500
// Address: 0x3281500
//
__int64 __fastcall sub_3281500(_WORD *a1, __int64 a2)
{
  if ( *a1 )
  {
    if ( (unsigned __int16)(*a1 - 176) > 0x34u )
      return word_4456340[(unsigned __int16)*a1 - 1];
  }
  else if ( !sub_3007100((__int64)a1) )
  {
    return sub_3007130((__int64)a1, a2);
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( *a1 )
  {
    if ( (unsigned __int16)(*a1 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    return word_4456340[(unsigned __int16)*a1 - 1];
  }
  return sub_3007130((__int64)a1, a2);
}
