// Function: sub_3034570
// Address: 0x3034570
//
__int64 __fastcall sub_3034570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  int v7; // ebx
  unsigned __int16 v9; // bx
  unsigned int v11; // eax
  __int64 v12; // [rsp+0h] [rbp-40h] BYREF
  __int64 v13; // [rsp+8h] [rbp-38h]
  __int64 v14; // [rsp+10h] [rbp-30h] BYREF
  __int64 v15; // [rsp+18h] [rbp-28h]

  v7 = (unsigned __int16)a4;
  v14 = a2;
  v15 = a3;
  v12 = a4;
  v13 = a5;
  if ( (_WORD)a4 == 9 )
    return 0;
  if ( (_WORD)a4 )
  {
    if ( (unsigned __int16)(a4 - 17) > 0xD3u )
      return sub_2FEBBF0(a1, (unsigned int)v14, v15, (unsigned int)v12, v13, a6, a7);
    if ( (unsigned __int16)(a4 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    if ( word_4456340[v7 - 1] <= 4u )
      return sub_2FEBBF0(a1, (unsigned int)v14, v15, (unsigned int)v12, v13, a6, a7);
  }
  else
  {
    if ( !sub_30070B0((__int64)&v12) )
      return sub_2FEBBF0(a1, (unsigned int)v14, v15, (unsigned int)v12, v13, a6, a7);
    if ( sub_3007100((__int64)&v12) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    if ( (unsigned int)sub_3007130((__int64)&v12, a2) <= 4 )
      return sub_2FEBBF0(a1, (unsigned int)v14, v15, (unsigned int)v12, v13, a6, a7);
  }
  v9 = v14;
  if ( (_WORD)v14 )
  {
    if ( (unsigned __int16)(v14 - 17) > 0xD3u )
      return 0;
    if ( (unsigned __int16)(v14 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v11 = word_4456340[v9 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v14) )
      return 0;
    if ( sub_3007100((__int64)&v14) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v11 = sub_3007130((__int64)&v14, a2);
  }
  if ( v11 <= 4 )
    return 0;
  return sub_2FEBBF0(a1, (unsigned int)v14, v15, (unsigned int)v12, v13, a6, a7);
}
