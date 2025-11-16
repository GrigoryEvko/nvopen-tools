// Function: sub_3288400
// Address: 0x3288400
//
__int64 __fastcall sub_3288400(__int64 a1, __int64 a2)
{
  __int16 *v2; // rax
  __int16 v3; // bx
  __int64 v4; // rax
  __int16 v6; // [rsp+0h] [rbp-30h] BYREF
  __int64 v7; // [rsp+8h] [rbp-28h]

  v2 = *(__int16 **)(a1 + 48);
  v3 = *v2;
  v4 = *((_QWORD *)v2 + 1);
  v6 = v3;
  v7 = v4;
  if ( v3 )
  {
    if ( (unsigned __int16)(v3 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
  }
  else
  {
    if ( sub_3007100((__int64)&v6) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    sub_3007130((__int64)&v6, a2);
  }
  return *(_QWORD *)(a1 + 96);
}
