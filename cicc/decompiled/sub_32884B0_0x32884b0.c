// Function: sub_32884B0
// Address: 0x32884b0
//
__int64 __fastcall sub_32884B0(__int64 a1, __int64 a2)
{
  unsigned __int16 *v2; // rdx
  unsigned __int16 v3; // ax
  __int64 v4; // rdx
  __int64 v5; // rsi
  unsigned __int16 v7; // [rsp+0h] [rbp-20h] BYREF
  __int64 v8; // [rsp+8h] [rbp-18h]

  v2 = *(unsigned __int16 **)(a1 + 48);
  v3 = *v2;
  v4 = *((_QWORD *)v2 + 1);
  v7 = v3;
  v8 = v4;
  if ( !v3 )
  {
    if ( !sub_3007100((__int64)&v7) )
      goto LABEL_5;
    goto LABEL_7;
  }
  if ( (unsigned __int16)(v3 - 176) <= 0x34u )
  {
LABEL_7:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( v7 )
    {
      if ( (unsigned __int16)(v7 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_3;
    }
LABEL_5:
    v5 = (unsigned int)sub_3007130((__int64)&v7, a2);
    return sub_33E2340(*(_QWORD *)(a1 + 96), v5);
  }
LABEL_3:
  v5 = word_4456340[v7 - 1];
  return sub_33E2340(*(_QWORD *)(a1 + 96), v5);
}
