// Function: sub_3032270
// Address: 0x3032270
//
__int64 __fastcall sub_3032270(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  _QWORD *v3; // rax
  unsigned __int16 *v4; // rdx
  int v5; // ebx
  __int64 v6; // rax
  int v8; // eax
  unsigned __int16 v9; // bx
  __int64 v10; // rax
  int v11; // eax
  unsigned __int16 v12; // [rsp+0h] [rbp-30h] BYREF
  __int64 v13; // [rsp+8h] [rbp-28h]
  __int16 v14; // [rsp+10h] [rbp-20h] BYREF
  __int64 v15; // [rsp+18h] [rbp-18h]

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL) + 96LL);
  v3 = *(_QWORD **)(v2 + 24);
  if ( *(_DWORD *)(v2 + 32) > 0x40u )
    v3 = (_QWORD *)*v3;
  v4 = *(unsigned __int16 **)(a1 + 48);
  if ( v3 != (_QWORD *)8938 )
  {
    v5 = *v4;
    v6 = *((_QWORD *)v4 + 1);
    v14 = v5;
    v15 = v6;
    if ( (_WORD)v5 )
    {
      if ( (unsigned __int16)(v5 - 17) > 0xD3u )
        return 556;
      if ( (unsigned __int16)(v5 - 176) <= 0x34u )
      {
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      }
      v8 = word_4456340[v5 - 1];
      if ( v8 == 2 )
        return 557;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v14) )
        return 556;
      if ( sub_3007100((__int64)&v14) )
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
      v8 = sub_3007130((__int64)&v14, a2);
      if ( v8 == 2 )
        return 557;
    }
    if ( v8 == 4 )
      return 558;
LABEL_31:
    BUG();
  }
  v9 = *v4;
  v10 = *((_QWORD *)v4 + 1);
  v12 = v9;
  v13 = v10;
  if ( v9 )
  {
    if ( (unsigned __int16)(v9 - 17) <= 0xD3u )
    {
      if ( (unsigned __int16)(v9 - 176) <= 0x34u )
      {
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      }
      v11 = word_4456340[v9 - 1];
      goto LABEL_18;
    }
    return 562;
  }
  if ( !sub_30070B0((__int64)&v12) )
    return 562;
  if ( sub_3007100((__int64)&v12) )
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
  v11 = sub_3007130((__int64)&v12, a2);
LABEL_18:
  if ( v11 == 2 )
    return 563;
  if ( v11 != 4 )
    goto LABEL_31;
  return 564;
}
