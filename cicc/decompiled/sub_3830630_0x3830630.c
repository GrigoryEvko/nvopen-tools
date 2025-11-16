// Function: sub_3830630
// Address: 0x3830630
//
_QWORD *__fastcall sub_3830630(__int64 a1, __int64 a2, __m128i a3)
{
  unsigned __int16 *v4; // rax
  __int64 v5; // rsi
  unsigned __int16 v6; // cx
  __int64 v7; // rdx
  unsigned __int16 v8; // dx
  __int64 v9; // rax
  const void *v10; // r14
  unsigned int v11; // eax
  __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // r14
  const void *v19; // [rsp+0h] [rbp-70h]
  __int64 v20; // [rsp+8h] [rbp-68h]
  unsigned __int16 v21; // [rsp+10h] [rbp-60h] BYREF
  __int64 v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+20h] [rbp-50h] BYREF
  int v24; // [rsp+28h] [rbp-48h]
  unsigned __int16 v25; // [rsp+30h] [rbp-40h] BYREF
  __int64 v26; // [rsp+38h] [rbp-38h]

  v4 = *(unsigned __int16 **)(a2 + 48);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *v4;
  v7 = *((_QWORD *)v4 + 1);
  v23 = v5;
  v21 = v6;
  v22 = v7;
  if ( v5 )
  {
    sub_B96E90((__int64)&v23, v5, 1);
    v4 = *(unsigned __int16 **)(a2 + 48);
  }
  v24 = *(_DWORD *)(a2 + 72);
  v8 = *v4;
  v9 = *((_QWORD *)v4 + 1);
  v25 = v8;
  v26 = v9;
  if ( v8 )
  {
    if ( (unsigned __int16)(v8 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
  }
  else if ( sub_3007100((__int64)&v25) )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
  }
  v10 = *(const void **)(a2 + 96);
  if ( v21 )
  {
    if ( (unsigned __int16)(v21 - 176) > 0x34u )
    {
LABEL_8:
      v11 = word_4456340[v21 - 1];
      goto LABEL_11;
    }
  }
  else if ( !sub_3007100((__int64)&v21) )
  {
    goto LABEL_10;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( v21 )
  {
    if ( (unsigned __int16)(v21 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_8;
  }
LABEL_10:
  v11 = sub_3007130((__int64)&v21, v5);
LABEL_11:
  v19 = v10;
  v20 = v11;
  v12 = sub_37AE0F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v14 = v13;
  v15 = sub_37AE0F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v17 = sub_33FCE10(
          *(_QWORD *)(a1 + 8),
          *(unsigned __int16 *)(*(_QWORD *)(v12 + 48) + 16LL * (unsigned int)v14),
          *(_QWORD *)(*(_QWORD *)(v12 + 48) + 16LL * (unsigned int)v14 + 8),
          (__int64)&v23,
          v12,
          v14,
          a3,
          v15,
          v16,
          v19,
          v20);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v17;
}
