// Function: sub_3260630
// Address: 0x3260630
//
unsigned int *__fastcall sub_3260630(unsigned int *a1, __int64 a2, unsigned int *a3)
{
  __int64 v3; // rsi
  __int64 v4; // r12
  __int64 v6; // r13
  unsigned int *v7; // rbx
  __int64 v8; // rax
  unsigned __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int16 v14; // dx
  unsigned __int16 v15; // [rsp+8h] [rbp-68h]
  unsigned int *v16; // [rsp+10h] [rbp-60h]
  unsigned int v17; // [rsp+1Ch] [rbp-54h]
  unsigned __int16 v18; // [rsp+1Ch] [rbp-54h]
  unsigned __int16 v19; // [rsp+20h] [rbp-50h] BYREF
  __int64 v20; // [rsp+28h] [rbp-48h]
  unsigned __int16 v21; // [rsp+30h] [rbp-40h] BYREF
  __int64 v22; // [rsp+38h] [rbp-38h]

  v3 = a2 - (_QWORD)a1;
  v4 = v3 >> 4;
  v16 = a1;
  if ( v3 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v6 = v4 >> 1;
        v7 = &v16[4 * (v4 >> 1)];
        v8 = *(_QWORD *)(*(_QWORD *)v7 + 48LL) + 16LL * v7[2];
        v9 = *(_WORD *)v8;
        v10 = *(_QWORD *)(v8 + 8);
        v21 = v9;
        v22 = v10;
        if ( v9 )
        {
          if ( (unsigned __int16)(v9 - 176) <= 0x34u )
          {
            v18 = v9;
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
            v9 = v18;
          }
          v17 = word_4456340[v9 - 1];
        }
        else
        {
          if ( sub_3007100((__int64)&v21) )
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
          v17 = sub_3007130((__int64)&v21, v3);
        }
        v11 = *(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * a3[2];
        v14 = *(_WORD *)v11;
        v12 = *(_QWORD *)(v11 + 8);
        v19 = v14;
        v20 = v12;
        if ( !v14 )
          break;
        if ( (unsigned __int16)(v14 - 176) <= 0x34u )
        {
          v15 = v14;
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
          v14 = v15;
        }
        if ( word_4456340[v14 - 1] >= v17 )
          goto LABEL_15;
LABEL_6:
        v4 = v4 - v6 - 1;
        v16 = v7 + 4;
        if ( v4 <= 0 )
          return v16;
      }
      if ( sub_3007100((__int64)&v19) )
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
      if ( (unsigned int)sub_3007130((__int64)&v19, v3) < v17 )
        goto LABEL_6;
LABEL_15:
      v4 >>= 1;
    }
    while ( v6 > 0 );
  }
  return v16;
}
