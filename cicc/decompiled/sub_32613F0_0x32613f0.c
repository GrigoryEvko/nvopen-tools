// Function: sub_32613F0
// Address: 0x32613f0
//
void __fastcall sub_32613F0(unsigned int *a1, __int64 a2)
{
  const __m128i *v2; // r12
  unsigned int v3; // ebx
  unsigned __int16 *v4; // rax
  int v5; // r14d
  __int64 v6; // rax
  __m128i v7; // xmm0
  unsigned __int16 *v8; // rax
  int v9; // ebx
  __int64 v10; // rax
  const __m128i *v11; // r15
  unsigned int v12; // ebx
  __int64 v13; // rax
  unsigned __int16 v14; // r14
  __int64 v15; // rax
  unsigned __int16 *v16; // rax
  int v17; // ebx
  __int64 v18; // rax
  unsigned __int32 v19; // [rsp+Ch] [rbp-94h]
  const __m128i *v20; // [rsp+10h] [rbp-90h]
  __m128i *v21; // [rsp+28h] [rbp-78h]
  __m128i v22; // [rsp+30h] [rbp-70h] BYREF
  __m128i v23; // [rsp+40h] [rbp-60h]
  unsigned __int16 v24; // [rsp+50h] [rbp-50h] BYREF
  __int64 v25; // [rsp+58h] [rbp-48h]
  __int16 v26; // [rsp+60h] [rbp-40h] BYREF
  __int64 v27; // [rsp+68h] [rbp-38h]

  v20 = (const __m128i *)a2;
  if ( a1 != (unsigned int *)a2 )
  {
    v2 = (const __m128i *)(a1 + 4);
    if ( (unsigned int *)a2 != a1 + 4 )
    {
      while ( 1 )
      {
        v8 = (unsigned __int16 *)(*(_QWORD *)(v2->m128i_i64[0] + 48) + 16LL * v2->m128i_u32[2]);
        v9 = *v8;
        v10 = *((_QWORD *)v8 + 1);
        v26 = v9;
        v27 = v10;
        if ( (_WORD)v9 )
        {
          if ( (unsigned __int16)(v9 - 176) <= 0x34u )
          {
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
          }
          v3 = word_4456340[v9 - 1];
        }
        else
        {
          if ( sub_3007100((__int64)&v26) )
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
          v3 = sub_3007130((__int64)&v26, a2);
        }
        v4 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a1 + 48LL) + 16LL * a1[2]);
        v5 = *v4;
        v6 = *((_QWORD *)v4 + 1);
        v24 = v5;
        v25 = v6;
        if ( !(_WORD)v5 )
          break;
        if ( (unsigned __int16)(v5 - 176) <= 0x34u )
        {
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
        }
        if ( word_4456340[v5 - 1] >= v3 )
        {
LABEL_21:
          v11 = v2;
          v22.m128i_i64[0] = v2->m128i_i64[0];
          v19 = v2->m128i_u32[2];
          while ( 1 )
          {
            a2 = v22.m128i_i64[0];
            v21 = (__m128i *)v11;
            v16 = (unsigned __int16 *)(*(_QWORD *)(v22.m128i_i64[0] + 48) + 16LL * v19);
            v17 = *v16;
            v18 = *((_QWORD *)v16 + 1);
            v26 = v17;
            v27 = v18;
            if ( (_WORD)v17 )
            {
              if ( (unsigned __int16)(v17 - 176) <= 0x34u )
              {
                sub_CA17B0(
                  "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be droppe"
                  "d, use EVT::getVectorElementCount() instead");
                sub_CA17B0(
                  "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be droppe"
                  "d, use MVT::getVectorElementCount() instead");
              }
              v12 = word_4456340[v17 - 1];
            }
            else
            {
              if ( sub_3007100((__int64)&v26) )
                sub_CA17B0(
                  "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be droppe"
                  "d, use EVT::getVectorElementCount() instead");
              v12 = sub_3007130((__int64)&v26, a2);
            }
            v13 = *(_QWORD *)(v11[-1].m128i_i64[0] + 48) + 16LL * v11[-1].m128i_u32[2];
            v14 = *(_WORD *)v13;
            v15 = *(_QWORD *)(v13 + 8);
            v24 = v14;
            v25 = v15;
            if ( !v14 )
              break;
            if ( (unsigned __int16)(v14 - 176) <= 0x34u )
            {
              sub_CA17B0(
                "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped,"
                " use EVT::getVectorElementCount() instead");
              sub_CA17B0(
                "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped,"
                " use MVT::getVectorElementCount() instead");
            }
            --v11;
            if ( word_4456340[v14 - 1] >= v12 )
              goto LABEL_37;
LABEL_29:
            v11[1].m128i_i64[0] = v11->m128i_i64[0];
            v11[1].m128i_i32[2] = v11->m128i_i32[2];
          }
          if ( sub_3007100((__int64)&v24) )
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
          --v11;
          if ( (unsigned int)sub_3007130((__int64)&v24, a2) < v12 )
            goto LABEL_29;
LABEL_37:
          ++v2;
          v21->m128i_i64[0] = v22.m128i_i64[0];
          v21->m128i_i32[2] = v19;
          if ( v20 == v2 )
            return;
        }
        else
        {
LABEL_11:
          a2 = (__int64)a1;
          v7 = _mm_loadu_si128(v2);
          if ( a1 != (unsigned int *)v2 )
          {
            v22 = v7;
            memmove(a1 + 4, a1, (char *)v2 - (char *)a1);
            v7 = _mm_load_si128(&v22);
          }
          v23 = v7;
          ++v2;
          *(_QWORD *)a1 = v7.m128i_i64[0];
          a1[2] = v23.m128i_u32[2];
          if ( v20 == v2 )
            return;
        }
      }
      if ( sub_3007100((__int64)&v24) )
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
      if ( (unsigned int)sub_3007130((__int64)&v24, a2) < v3 )
        goto LABEL_11;
      goto LABEL_21;
    }
  }
}
