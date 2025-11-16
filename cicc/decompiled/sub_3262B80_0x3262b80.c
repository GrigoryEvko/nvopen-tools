// Function: sub_3262B80
// Address: 0x3262b80
//
void __fastcall sub_3262B80(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  unsigned int *v6; // r11
  unsigned int *v7; // r10
  __int64 v8; // rbx
  __int64 v9; // r13
  unsigned int *v10; // rax
  const __m128i *v11; // r10
  const __m128i *v12; // r9
  unsigned int *v13; // r11
  __m128i *v14; // r15
  __int64 v15; // r14
  __m128i *v16; // rdx
  unsigned int *v17; // rax
  __int64 v18; // rax
  unsigned __int16 v19; // bx
  __int64 v20; // rax
  unsigned int v21; // ebx
  __int64 v22; // rax
  unsigned __int16 v23; // r12
  __int64 v24; // rax
  unsigned int v25; // eax
  __m128i v26; // xmm0
  bool v27; // al
  __m128i *v28; // rdx
  unsigned int v29; // eax
  bool v30; // al
  __m128i *v31; // rdx
  unsigned int *v33; // [rsp+8h] [rbp-78h]
  const __m128i *v34; // [rsp+8h] [rbp-78h]
  unsigned int *v35; // [rsp+10h] [rbp-70h]
  unsigned int *v36; // [rsp+10h] [rbp-70h]
  const __m128i *v37; // [rsp+10h] [rbp-70h]
  const __m128i *v38; // [rsp+18h] [rbp-68h]
  const __m128i *v39; // [rsp+18h] [rbp-68h]
  unsigned int *v40; // [rsp+18h] [rbp-68h]
  __m128i *v41; // [rsp+18h] [rbp-68h]
  __m128i *v42; // [rsp+18h] [rbp-68h]
  __m128i *v43; // [rsp+18h] [rbp-68h]
  __m128i *v44; // [rsp+18h] [rbp-68h]
  __m128i *v45; // [rsp+18h] [rbp-68h]
  __m128i *v46; // [rsp+18h] [rbp-68h]
  unsigned __int16 v47; // [rsp+30h] [rbp-50h] BYREF
  __int64 v48; // [rsp+38h] [rbp-48h]
  unsigned __int16 v49; // [rsp+40h] [rbp-40h] BYREF
  __int64 v50; // [rsp+48h] [rbp-38h]

  if ( a4 )
  {
    v5 = a5;
    if ( a5 )
    {
      v6 = (unsigned int *)a1;
      v7 = (unsigned int *)a2;
      v8 = a4;
      if ( a4 + a5 == 2 )
      {
        v14 = (__m128i *)a2;
        v16 = a1;
LABEL_12:
        v18 = *(_QWORD *)(v14->m128i_i64[0] + 48) + 16LL * v14->m128i_u32[2];
        v19 = *(_WORD *)v18;
        v20 = *(_QWORD *)(v18 + 8);
        v49 = v19;
        v50 = v20;
        if ( v19 )
        {
          if ( (unsigned __int16)(v19 - 176) <= 0x34u )
          {
            v46 = v16;
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
            v16 = v46;
          }
          v21 = word_4456340[v19 - 1];
        }
        else
        {
          v41 = v16;
          v27 = sub_3007100((__int64)&v49);
          v28 = v41;
          if ( v27 )
          {
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            v28 = v41;
          }
          v42 = v28;
          v29 = sub_3007130((__int64)&v49, a2);
          v16 = v42;
          v21 = v29;
        }
        v22 = *(_QWORD *)(v16->m128i_i64[0] + 48) + 16LL * v16->m128i_u32[2];
        v23 = *(_WORD *)v22;
        v24 = *(_QWORD *)(v22 + 8);
        v47 = v23;
        v48 = v24;
        if ( v23 )
        {
          if ( (unsigned __int16)(v23 - 176) <= 0x34u )
          {
            v45 = v16;
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
            v16 = v45;
          }
          v25 = word_4456340[v23 - 1];
        }
        else
        {
          v43 = v16;
          v30 = sub_3007100((__int64)&v47);
          v31 = v43;
          if ( v30 )
          {
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            v31 = v43;
          }
          v44 = v31;
          v25 = sub_3007130((__int64)&v47, a2);
          v16 = v44;
        }
        if ( v25 < v21 )
        {
          v26 = _mm_loadu_si128(v16);
          v16->m128i_i64[0] = v14->m128i_i64[0];
          v16->m128i_i32[2] = v14->m128i_i32[2];
          v14->m128i_i64[0] = v26.m128i_i64[0];
          v14->m128i_i32[2] = v26.m128i_i32[2];
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v33 = v6;
        v38 = (const __m128i *)v7;
        v9 = v8 / 2;
        v35 = &v6[4 * (v8 / 2)];
        v10 = sub_3260630(v7, a3, v35);
        v11 = v38;
        v12 = (const __m128i *)v35;
        v13 = v33;
        v14 = (__m128i *)v10;
        v15 = ((char *)v10 - (char *)v38) >> 4;
        while ( 1 )
        {
          v36 = v13;
          v39 = v12;
          v5 -= v15;
          v34 = sub_325E770(v12, v11, v14);
          a2 = (__int64)v39;
          sub_3262B80(v36, v39, v34, v9, v15);
          v8 -= v9;
          if ( !v8 )
            break;
          v16 = (__m128i *)v34;
          if ( !v5 )
            break;
          if ( v5 + v8 == 2 )
            goto LABEL_12;
          v7 = (unsigned int *)v14;
          v6 = (unsigned int *)v34;
          if ( v8 > v5 )
            goto LABEL_5;
LABEL_10:
          v37 = (const __m128i *)v7;
          v40 = v6;
          v15 = v5 / 2;
          v14 = (__m128i *)&v7[4 * (v5 / 2)];
          v17 = sub_3260820(v6, (__int64)v7, (unsigned int *)v14);
          v13 = v40;
          v11 = v37;
          v12 = (const __m128i *)v17;
          v9 = ((char *)v17 - (char *)v40) >> 4;
        }
      }
    }
  }
}
