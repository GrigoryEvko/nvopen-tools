// Function: sub_3263A70
// Address: 0x3263a70
//
char *__fastcall sub_3263A70(
        unsigned int *a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        __int64 a7)
{
  __int64 v7; // rax
  unsigned int *v8; // r14
  unsigned int *v9; // r13
  unsigned int *v10; // r12
  __int64 v11; // rbx
  __int64 v12; // r15
  unsigned int *v13; // r9
  unsigned int *i; // r11
  __int64 v15; // r12
  unsigned int *v16; // rax
  int v17; // r11d
  const __m128i *v18; // r9
  __m128i *v19; // r13
  __int64 v20; // rcx
  const __m128i *v21; // r10
  size_t v22; // r10
  __int64 v23; // rax
  signed __int64 v24; // rbx
  char *result; // rax
  __int64 v26; // rax
  __int64 v27; // rsi
  unsigned __int16 *v28; // rax
  int v29; // ebx
  __int64 v30; // rax
  unsigned int v31; // ebx
  unsigned __int16 *v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int *v36; // r15
  __int64 v37; // rbx
  unsigned int *v38; // rax
  unsigned int *v39; // r14
  unsigned int *v40; // r12
  unsigned int *v41; // r15
  __int64 v42; // rsi
  unsigned __int16 *v43; // rax
  int v44; // r13d
  __int64 v45; // rax
  unsigned int v46; // r13d
  unsigned __int16 *v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rax
  unsigned int v50; // eax
  size_t v51; // rdx
  unsigned int *v52; // rsi
  unsigned int *v53; // rdi
  unsigned int *v54; // rax
  size_t v55; // r8
  const __m128i *v56; // rax
  unsigned int *v57; // rdx
  const __m128i *v58; // rax
  int v59; // [rsp+8h] [rbp-88h]
  const __m128i *v60; // [rsp+8h] [rbp-88h]
  int v61; // [rsp+10h] [rbp-80h]
  int v62; // [rsp+10h] [rbp-80h]
  int v63; // [rsp+10h] [rbp-80h]
  int v64; // [rsp+10h] [rbp-80h]
  int v65; // [rsp+10h] [rbp-80h]
  const __m128i *src; // [rsp+18h] [rbp-78h]
  void *srca; // [rsp+18h] [rbp-78h]
  void *srcb; // [rsp+18h] [rbp-78h]
  int srcc; // [rsp+18h] [rbp-78h]
  int srcd; // [rsp+18h] [rbp-78h]
  int srce; // [rsp+18h] [rbp-78h]
  void *srcf; // [rsp+18h] [rbp-78h]
  int srcg; // [rsp+18h] [rbp-78h]
  int srch; // [rsp+18h] [rbp-78h]
  unsigned int *v75; // [rsp+20h] [rbp-70h]
  const __m128i *v76; // [rsp+20h] [rbp-70h]
  int v77; // [rsp+20h] [rbp-70h]
  void *v78; // [rsp+20h] [rbp-70h]
  unsigned int *v79; // [rsp+20h] [rbp-70h]
  const __m128i *v80; // [rsp+20h] [rbp-70h]
  void *v81; // [rsp+20h] [rbp-70h]
  int v82; // [rsp+20h] [rbp-70h]
  int v83; // [rsp+20h] [rbp-70h]
  int v84; // [rsp+20h] [rbp-70h]
  unsigned int *desta; // [rsp+28h] [rbp-68h]
  __m128i *v87; // [rsp+30h] [rbp-60h]
  unsigned int *v88; // [rsp+30h] [rbp-60h]
  __int16 v90; // [rsp+40h] [rbp-50h] BYREF
  __int64 v91; // [rsp+48h] [rbp-48h]
  __int16 v92; // [rsp+50h] [rbp-40h] BYREF
  __int64 v93; // [rsp+58h] [rbp-38h]

  v7 = a5;
  v8 = a2;
  v9 = a1;
  v10 = a6;
  if ( a7 <= a5 )
    v7 = a7;
  if ( a4 <= v7 )
    goto LABEL_22;
  v11 = a5;
  if ( a7 >= a5 )
    goto LABEL_47;
  v12 = a4;
  v13 = a2;
  for ( i = a1; ; i = v79 )
  {
    if ( v11 < v12 )
    {
      srcd = (int)i;
      v80 = (const __m128i *)v13;
      v19 = (__m128i *)&i[4 * (v12 / 2)];
      v54 = sub_3260630(v13, a3, (unsigned int *)v19);
      v18 = v80;
      v17 = srcd;
      v87 = (__m128i *)v54;
      v20 = v12 / 2;
      v15 = ((char *)v54 - (char *)v80) >> 4;
    }
    else
    {
      src = (const __m128i *)v13;
      v75 = i;
      v15 = v11 / 2;
      v87 = (__m128i *)&v13[4 * (v11 / 2)];
      v16 = sub_3260820(i, (__int64)v13, (unsigned int *)v87);
      v17 = (int)v75;
      v18 = src;
      v19 = (__m128i *)v16;
      v20 = ((char *)v16 - (char *)v75) >> 4;
    }
    v12 -= v20;
    if ( v12 <= v15 || v15 > a7 )
    {
      if ( v12 > a7 )
      {
        srch = v17;
        v84 = v20;
        v58 = sub_325E770(v19, v18, v87);
        v17 = srch;
        LODWORD(v20) = v84;
        v21 = v58;
      }
      else
      {
        v21 = v87;
        if ( v12 )
        {
          v55 = (char *)v18 - (char *)v19;
          if ( v18 != v19 )
          {
            v64 = v17;
            v60 = v18;
            srce = v20;
            v81 = (void *)((char *)v18 - (char *)v19);
            memmove(a6, v19, (char *)v18 - (char *)v19);
            v18 = v60;
            v17 = v64;
            LODWORD(v20) = srce;
            v55 = (size_t)v81;
          }
          if ( v18 != v87 )
          {
            v65 = v17;
            srcf = (void *)v55;
            v82 = v20;
            memmove(v19, v18, (char *)v87 - (char *)v18);
            v17 = v65;
            v55 = (size_t)srcf;
            LODWORD(v20) = v82;
          }
          v21 = (__m128i *)((char *)v87 - v55);
          if ( v55 )
          {
            srcg = v17;
            v83 = v20;
            v56 = (const __m128i *)memmove((char *)v87 - v55, a6, v55);
            LODWORD(v20) = v83;
            v17 = srcg;
            v21 = v56;
          }
        }
      }
    }
    else
    {
      v21 = v19;
      if ( v15 )
      {
        v22 = (char *)v87 - (char *)v18;
        if ( v18 != v87 )
        {
          v61 = v20;
          v59 = v17;
          srca = (void *)((char *)v87 - (char *)v18);
          v76 = v18;
          memmove(a6, v18, (char *)v87 - (char *)v18);
          v17 = v59;
          LODWORD(v20) = v61;
          v22 = (size_t)srca;
          v18 = v76;
        }
        if ( v18 != v19 )
        {
          v62 = v17;
          srcb = (void *)v22;
          v77 = v20;
          memmove((char *)v87 - ((char *)v18 - (char *)v19), v19, (char *)v18 - (char *)v19);
          v17 = v62;
          v22 = (size_t)srcb;
          LODWORD(v20) = v77;
        }
        if ( v22 )
        {
          v63 = v17;
          srcc = v20;
          v78 = (void *)v22;
          memmove(v19, a6, v22);
          v17 = v63;
          LODWORD(v20) = srcc;
          v22 = (size_t)v78;
        }
        v21 = (__m128i *)((char *)v19 + v22);
      }
    }
    v11 -= v15;
    v79 = (unsigned int *)v21;
    sub_3263A70(v17, (_DWORD)v19, (_DWORD)v21, v20, v15, (_DWORD)a6, a7);
    v23 = a7;
    if ( v11 <= a7 )
      v23 = v11;
    if ( v12 <= v23 )
    {
      v10 = a6;
      v8 = (unsigned int *)v87;
      v9 = v79;
LABEL_22:
      v24 = (char *)v8 - (char *)v9;
      if ( v9 != v8 )
        memmove(v10, v9, (char *)v8 - (char *)v9);
      result = (char *)v10 + v24;
      v88 = (unsigned int *)((char *)v10 + v24);
      if ( (unsigned int *)((char *)v10 + v24) == v10 )
        return result;
      while ( (unsigned int *)a3 != v8 )
      {
        v27 = *(_QWORD *)v8;
        v28 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v8 + 48LL) + 16LL * v8[2]);
        v29 = *v28;
        v30 = *((_QWORD *)v28 + 1);
        v92 = v29;
        v93 = v30;
        if ( (_WORD)v29 )
        {
          if ( (unsigned __int16)(v29 - 176) <= 0x34u )
          {
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
          }
          v31 = word_4456340[v29 - 1];
        }
        else
        {
          if ( sub_3007100((__int64)&v92) )
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
          v31 = sub_3007130((__int64)&v92, v27);
        }
        v32 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v10 + 48LL) + 16LL * v10[2]);
        v33 = *v32;
        v34 = *((_QWORD *)v32 + 1);
        v90 = v33;
        v91 = v34;
        if ( (_WORD)v33 )
        {
          if ( (unsigned __int16)(v33 - 176) <= 0x34u )
          {
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
          }
          if ( word_4456340[(unsigned __int16)v33 - 1] < v31 )
          {
LABEL_30:
            v26 = *(_QWORD *)v8;
            v9 += 4;
            v8 += 4;
            *((_QWORD *)v9 - 2) = v26;
            result = (char *)*(v8 - 2);
            *(v9 - 2) = (unsigned int)result;
            if ( v88 == v10 )
              return result;
            continue;
          }
        }
        else
        {
          if ( sub_3007100((__int64)&v90) )
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
          if ( (unsigned int)sub_3007130((__int64)&v90, v33) < v31 )
            goto LABEL_30;
        }
        v35 = *(_QWORD *)v10;
        v9 += 4;
        v10 += 4;
        *((_QWORD *)v9 - 2) = v35;
        result = (char *)*(v10 - 2);
        *(v9 - 2) = (unsigned int)result;
        if ( v88 == v10 )
          return result;
      }
      if ( v88 == v10 )
        return result;
      v51 = (char *)v88 - (char *)v10;
      v52 = v10;
      v53 = v9;
      return (char *)memmove(v53, v52, v51);
    }
    if ( v11 <= a7 )
      break;
    v13 = (unsigned int *)v87;
  }
  v10 = a6;
  v8 = (unsigned int *)v87;
  v9 = v79;
LABEL_47:
  result = (char *)a3;
  if ( (unsigned int *)a3 != v8 )
    result = (char *)memmove(v10, v8, a3 - (_QWORD)v8);
  v36 = (unsigned int *)((char *)v10 + a3 - (_QWORD)v8);
  if ( v8 == v9 )
  {
    if ( v10 == v36 )
      return result;
    v51 = a3 - (_QWORD)v8;
    v53 = v8;
LABEL_85:
    v52 = v10;
    return (char *)memmove(v53, v52, v51);
  }
  if ( v10 != v36 )
  {
    desta = v9;
    v37 = a3;
    v38 = v10;
    v39 = v8 - 4;
    v40 = v36 - 4;
    v41 = v38;
    while ( 1 )
    {
      while ( 1 )
      {
        v42 = *(_QWORD *)v40;
        v43 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v40 + 48LL) + 16LL * v40[2]);
        v44 = *v43;
        v45 = *((_QWORD *)v43 + 1);
        v92 = v44;
        v93 = v45;
        if ( (_WORD)v44 )
        {
          if ( (unsigned __int16)(v44 - 176) <= 0x34u )
          {
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
          }
          v46 = word_4456340[v44 - 1];
        }
        else
        {
          if ( sub_3007100((__int64)&v92) )
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
          v46 = sub_3007130((__int64)&v92, v42);
        }
        v47 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v39 + 48LL) + 16LL * v39[2]);
        v48 = *v47;
        v49 = *((_QWORD *)v47 + 1);
        v90 = v48;
        v91 = v49;
        if ( (_WORD)v48 )
        {
          if ( (unsigned __int16)(v48 - 176) <= 0x34u )
          {
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
          }
          v50 = word_4456340[(unsigned __int16)v48 - 1];
        }
        else
        {
          if ( sub_3007100((__int64)&v90) )
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
          v50 = sub_3007130((__int64)&v90, v48);
        }
        v37 -= 16;
        if ( v50 < v46 )
          break;
        *(_QWORD *)v37 = *(_QWORD *)v40;
        result = (char *)v40[2];
        *(_DWORD *)(v37 + 8) = (_DWORD)result;
        if ( v41 == v40 )
          return result;
        v40 -= 4;
      }
      *(_QWORD *)v37 = *(_QWORD *)v39;
      *(_DWORD *)(v37 + 8) = v39[2];
      if ( v39 == desta )
        break;
      v39 -= 4;
    }
    result = (char *)v41;
    v57 = v40 + 4;
    v10 = v41;
    if ( v41 != v57 )
    {
      v51 = (char *)v57 - (char *)v41;
      v53 = (unsigned int *)(v37 - v51);
      goto LABEL_85;
    }
  }
  return result;
}
