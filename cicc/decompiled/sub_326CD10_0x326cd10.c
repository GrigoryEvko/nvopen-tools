// Function: sub_326CD10
// Address: 0x326cd10
//
__int64 __fastcall sub_326CD10(
        const __m128i *a1,
        const __m128i *a2,
        const void *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7)
{
  unsigned int v7; // r14d
  __int64 *v8; // rax
  __int64 v9; // rbx
  unsigned int v10; // eax
  __m128i v12; // xmm0
  __m128i v13; // xmm2
  __int64 v14; // r13
  int *v15; // rcx
  unsigned int i; // eax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rbx
  int v20; // r12d
  __int64 v21; // rcx
  __int64 v22; // r13
  unsigned __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rax
  signed __int64 v26; // rbx
  unsigned __int64 v27; // r8
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rcx
  __int16 v35; // ax
  __int64 v36; // rdx
  int v37; // eax
  unsigned int *v38; // r9
  __int64 v39; // r8
  __int64 v40; // rax
  __int16 v41; // dx
  __int64 v42; // rax
  int v43; // r10d
  bool v44; // al
  unsigned int *v45; // r9
  int v46; // eax
  __int64 v47; // rbx
  __int64 v48; // rdx
  __int64 v49; // r12
  unsigned __int64 v50; // rcx
  const __m128i *v51; // r12
  __m128i *v52; // rax
  char *v53; // r12
  int v54; // [rsp+Ch] [rbp-174h]
  unsigned int *v55; // [rsp+10h] [rbp-170h]
  __int64 v56; // [rsp+10h] [rbp-170h]
  int v57; // [rsp+10h] [rbp-170h]
  __int64 v58; // [rsp+10h] [rbp-170h]
  __int64 v59; // [rsp+18h] [rbp-168h]
  __int64 v60; // [rsp+20h] [rbp-160h]
  int v61; // [rsp+40h] [rbp-140h]
  int v62; // [rsp+40h] [rbp-140h]
  int v63; // [rsp+40h] [rbp-140h]
  unsigned int v64; // [rsp+44h] [rbp-13Ch]
  unsigned int *v66; // [rsp+50h] [rbp-130h]
  int v68; // [rsp+58h] [rbp-128h]
  int v69; // [rsp+60h] [rbp-120h] BYREF
  __int64 v70; // [rsp+68h] [rbp-118h]
  int v71; // [rsp+70h] [rbp-110h]
  __m128i *v72; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v73; // [rsp+88h] [rbp-F8h]
  unsigned int v74; // [rsp+8Ch] [rbp-F4h]
  int v75; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v76; // [rsp+98h] [rbp-E8h]
  int v77; // [rsp+A8h] [rbp-D8h]
  __m128i v78; // [rsp+B0h] [rbp-D0h]

  v7 = 0;
  v66 = (unsigned int *)a2;
  if ( *(_DWORD *)(a6 + 24) != 158 )
    return v7;
  v8 = *(__int64 **)(a6 + 40);
  LOBYTE(v7) = *(_DWORD *)(v8[5] + 24) == 11 || *(_DWORD *)(v8[5] + 24) == 35;
  if ( !(_BYTE)v7 )
    return v7;
  v9 = *v8;
  v10 = *((_DWORD *)v8 + 2);
  v60 = a6;
  v12 = _mm_loadu_si128(a2);
  v13 = _mm_loadu_si128(a1);
  v75 = a4;
  v64 = v10;
  v14 = v9;
  v15 = &v75;
  v72 = (__m128i *)&v75;
  i = 2;
  v59 = v9;
  v74 = 8;
  v77 = 0;
  v73 = 2;
  v76 = v12;
  v78 = v13;
  while ( 1 )
  {
    v17 = i--;
    v18 = (__int64)&v15[6 * v17 - 6];
    v19 = *(_QWORD *)(v18 + 8);
    v20 = *(_DWORD *)v18;
    v73 = i;
    v21 = *(unsigned int *)(v18 + 16);
    if ( v14 == v19 && v64 == (_DWORD)v21 )
      break;
    if ( *(_DWORD *)(v19 + 24) != 159 )
      goto LABEL_5;
    v34 = *(_QWORD *)(v19 + 48) + 16 * v21;
    v35 = *(_WORD *)v34;
    v36 = *(_QWORD *)(v34 + 8);
    LOWORD(v69) = v35;
    v70 = v36;
    if ( v35 )
    {
      if ( (unsigned __int16)(v35 - 176) > 0x34u )
        goto LABEL_29;
    }
    else if ( !sub_3007100((__int64)&v69) )
    {
      goto LABEL_31;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v69 )
    {
      if ( (unsigned __int16)(v69 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
LABEL_29:
      v37 = word_4456340[(unsigned __int16)v69 - 1];
      goto LABEL_32;
    }
LABEL_31:
    v37 = sub_3007130((__int64)&v69, (__int64)a2);
LABEL_32:
    v38 = *(unsigned int **)(v19 + 40);
    LODWORD(v39) = v20 + v37;
    v40 = *(_QWORD *)(*(_QWORD *)v38 + 48LL) + 16LL * v38[2];
    v41 = *(_WORD *)v40;
    v42 = *(_QWORD *)(v40 + 8);
    LOWORD(v69) = v41;
    v70 = v42;
    if ( v41 )
    {
      if ( (unsigned __int16)(v41 - 176) > 0x34u )
        goto LABEL_34;
    }
    else
    {
      v61 = v39;
      v55 = v38;
      v44 = sub_3007100((__int64)&v69);
      v45 = v55;
      LODWORD(v39) = v61;
      if ( !v44 )
        goto LABEL_36;
    }
    v57 = v39;
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    LODWORD(v39) = v57;
    if ( (_WORD)v69 )
    {
      if ( (unsigned __int16)(v69 - 176) <= 0x34u )
      {
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
        LODWORD(v39) = v57;
      }
LABEL_34:
      a6 = *(_QWORD *)(v19 + 40);
      v43 = word_4456340[(unsigned __int16)v69 - 1];
      goto LABEL_37;
    }
    v45 = *(unsigned int **)(v19 + 40);
LABEL_36:
    v62 = v39;
    v56 = (__int64)v45;
    v46 = sub_3007130((__int64)&v69, (__int64)a2);
    LODWORD(v39) = v62;
    a6 = v56;
    v43 = v46;
LABEL_37:
    v47 = a6 + 40LL * *(unsigned int *)(v19 + 64);
    for ( i = v73; v47 != a6; v47 -= 40 )
    {
      a2 = v72;
      v48 = i;
      v49 = *(_QWORD *)(v47 - 40);
      v39 = (unsigned int)(v39 - v43);
      v50 = (unsigned __int64)v72 + 24 * i;
      if ( i < (unsigned __int64)v74 )
      {
        if ( v50 )
        {
          *(_DWORD *)(v50 + 16) = *(_DWORD *)(v47 - 32);
          *(_DWORD *)v50 = v39;
          *(_QWORD *)(v50 + 8) = v49;
          i = v73;
        }
        v73 = ++i;
        continue;
      }
      v71 = *(_DWORD *)(v47 - 32);
      v70 = v49;
      v51 = (const __m128i *)&v69;
      v69 = v39;
      if ( v74 < (unsigned __int64)i + 1 )
      {
        if ( v72 > (__m128i *)&v69 )
        {
          v54 = v43;
          v63 = v39;
          v58 = a6;
LABEL_58:
          sub_C8D5F0((__int64)&v72, &v75, i + 1LL, 0x18u, v39, a6);
          a2 = v72;
          v48 = v73;
          a6 = v58;
          LODWORD(v39) = v63;
          v43 = v54;
          goto LABEL_45;
        }
        v54 = v43;
        v63 = v39;
        v58 = a6;
        if ( v50 <= (unsigned __int64)&v69 )
          goto LABEL_58;
        v53 = (char *)((char *)&v69 - (char *)v72);
        sub_C8D5F0((__int64)&v72, &v75, i + 1LL, 0x18u, v39, a6);
        a2 = v72;
        v48 = v73;
        v43 = v54;
        LODWORD(v39) = v63;
        a6 = v58;
        v51 = (const __m128i *)&v53[(_QWORD)v72];
      }
LABEL_45:
      v52 = (__m128i *)((char *)a2 + 24 * v48);
      *v52 = _mm_loadu_si128(v51);
      v52[1].m128i_i64[0] = v51[1].m128i_i64[0];
      i = ++v73;
    }
LABEL_5:
    if ( !i )
    {
      v22 = v60;
      goto LABEL_21;
    }
    v15 = (int *)v72;
  }
  v22 = v60;
  if ( v20 != -1 )
  {
LABEL_10:
    v23 = *(unsigned int *)(a5 + 12);
    v24 = 0;
    LODWORD(v25) = 0;
    *(_DWORD *)(a5 + 8) = 0;
    v26 = 4 * a4;
    v27 = (4 * a4) >> 2;
    if ( v27 > v23 )
    {
      sub_C8D5F0(a5, (const void *)(a5 + 16), v26 >> 2, 4u, v27, a6);
      v25 = *(unsigned int *)(a5 + 8);
      v27 = v26 >> 2;
      v24 = 4 * v25;
    }
    v28 = *(_QWORD *)a5;
    if ( v26 )
    {
      v68 = v27;
      memcpy((void *)(v24 + v28), a3, v26);
      v28 = *(_QWORD *)a5;
      LODWORD(v25) = *(_DWORD *)(a5 + 8);
      LODWORD(v27) = v68;
    }
    *(_DWORD *)(a5 + 8) = v27 + v25;
    v29 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v22 + 40) + 40LL) + 96LL);
    if ( *(_DWORD *)(v29 + 32) <= 0x40u )
      v30 = *(_QWORD *)(v29 + 24);
    else
      v30 = **(_QWORD **)(v29 + 24);
    *(_DWORD *)(v28 + 4LL * a7) = v20 + v30;
    goto LABEL_17;
  }
LABEL_21:
  if ( *(_DWORD *)(*(_QWORD *)v66 + 24LL) == 51 )
  {
    v32 = *(_QWORD *)(*(_QWORD *)v66 + 48LL) + 16LL * v66[2];
    v33 = *(_QWORD *)(v59 + 48) + 16LL * v64;
    if ( *(_WORD *)v32 == *(_WORD *)v33 && (*(_QWORD *)(v32 + 8) == *(_QWORD *)(v33 + 8) || *(_WORD *)v33) )
    {
      v20 = a4;
      *(_QWORD *)v66 = v59;
      v66[2] = v64;
      goto LABEL_10;
    }
  }
  v7 = 0;
LABEL_17:
  if ( v72 != (__m128i *)&v75 )
    _libc_free((unsigned __int64)v72);
  return v7;
}
