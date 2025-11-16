// Function: sub_3033D10
// Address: 0x3033d10
//
void __fastcall sub_3033D10(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 *a5, __int64 a6)
{
  _OWORD *v6; // rbx
  __int64 *v8; // r8
  __int64 v11; // rsi
  __int16 *v12; // rdx
  __int16 v13; // ax
  __int64 v14; // rdx
  __int64 v15; // r12
  bool v16; // al
  unsigned int v17; // eax
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // r13
  int v21; // ebx
  __int64 **v22; // rax
  __int64 **v23; // rax
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rdi
  unsigned __int64 v27; // rsi
  int v28; // r15d
  __int64 **v29; // rdx
  __int64 v30; // rax
  __m128i v31; // xmm1
  __m128i v32; // xmm2
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // r8
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r9
  unsigned __int64 v40; // rdx
  __int64 *v41; // rax
  __int64 v42; // rax
  __m128i v43; // xmm0
  __int64 v44; // rax
  int v45; // ecx
  __int64 **v46; // r15
  __int64 v47; // rax
  int v48; // r9d
  __int64 v49; // r14
  __int64 **v50; // rsi
  __int128 v51; // rax
  unsigned int v52; // r15d
  unsigned __int64 v53; // rcx
  unsigned __int64 v54; // r13
  unsigned int v55; // r12d
  unsigned __int64 v56; // rbx
  __int64 *v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // r8
  __int64 v60; // rax
  __int64 v61; // r9
  __int64 *v62; // rax
  unsigned __int64 v63; // rcx
  __int64 v64; // rax
  __int64 *v65; // rax
  unsigned __int64 v66; // rcx
  __int64 v67; // rax
  __int64 *v68; // rax
  unsigned __int64 v69; // rdi
  __m128i v70; // xmm0
  __m128i v71; // [rsp+0h] [rbp-1E0h] BYREF
  __int64 **v72; // [rsp+10h] [rbp-1D0h]
  _OWORD *v73; // [rsp+18h] [rbp-1C8h]
  __m128i v74; // [rsp+20h] [rbp-1C0h] BYREF
  __int64 *v75; // [rsp+30h] [rbp-1B0h]
  __int64 *v76; // [rsp+38h] [rbp-1A8h]
  __int64 v77; // [rsp+40h] [rbp-1A0h]
  __int64 v78; // [rsp+48h] [rbp-198h]
  __int64 v79; // [rsp+50h] [rbp-190h] BYREF
  int v80; // [rsp+58h] [rbp-188h]
  int v81; // [rsp+60h] [rbp-180h] BYREF
  __int64 v82; // [rsp+68h] [rbp-178h]
  _BYTE *v83; // [rsp+70h] [rbp-170h] BYREF
  __int64 v84; // [rsp+78h] [rbp-168h]
  _BYTE v85[64]; // [rsp+80h] [rbp-160h] BYREF
  __int64 *v86; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v87; // [rsp+C8h] [rbp-118h]
  _BYTE v88[80]; // [rsp+D0h] [rbp-110h] BYREF
  _OWORD *v89; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v90; // [rsp+128h] [rbp-B8h]
  _OWORD v91[11]; // [rsp+130h] [rbp-B0h] BYREF

  v8 = a4;
  v77 = a2;
  v11 = *(_QWORD *)(a1 + 80);
  v74.m128i_i32[0] = a6;
  v79 = v11;
  if ( v11 )
  {
    v76 = a4;
    sub_B96E90((__int64)&v79, v11, 1);
    v8 = v76;
  }
  v12 = *(__int16 **)(a1 + 48);
  v80 = *(_DWORD *)(a1 + 72);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  LOWORD(v81) = v13;
  v82 = v14;
  if ( v13 )
  {
    if ( (unsigned __int16)(v13 - 17) > 0xD3u )
      goto LABEL_5;
    if ( (unsigned __int16)(v13 - 176) > 0x34u )
      goto LABEL_9;
  }
  else
  {
    v76 = v8;
    if ( !sub_30070B0((__int64)&v81) )
      goto LABEL_5;
    v16 = sub_3007100((__int64)&v81);
    v8 = v76;
    if ( !v16 )
      goto LABEL_12;
  }
  v76 = v8;
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  v8 = v76;
  if ( !(_WORD)v81 )
  {
LABEL_12:
    v76 = v8;
    v17 = sub_3007130((__int64)&v81, v11);
    v8 = v76;
    v15 = v17;
    goto LABEL_13;
  }
  if ( (unsigned __int16)(v81 - 176) <= 0x34u )
  {
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
    v8 = v76;
  }
LABEL_9:
  v15 = word_4456340[(unsigned __int16)v81 - 1];
LABEL_13:
  v18 = (__int64 *)v88;
  v76 = (__int64 *)v88;
  v86 = (__int64 *)v88;
  v87 = 0x500000000LL;
  if ( (_DWORD)v15 )
  {
    a6 = 0;
    v75 = (__int64 *)a3;
    v19 = 0;
    v20 = v8;
    v73 = v6;
    v21 = 0;
    while ( 1 )
    {
      ++v21;
      v22 = (__int64 **)&v18[2 * v19];
      *v22 = v20;
      v22[1] = a5;
      v19 = (unsigned int)(v87 + 1);
      LODWORD(v87) = v87 + 1;
      if ( v21 == (_DWORD)v15 )
        break;
      if ( v19 + 1 > (unsigned __int64)HIDWORD(v87) )
      {
        v72 = &v86;
        sub_C8D5F0((__int64)&v86, v76, v19 + 1, 0x10u, (__int64)v8, a6);
        v19 = (unsigned int)v87;
      }
      v18 = v86;
    }
    v8 = v20;
    v6 = v73;
    a3 = (__int64)v75;
    if ( v19 + 1 > (unsigned __int64)HIDWORD(v87) )
    {
      v75 = v8;
      sub_C8D5F0((__int64)&v86, v76, v19 + 1, 0x10u, (__int64)v8, a6);
      v8 = v75;
      v23 = (__int64 **)&v86[2 * (unsigned int)v87];
    }
    else
    {
      v23 = (__int64 **)&v86[2 * v19];
    }
  }
  else
  {
    v23 = (__int64 **)v76;
  }
  *v23 = v8;
  v23[1] = a5;
  LODWORD(v87) = v87 + 1;
  v24 = (unsigned int)v87;
  if ( (unsigned __int64)(unsigned int)v87 + 1 > HIDWORD(v87) )
  {
    sub_C8D5F0((__int64)&v86, v76, (unsigned int)v87 + 1LL, 0x10u, (__int64)v8, a6);
    v24 = (unsigned int)v87;
  }
  v25 = &v86[2 * v24];
  v26 = v77;
  *v25 = 1;
  v27 = (unsigned __int64)v86;
  v25[1] = 0;
  LODWORD(v87) = v87 + 1;
  v28 = sub_33E5830(v26, v27);
  v72 = v29;
  v89 = v91;
  v73 = v91;
  v90 = 0x800000000LL;
  v30 = *(_QWORD *)(a1 + 40);
  v31 = _mm_loadu_si128((const __m128i *)v30);
  v32 = _mm_loadu_si128((const __m128i *)(v30 + 40));
  LODWORD(v90) = 2;
  v33 = *(_QWORD *)(v30 + 80);
  v91[1] = v32;
  v34 = *(_QWORD *)(v33 + 96);
  v91[0] = v31;
  if ( *(_DWORD *)(v34 + 32) <= 0x40u )
    v35 = *(_QWORD *)(v34 + 24);
  else
    v35 = **(_QWORD **)(v34 + 24);
  v75 = &v79;
  v36 = sub_3400BD0(v77, v35, (unsigned int)&v79, 7, 0, 1, 0);
  v37 = (unsigned int)v90;
  v39 = v38;
  v40 = (unsigned int)v90 + 1LL;
  if ( v40 > HIDWORD(v90) )
  {
    v71.m128i_i64[0] = v36;
    v71.m128i_i64[1] = v39;
    sub_C8D5F0((__int64)&v89, v73, v40, 0x10u, v36, v39);
    v37 = (unsigned int)v90;
    v39 = v71.m128i_i64[1];
    v36 = v71.m128i_i64[0];
  }
  v41 = (__int64 *)&v89[v37];
  *v41 = v36;
  v41[1] = v39;
  v43 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 120LL));
  LODWORD(v90) = v90 + 1;
  v42 = (unsigned int)v90;
  if ( (unsigned __int64)(unsigned int)v90 + 1 > HIDWORD(v90) )
  {
    v71 = v43;
    sub_C8D5F0((__int64)&v89, v73, (unsigned int)v90 + 1LL, 0x10u, v36, v39);
    v42 = (unsigned int)v90;
    v43 = _mm_load_si128(&v71);
  }
  v89[v42] = v43;
  v44 = (unsigned int)(v90 + 1);
  LODWORD(v90) = v90 + 1;
  if ( v74.m128i_i8[0] )
  {
    v70 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 160LL));
    if ( v44 + 1 > (unsigned __int64)HIDWORD(v90) )
    {
      v74 = v70;
      sub_C8D5F0((__int64)&v89, v73, v44 + 1, 0x10u, v36, v39);
      v44 = (unsigned int)v90;
      v70 = _mm_load_si128(&v74);
    }
    v89[v44] = v70;
    LODWORD(v44) = v90 + 1;
    LODWORD(v90) = v90 + 1;
  }
  v45 = v28;
  v46 = (__int64 **)v85;
  v47 = sub_33EA9D0(
          v77,
          47,
          (_DWORD)v75,
          v45,
          (_DWORD)v72,
          *(_QWORD *)(a1 + 112),
          (__int64)v89,
          (unsigned int)v44,
          *(unsigned __int16 *)(a1 + 96),
          *(_QWORD *)(a1 + 104));
  v83 = v85;
  v49 = v47;
  v84 = 0x400000000LL;
  if ( (_DWORD)v15 )
  {
    v74.m128i_i64[0] = a3;
    v50 = (__int64 **)v85;
    *((_QWORD *)&v51 + 1) = 0;
    v52 = v15;
    v53 = 4;
    v54 = (unsigned __int64)v6;
    v55 = 0;
    v48 = 0;
    while ( 1 )
    {
      v56 = v54 & 0xFFFFFFFF00000000LL | v55;
      v54 = v56;
      if ( *((_QWORD *)&v51 + 1) + 1LL > v53 )
      {
        v72 = v50;
        sub_C8D5F0(
          (__int64)&v83,
          v50,
          *((_QWORD *)&v51 + 1) + 1LL,
          0x10u,
          *((_QWORD *)&v51 + 1) + 1LL,
          0xFFFFFFFF00000000LL);
        *((_QWORD *)&v51 + 1) = (unsigned int)v84;
        v50 = v72;
        v48 = 0;
      }
      v57 = (__int64 *)&v83[16 * *((_QWORD *)&v51 + 1)];
      ++v55;
      *v57 = v49;
      v57[1] = v56;
      *((_QWORD *)&v51 + 1) = (unsigned int)(v84 + 1);
      LODWORD(v84) = v84 + 1;
      if ( v55 == v52 )
        break;
      v53 = HIDWORD(v84);
    }
    a3 = v74.m128i_i64[0];
    *(_QWORD *)&v51 = v83;
    v15 = v52;
    v46 = v50;
  }
  else
  {
    *((_QWORD *)&v51 + 1) = 0;
    *(_QWORD *)&v51 = v85;
  }
  v59 = sub_33FC220(v77, 156, (_DWORD)v75, v81, v82, v48, v51);
  v60 = *(unsigned int *)(a3 + 8);
  v61 = v58;
  if ( v60 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v77 = v59;
    v78 = v58;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v60 + 1, 0x10u, v59, v58);
    v60 = *(unsigned int *)(a3 + 8);
    v59 = v77;
    v61 = v78;
  }
  v62 = (__int64 *)(*(_QWORD *)a3 + 16 * v60);
  *v62 = v59;
  v62[1] = v61;
  v63 = *(unsigned int *)(a3 + 12);
  v64 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v64;
  if ( v64 + 1 > v63 )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v64 + 1, 0x10u, v59, v61);
    v64 = *(unsigned int *)(a3 + 8);
  }
  v65 = (__int64 *)(*(_QWORD *)a3 + 16 * v64);
  *v65 = v49;
  v65[1] = v15;
  v66 = *(unsigned int *)(a3 + 12);
  v67 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v67;
  if ( v67 + 1 > v66 )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v67 + 1, 0x10u, v59, v61);
    v67 = *(unsigned int *)(a3 + 8);
  }
  v68 = (__int64 *)(*(_QWORD *)a3 + 16 * v67);
  *v68 = v49;
  v69 = (unsigned __int64)v83;
  v68[1] = (unsigned int)(v15 + 1);
  ++*(_DWORD *)(a3 + 8);
  if ( (__int64 **)v69 != v46 )
    _libc_free(v69);
  if ( v89 != v73 )
    _libc_free((unsigned __int64)v89);
  if ( v86 != v76 )
    _libc_free((unsigned __int64)v86);
LABEL_5:
  if ( v79 )
    sub_B91220((__int64)&v79, v79);
}
