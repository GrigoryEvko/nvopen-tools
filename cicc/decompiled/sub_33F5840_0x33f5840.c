// Function: sub_33F5840
// Address: 0x33f5840
//
__m128i *__fastcall sub_33F5840(
        __int64 *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int64 a10,
        __int64 a11,
        const __m128i *a12,
        char a13)
{
  unsigned __int16 v16; // bx
  __int64 v17; // rax
  unsigned __int16 v18; // dx
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // r15
  unsigned __int16 *v22; // rcx
  unsigned int v23; // eax
  __int64 v24; // r8
  char v25; // bl
  __int128 v26; // rax
  __int64 v27; // r10
  __m128i *v28; // r13
  __m128i *v29; // r15
  __int32 v30; // edx
  __int64 v31; // r8
  unsigned int v32; // ecx
  _QWORD *v33; // rax
  int v34; // edx
  __int64 v35; // r9
  unsigned __int64 v36; // r10
  __m128i v37; // xmm0
  __m128i v38; // xmm1
  __m128i v39; // xmm2
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rax
  unsigned __int64 v43; // rbx
  int v44; // r13d
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  unsigned __int64 v47; // rbx
  __int64 v48; // rax
  __int64 v49; // r9
  __int16 v50; // ax
  __int16 v51; // ax
  unsigned __int16 v52; // bx
  __int64 v53; // rdx
  int v54; // eax
  unsigned __int64 v55; // r8
  const __m128i *v56; // rdi
  int v57; // eax
  __int64 v58; // r9
  __int64 v59; // rdx
  unsigned __int64 v60; // r8
  const __m128i *v61; // rbx
  __int64 v62; // r8
  __int64 v63; // rax
  __m128i *v64; // rax
  __int32 v66; // r14d
  __int64 v67; // rcx
  unsigned __int64 v68; // rax
  __int128 v69; // [rsp-20h] [rbp-230h]
  __int128 v70; // [rsp-20h] [rbp-230h]
  __int64 v72; // [rsp+10h] [rbp-200h]
  int v73; // [rsp+10h] [rbp-200h]
  __int64 v74; // [rsp+18h] [rbp-1F8h]
  _QWORD *v75; // [rsp+18h] [rbp-1F8h]
  __int32 v76; // [rsp+20h] [rbp-1F0h]
  unsigned __int16 v77; // [rsp+28h] [rbp-1E8h]
  __int64 v78; // [rsp+28h] [rbp-1E8h]
  unsigned __int8 v79; // [rsp+30h] [rbp-1E0h]
  int v81; // [rsp+38h] [rbp-1D8h]
  int v82; // [rsp+38h] [rbp-1D8h]
  __int64 v83; // [rsp+40h] [rbp-1D0h]
  __int64 v84; // [rsp+40h] [rbp-1D0h]
  __int128 v85; // [rsp+40h] [rbp-1D0h]
  unsigned __int8 *v87; // [rsp+68h] [rbp-1A8h] BYREF
  unsigned __int64 v88[4]; // [rsp+70h] [rbp-1A0h] BYREF
  __m128i v89; // [rsp+90h] [rbp-180h]
  _QWORD *v90; // [rsp+A0h] [rbp-170h]
  int v91; // [rsp+A8h] [rbp-168h]
  __m128i v92; // [rsp+B0h] [rbp-160h]
  __m128i v93; // [rsp+C0h] [rbp-150h]
  __m128i v94[2]; // [rsp+D0h] [rbp-140h] BYREF
  __int16 v95; // [rsp+F0h] [rbp-120h]
  __int64 v96[6]; // [rsp+120h] [rbp-F0h] BYREF
  _BYTE *v97; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v98; // [rsp+158h] [rbp-B8h]
  _BYTE v99[176]; // [rsp+160h] [rbp-B0h] BYREF

  v16 = a10;
  v79 = a13;
  v17 = *(_QWORD *)(a5 + 48) + 16LL * (unsigned int)a6;
  v18 = *(_WORD *)v17;
  v83 = a11;
  v77 = a10;
  v19 = *(_QWORD *)(v17 + 8);
  v20 = v18;
  if ( v18 == (_WORD)a10 )
  {
    v21 = v19;
    if ( (_WORD)a10 || v19 == a11 )
    {
      v22 = (unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * DWORD2(a7));
      v23 = *v22;
      v24 = *((_QWORD *)v22 + 1);
      v84 = a5;
      v25 = a13;
      v97 = 0;
      LODWORD(v98) = 0;
      *(_QWORD *)&v26 = sub_33F17F0(a1, 51, (__int64)&v97, v23, v24);
      v27 = v84;
      if ( v97 )
      {
        v78 = v84;
        v85 = v26;
        sub_B91220((__int64)&v97, (__int64)v97);
        v27 = v78;
        v26 = v85;
      }
      return sub_33F51B0(a1, a2, a3, a4, v27, a6, a7, *((__int64 *)&a7 + 1), v26, a8, a9, v20, v21, a12, 0, 0, v25);
    }
  }
  v72 = 16LL * DWORD2(a7);
  v74 = a7;
  v29 = sub_33ED250((__int64)a1, 1, 0);
  v76 = v30;
  v31 = *(_QWORD *)(*(_QWORD *)(v74 + 48) + v72 + 8);
  v32 = *(unsigned __int16 *)(*(_QWORD *)(v74 + 48) + v72);
  v97 = 0;
  LODWORD(v98) = 0;
  v33 = sub_33F17F0(a1, 51, (__int64)&v97, v32, v31);
  v36 = a5;
  if ( v97 )
  {
    v73 = v34;
    v75 = v33;
    sub_B91220((__int64)&v97, (__int64)v97);
    v36 = a5;
    v34 = v73;
    v33 = v75;
  }
  v90 = v33;
  v37 = _mm_loadu_si128((const __m128i *)&a7);
  v38 = _mm_loadu_si128((const __m128i *)&a8);
  v91 = v34;
  v88[0] = a2;
  v39 = _mm_loadu_si128((const __m128i *)&a9);
  v97 = v99;
  v98 = 0x2000000000LL;
  v88[1] = a3;
  v88[3] = a6;
  v88[2] = v36;
  v89 = v37;
  v92 = v38;
  v93 = v39;
  sub_33C9670((__int64)&v97, 465, (unsigned __int64)v29, v88, 6, v35);
  v42 = v16;
  if ( !v16 )
    v42 = v83;
  v43 = v42;
  v44 = v42;
  v45 = (unsigned int)v98;
  v46 = (unsigned int)v98 + 1LL;
  if ( v46 > HIDWORD(v98) )
  {
    sub_C8D5F0((__int64)&v97, v99, v46, 4u, v40, v41);
    v45 = (unsigned int)v98;
  }
  v47 = HIDWORD(v43);
  *(_DWORD *)&v97[4 * v45] = v44;
  LODWORD(v98) = v98 + 1;
  v48 = (unsigned int)v98;
  if ( (unsigned __int64)(unsigned int)v98 + 1 > HIDWORD(v98) )
  {
    sub_C8D5F0((__int64)&v97, v99, (unsigned int)v98 + 1LL, 4u, v40, v41);
    v48 = (unsigned int)v98;
  }
  *(_DWORD *)&v97[4 * v48] = v47;
  v87 = 0;
  LODWORD(v98) = v98 + 1;
  *((_QWORD *)&v69 + 1) = v83;
  *(_QWORD *)&v69 = v77;
  sub_33CF750(v94, 465, *(_DWORD *)(a4 + 8), &v87, (__int64)v29, v76, v69, (__int64)a12);
  LOBYTE(v50) = 0;
  HIBYTE(v50) = (8 * v79) | 4;
  v51 = v95 & 0xF07F | v50 & 0xF80;
  HIBYTE(v52) = HIBYTE(v51);
  v95 = v51;
  LOBYTE(v52) = v51 & 0x7A;
  if ( v96[0] )
    sub_B91220((__int64)v96, v96[0]);
  if ( v87 )
    sub_B91220((__int64)&v87, (__int64)v87);
  v53 = (unsigned int)v98;
  v54 = v52;
  v55 = (unsigned int)v98 + 1LL;
  if ( v55 > HIDWORD(v98) )
  {
    sub_C8D5F0((__int64)&v97, v99, (unsigned int)v98 + 1LL, 4u, v55, v49);
    v53 = (unsigned int)v98;
    v54 = v52;
  }
  v56 = a12;
  *(_DWORD *)&v97[4 * v53] = v54;
  LODWORD(v98) = v98 + 1;
  v57 = sub_2EAC1E0((__int64)v56);
  v59 = (unsigned int)v98;
  v60 = (unsigned int)v98 + 1LL;
  if ( v60 > HIDWORD(v98) )
  {
    v81 = v57;
    sub_C8D5F0((__int64)&v97, v99, (unsigned int)v98 + 1LL, 4u, v60, v58);
    v59 = (unsigned int)v98;
    v57 = v81;
  }
  v61 = a12;
  *(_DWORD *)&v97[4 * v59] = v57;
  v62 = v61[2].m128i_u16[0];
  LODWORD(v98) = v98 + 1;
  v63 = (unsigned int)v98;
  if ( (unsigned __int64)(unsigned int)v98 + 1 > HIDWORD(v98) )
  {
    v82 = v62;
    sub_C8D5F0((__int64)&v97, v99, (unsigned int)v98 + 1LL, 4u, v62, v58);
    v63 = (unsigned int)v98;
    LODWORD(v62) = v82;
  }
  *(_DWORD *)&v97[4 * v63] = v62;
  LODWORD(v98) = v98 + 1;
  v94[0].m128i_i64[0] = 0;
  v64 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v97, a4, v94[0].m128i_i64);
  v28 = v64;
  if ( v64 )
  {
    sub_2EAC4C0((__m128i *)v64[7].m128i_i64[0], a12);
    goto LABEL_27;
  }
  v28 = (__m128i *)a1[52];
  v66 = *(_DWORD *)(a4 + 8);
  if ( v28 )
  {
    a1[52] = v28->m128i_i64[0];
  }
  else
  {
    v67 = a1[53];
    a1[63] += 120;
    v68 = (v67 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v68 + 120 && v67 )
    {
      a1[53] = v68 + 120;
      if ( !v68 )
        goto LABEL_33;
    }
    else
    {
      v68 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    }
    v28 = (__m128i *)v68;
  }
  *((_QWORD *)&v70 + 1) = v83;
  *(_QWORD *)&v70 = v77;
  sub_33CF750(v28, 465, v66, (unsigned __int8 **)a4, (__int64)v29, v76, v70, (__int64)a12);
  v28[2].m128i_i16[0] = v28[2].m128i_i16[0] & 0xF07F | (v79 << 11) & 0xB80 | 0x400;
LABEL_33:
  sub_33E4EC0((__int64)a1, (__int64)v28, (__int64)v88, 6);
  sub_C657C0(a1 + 65, v28->m128i_i64, (__int64 *)v94[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v28);
LABEL_27:
  if ( v97 != v99 )
    _libc_free((unsigned __int64)v97);
  return v28;
}
