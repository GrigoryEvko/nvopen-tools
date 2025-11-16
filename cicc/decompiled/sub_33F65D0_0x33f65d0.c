// Function: sub_33F65D0
// Address: 0x33f65d0
//
__m128i *__fastcall sub_33F65D0(
        __int64 *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int128 a9,
        __int128 a10,
        __int64 a11,
        __int64 a12,
        const __m128i *a13,
        int a14,
        int a15,
        char a16)
{
  unsigned __int16 v19; // r14
  unsigned __int16 *v20; // rax
  __int64 v21; // rax
  __int32 v22; // edx
  __int64 v23; // r9
  __int64 v24; // r11
  unsigned __int64 v25; // r10
  unsigned __int64 v26; // rbx
  __m128i v27; // xmm1
  __m128i v28; // xmm2
  __m128i v29; // xmm0
  __int64 v30; // rdx
  __int32 v31; // eax
  unsigned __int64 v32; // r14
  __int64 v33; // r9
  __int64 v34; // rdx
  unsigned __int64 v35; // r8
  unsigned __int64 v36; // r14
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // rdi
  __int32 v40; // edx
  __int64 v41; // r9
  __int16 v42; // dx
  int v43; // eax
  __int64 v44; // rdx
  unsigned __int64 v45; // r8
  const __m128i *v46; // rdi
  int v47; // eax
  __int64 v48; // r9
  __int64 v49; // rdx
  unsigned __int64 v50; // r8
  const __m128i *v51; // rdi
  __int64 v52; // r8
  __int64 v53; // rax
  __m128i *v54; // rax
  __m128i *v55; // r14
  __m128i *v57; // rax
  __int32 v58; // r15d
  __int16 v59; // ax
  __int64 v60; // rcx
  unsigned __int64 v61; // rax
  __int128 v62; // [rsp-20h] [rbp-210h]
  __int128 v63; // [rsp-20h] [rbp-210h]
  __int32 v64; // [rsp+10h] [rbp-1E0h]
  __int32 v65; // [rsp+20h] [rbp-1D0h]
  __int16 v66; // [rsp+20h] [rbp-1D0h]
  __int16 v67; // [rsp+20h] [rbp-1D0h]
  int v68; // [rsp+20h] [rbp-1D0h]
  __int32 v69; // [rsp+20h] [rbp-1D0h]
  int v70; // [rsp+20h] [rbp-1D0h]
  int v71; // [rsp+20h] [rbp-1D0h]
  __int16 v72; // [rsp+2Ah] [rbp-1C6h]
  char v73; // [rsp+2Ch] [rbp-1C4h]
  __m128i v74; // [rsp+30h] [rbp-1C0h] BYREF
  int v75; // [rsp+40h] [rbp-1B0h]
  unsigned __int16 v76; // [rsp+46h] [rbp-1AAh]
  __int64 v77; // [rsp+48h] [rbp-1A8h]
  unsigned __int8 *v78; // [rsp+58h] [rbp-198h] BYREF
  unsigned __int64 v79[2]; // [rsp+60h] [rbp-190h] BYREF
  __m128i v80; // [rsp+70h] [rbp-180h]
  __int64 v81; // [rsp+80h] [rbp-170h]
  __int64 v82; // [rsp+88h] [rbp-168h]
  __m128i v83; // [rsp+90h] [rbp-160h]
  __m128i v84; // [rsp+A0h] [rbp-150h]
  __m128i v85[2]; // [rsp+B0h] [rbp-140h] BYREF
  __int16 v86; // [rsp+D0h] [rbp-120h]
  __int64 v87[6]; // [rsp+100h] [rbp-F0h] BYREF
  _BYTE *v88; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v89; // [rsp+138h] [rbp-B8h]
  _BYTE v90[176]; // [rsp+140h] [rbp-B0h] BYREF

  v19 = a11;
  v74.m128i_i64[1] = a6;
  v74.m128i_i64[0] = a5;
  v77 = a12;
  v76 = a11;
  v75 = a15;
  v73 = a16;
  if ( a14 )
  {
    v20 = (unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8);
    v21 = sub_33E5110(a1, *v20, *((_QWORD *)v20 + 1), 1, 0);
    v23 = a7;
    v24 = a8;
    v25 = a2;
    v26 = v21;
  }
  else
  {
    v57 = sub_33ED250((__int64)a1, 1, 0);
    v24 = a8;
    v23 = a7;
    v26 = (unsigned __int64)v57;
    v25 = a2;
  }
  v65 = v22;
  v27 = _mm_loadu_si128((const __m128i *)&a9);
  v28 = _mm_loadu_si128((const __m128i *)&a10);
  v29 = _mm_load_si128(&v74);
  v88 = v90;
  v74.m128i_i64[0] = (__int64)v90;
  v79[1] = a3;
  v89 = 0x2000000000LL;
  v81 = v23;
  v79[0] = v25;
  v82 = v24;
  v80 = v29;
  v83 = v27;
  v84 = v28;
  sub_33C9670((__int64)&v88, 363, v26, v79, 5, v23);
  v30 = v19;
  if ( !v19 )
    v30 = v77;
  v31 = v65;
  v32 = v30;
  v33 = (unsigned int)v30;
  v34 = (unsigned int)v89;
  v35 = (unsigned int)v89 + 1LL;
  if ( v35 > HIDWORD(v89) )
  {
    sub_C8D5F0((__int64)&v88, (const void *)v74.m128i_i64[0], (unsigned int)v89 + 1LL, 4u, v35, v33);
    v34 = (unsigned int)v89;
    v31 = v65;
    v33 = (unsigned int)v32;
  }
  v36 = HIDWORD(v32);
  *(_DWORD *)&v88[4 * v34] = v33;
  LODWORD(v89) = v89 + 1;
  v37 = (unsigned int)v89;
  if ( (unsigned __int64)(unsigned int)v89 + 1 > HIDWORD(v89) )
  {
    v69 = v31;
    sub_C8D5F0((__int64)&v88, (const void *)v74.m128i_i64[0], (unsigned int)v89 + 1LL, 4u, (unsigned int)v89 + 1LL, v33);
    v37 = (unsigned int)v89;
    v31 = v69;
  }
  v38 = v76;
  v39 = v77;
  *(_DWORD *)&v88[4 * v37] = v36;
  v40 = *(_DWORD *)(a4 + 8);
  *((_QWORD *)&v62 + 1) = v39;
  *(_QWORD *)&v62 = v38;
  LODWORD(v89) = v89 + 1;
  v64 = v31;
  v78 = 0;
  sub_33CF750(v85, 363, v40, &v78, v26, v31, v62, (__int64)a13);
  v72 = a14 & 7;
  v42 = (v72 << 7) | v86 & 0xFC7F;
  LOBYTE(v86) = ((a14 & 7) << 7) | v86 & 0x7F;
  HIBYTE(v86) = (8 * (v73 & 1)) | (4 * (v75 & 1)) | HIBYTE(v42) & 0xF3;
  LOWORD(v43) = v86 & 0xFFFA;
  if ( v87[0] )
  {
    v66 = v86 & 0xFFFA;
    sub_B91220((__int64)v87, v87[0]);
    LOWORD(v43) = v66;
  }
  if ( v78 )
  {
    v67 = v43;
    sub_B91220((__int64)&v78, (__int64)v78);
    LOWORD(v43) = v67;
  }
  v44 = (unsigned int)v89;
  v43 = (unsigned __int16)v43;
  v45 = (unsigned int)v89 + 1LL;
  if ( v45 > HIDWORD(v89) )
  {
    v70 = (unsigned __int16)v43;
    sub_C8D5F0((__int64)&v88, (const void *)v74.m128i_i64[0], (unsigned int)v89 + 1LL, 4u, v45, v41);
    v44 = (unsigned int)v89;
    v43 = v70;
  }
  v46 = a13;
  *(_DWORD *)&v88[4 * v44] = v43;
  LODWORD(v89) = v89 + 1;
  v47 = sub_2EAC1E0((__int64)v46);
  v49 = (unsigned int)v89;
  v50 = (unsigned int)v89 + 1LL;
  if ( v50 > HIDWORD(v89) )
  {
    v71 = v47;
    sub_C8D5F0((__int64)&v88, (const void *)v74.m128i_i64[0], (unsigned int)v89 + 1LL, 4u, v50, v48);
    v49 = (unsigned int)v89;
    v47 = v71;
  }
  v51 = a13;
  *(_DWORD *)&v88[4 * v49] = v47;
  v52 = v51[2].m128i_u16[0];
  LODWORD(v89) = v89 + 1;
  v53 = (unsigned int)v89;
  if ( (unsigned __int64)(unsigned int)v89 + 1 > HIDWORD(v89) )
  {
    v68 = v52;
    sub_C8D5F0((__int64)&v88, (const void *)v74.m128i_i64[0], (unsigned int)v89 + 1LL, 4u, v52, v48);
    v53 = (unsigned int)v89;
    LODWORD(v52) = v68;
  }
  *(_DWORD *)&v88[4 * v53] = v52;
  LODWORD(v89) = v89 + 1;
  v85[0].m128i_i64[0] = 0;
  v54 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v88, a4, v85[0].m128i_i64);
  v55 = v54;
  if ( v54 )
  {
    sub_2EAC4C0((__m128i *)v54[7].m128i_i64[0], a13);
    goto LABEL_21;
  }
  v55 = (__m128i *)a1[52];
  v58 = *(_DWORD *)(a4 + 8);
  if ( v55 )
  {
    a1[52] = v55->m128i_i64[0];
  }
  else
  {
    v60 = a1[53];
    a1[63] += 120;
    v61 = (v60 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v61 + 120 && v60 )
    {
      a1[53] = v61 + 120;
      if ( !v61 )
        goto LABEL_28;
    }
    else
    {
      v61 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    }
    v55 = (__m128i *)v61;
  }
  *((_QWORD *)&v63 + 1) = v77;
  *(_QWORD *)&v63 = v76;
  sub_33CF750(v55, 363, v58, (unsigned __int8 **)a4, v26, v64, v63, (__int64)a13);
  v59 = v55[2].m128i_i16[0] & 0xFC7F | (v72 << 7);
  v55[2].m128i_i16[0] = v59;
  v55[2].m128i_i8[1] = HIBYTE(v59) & 0xF3 | (4 * (v75 & 1)) | (8 * (v73 & 1));
LABEL_28:
  sub_33E4EC0((__int64)a1, (__int64)v55, (__int64)v79, 5);
  sub_C657C0(a1 + 65, v55->m128i_i64, (__int64 *)v85[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v55);
LABEL_21:
  if ( v88 != (_BYTE *)v74.m128i_i64[0] )
    _libc_free((unsigned __int64)v88);
  return v55;
}
