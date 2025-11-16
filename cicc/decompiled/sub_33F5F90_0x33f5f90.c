// Function: sub_33F5F90
// Address: 0x33f5f90
//
__m128i *__fastcall sub_33F5F90(
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
        __int128 a11,
        __int128 a12,
        __int64 a13,
        __int64 a14,
        const __m128i *a15,
        int a16,
        int a17,
        char a18)
{
  unsigned __int16 v21; // r14
  unsigned __int16 *v22; // rax
  __int64 v23; // rax
  __int32 v24; // edx
  __int64 v25; // r9
  __int64 v26; // r11
  unsigned __int64 v27; // r10
  unsigned __int64 v28; // rbx
  __m128i v29; // xmm1
  __m128i v30; // xmm0
  __m128i v31; // xmm2
  __m128i v32; // xmm3
  __m128i v33; // xmm4
  __int64 v34; // rdx
  __int32 v35; // eax
  unsigned __int64 v36; // r14
  __int64 v37; // r9
  __int64 v38; // rdx
  unsigned __int64 v39; // r8
  unsigned __int64 v40; // r14
  __int64 v41; // rdx
  __int64 v42; // rsi
  __int64 v43; // rdi
  __int32 v44; // edx
  __int64 v45; // r9
  __int16 v46; // dx
  int v47; // eax
  __int64 v48; // rdx
  unsigned __int64 v49; // r8
  const __m128i *v50; // rdi
  int v51; // eax
  __int64 v52; // r9
  __int64 v53; // rdx
  unsigned __int64 v54; // r8
  __m128i *v55; // rax
  __m128i *v56; // r14
  __m128i *v58; // rax
  __int32 v59; // r15d
  __int16 v60; // ax
  __int64 v61; // rcx
  unsigned __int64 v62; // rax
  __int128 v63; // [rsp-20h] [rbp-230h]
  __int128 v64; // [rsp-20h] [rbp-230h]
  __int32 v65; // [rsp+10h] [rbp-200h]
  __int32 v66; // [rsp+20h] [rbp-1F0h]
  __int16 v67; // [rsp+20h] [rbp-1F0h]
  __int16 v68; // [rsp+20h] [rbp-1F0h]
  int v69; // [rsp+20h] [rbp-1F0h]
  __int32 v70; // [rsp+20h] [rbp-1F0h]
  int v71; // [rsp+20h] [rbp-1F0h]
  __int16 v72; // [rsp+2Ah] [rbp-1E6h]
  char v73; // [rsp+2Ch] [rbp-1E4h]
  __m128i v74; // [rsp+30h] [rbp-1E0h] BYREF
  int v75; // [rsp+40h] [rbp-1D0h]
  unsigned __int16 v76; // [rsp+46h] [rbp-1CAh]
  __int64 v77; // [rsp+48h] [rbp-1C8h]
  unsigned __int8 *v78; // [rsp+58h] [rbp-1B8h] BYREF
  unsigned __int64 v79[2]; // [rsp+60h] [rbp-1B0h] BYREF
  __m128i v80; // [rsp+70h] [rbp-1A0h]
  __int64 v81; // [rsp+80h] [rbp-190h]
  __int64 v82; // [rsp+88h] [rbp-188h]
  __m128i v83; // [rsp+90h] [rbp-180h]
  __m128i v84; // [rsp+A0h] [rbp-170h]
  __m128i v85; // [rsp+B0h] [rbp-160h]
  __m128i v86; // [rsp+C0h] [rbp-150h]
  __m128i v87[2]; // [rsp+D0h] [rbp-140h] BYREF
  __int16 v88; // [rsp+F0h] [rbp-120h]
  __int64 v89[6]; // [rsp+120h] [rbp-F0h] BYREF
  _BYTE *v90; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v91; // [rsp+158h] [rbp-B8h]
  _BYTE v92[176]; // [rsp+160h] [rbp-B0h] BYREF

  v21 = a13;
  v74.m128i_i64[1] = a6;
  v74.m128i_i64[0] = a5;
  v77 = a14;
  v76 = a13;
  v75 = a17;
  v73 = a18;
  if ( a16 )
  {
    v22 = (unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8);
    v23 = sub_33E5110(a1, *v22, *((_QWORD *)v22 + 1), 1, 0);
    v25 = a7;
    v26 = a8;
    v27 = a2;
    v28 = v23;
  }
  else
  {
    v58 = sub_33ED250((__int64)a1, 1, 0);
    v26 = a8;
    v25 = a7;
    v28 = (unsigned __int64)v58;
    v27 = a2;
  }
  v66 = v24;
  v29 = _mm_loadu_si128((const __m128i *)&a9);
  v30 = _mm_load_si128(&v74);
  v90 = v92;
  v74.m128i_i64[0] = (__int64)v92;
  v31 = _mm_loadu_si128((const __m128i *)&a10);
  v79[1] = a3;
  v32 = _mm_loadu_si128((const __m128i *)&a11);
  v91 = 0x2000000000LL;
  v33 = _mm_loadu_si128((const __m128i *)&a12);
  v81 = v25;
  v79[0] = v27;
  v82 = v26;
  v80 = v30;
  v83 = v29;
  v84 = v31;
  v85 = v32;
  v86 = v33;
  sub_33C9670((__int64)&v90, 466, v28, v79, 7, v25);
  v34 = v21;
  if ( !v21 )
    v34 = v77;
  v35 = v66;
  v36 = v34;
  v37 = (unsigned int)v34;
  v38 = (unsigned int)v91;
  v39 = (unsigned int)v91 + 1LL;
  if ( v39 > HIDWORD(v91) )
  {
    sub_C8D5F0((__int64)&v90, (const void *)v74.m128i_i64[0], (unsigned int)v91 + 1LL, 4u, v39, v37);
    v38 = (unsigned int)v91;
    v35 = v66;
    v37 = (unsigned int)v36;
  }
  v40 = HIDWORD(v36);
  *(_DWORD *)&v90[4 * v38] = v37;
  LODWORD(v91) = v91 + 1;
  v41 = (unsigned int)v91;
  if ( (unsigned __int64)(unsigned int)v91 + 1 > HIDWORD(v91) )
  {
    v70 = v35;
    sub_C8D5F0((__int64)&v90, (const void *)v74.m128i_i64[0], (unsigned int)v91 + 1LL, 4u, (unsigned int)v91 + 1LL, v37);
    v41 = (unsigned int)v91;
    v35 = v70;
  }
  v42 = v76;
  v43 = v77;
  *(_DWORD *)&v90[4 * v41] = v40;
  v44 = *(_DWORD *)(a4 + 8);
  *((_QWORD *)&v63 + 1) = v43;
  *(_QWORD *)&v63 = v42;
  LODWORD(v91) = v91 + 1;
  v65 = v35;
  v78 = 0;
  sub_33CF750(v87, 466, v44, &v78, v28, v35, v63, (__int64)a15);
  v72 = a16 & 7;
  v46 = (v72 << 7) | v88 & 0xFC7F;
  LOBYTE(v88) = ((a16 & 7) << 7) | v88 & 0x7F;
  HIBYTE(v88) = (8 * (v73 & 1)) | (4 * (v75 & 1)) | HIBYTE(v46) & 0xF3;
  LOWORD(v47) = v88 & 0xFFFA;
  if ( v89[0] )
  {
    v67 = v88 & 0xFFFA;
    sub_B91220((__int64)v89, v89[0]);
    LOWORD(v47) = v67;
  }
  if ( v78 )
  {
    v68 = v47;
    sub_B91220((__int64)&v78, (__int64)v78);
    LOWORD(v47) = v68;
  }
  v48 = (unsigned int)v91;
  v47 = (unsigned __int16)v47;
  v49 = (unsigned int)v91 + 1LL;
  if ( v49 > HIDWORD(v91) )
  {
    v71 = (unsigned __int16)v47;
    sub_C8D5F0((__int64)&v90, (const void *)v74.m128i_i64[0], (unsigned int)v91 + 1LL, 4u, v49, v45);
    v48 = (unsigned int)v91;
    v47 = v71;
  }
  v50 = a15;
  *(_DWORD *)&v90[4 * v48] = v47;
  LODWORD(v91) = v91 + 1;
  v51 = sub_2EAC1E0((__int64)v50);
  v53 = (unsigned int)v91;
  v54 = (unsigned int)v91 + 1LL;
  if ( v54 > HIDWORD(v91) )
  {
    v69 = v51;
    sub_C8D5F0((__int64)&v90, (const void *)v74.m128i_i64[0], (unsigned int)v91 + 1LL, 4u, v54, v52);
    v53 = (unsigned int)v91;
    v51 = v69;
  }
  *(_DWORD *)&v90[4 * v53] = v51;
  LODWORD(v91) = v91 + 1;
  v87[0].m128i_i64[0] = 0;
  v55 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v90, a4, v87[0].m128i_i64);
  v56 = v55;
  if ( v55 )
  {
    sub_2EAC4C0((__m128i *)v55[7].m128i_i64[0], a15);
    goto LABEL_19;
  }
  v56 = (__m128i *)a1[52];
  v59 = *(_DWORD *)(a4 + 8);
  if ( v56 )
  {
    a1[52] = v56->m128i_i64[0];
  }
  else
  {
    v61 = a1[53];
    a1[63] += 120;
    v62 = (v61 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v62 + 120 && v61 )
    {
      a1[53] = v62 + 120;
      if ( !v62 )
        goto LABEL_26;
    }
    else
    {
      v62 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    }
    v56 = (__m128i *)v62;
  }
  *((_QWORD *)&v64 + 1) = v77;
  *(_QWORD *)&v64 = v76;
  sub_33CF750(v56, 466, v59, (unsigned __int8 **)a4, v28, v65, v64, (__int64)a15);
  v60 = v56[2].m128i_i16[0] & 0xFC7F | (v72 << 7);
  v56[2].m128i_i16[0] = v60;
  v56[2].m128i_i8[1] = HIBYTE(v60) & 0xF3 | (4 * (v75 & 1)) | (8 * (v73 & 1));
LABEL_26:
  sub_33E4EC0((__int64)a1, (__int64)v56, (__int64)v79, 7);
  sub_C657C0(a1 + 65, v56->m128i_i64, (__int64 *)v87[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v56);
LABEL_19:
  if ( v90 != (_BYTE *)v74.m128i_i64[0] )
    _libc_free((unsigned __int64)v90);
  return v56;
}
