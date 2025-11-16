// Function: sub_33F51B0
// Address: 0x33f51b0
//
__m128i *__fastcall sub_33F51B0(
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
        __int64 a12,
        __int64 a13,
        const __m128i *a14,
        int a15,
        int a16,
        char a17)
{
  unsigned __int16 v20; // r14
  unsigned __int16 *v21; // rax
  __int64 v22; // rax
  __int32 v23; // edx
  __int64 v24; // r9
  __int64 v25; // r11
  unsigned __int64 v26; // r10
  unsigned __int64 v27; // rbx
  __m128i v28; // xmm1
  __m128i v29; // xmm0
  __m128i v30; // xmm2
  __m128i v31; // xmm3
  __int64 v32; // rdx
  __int32 v33; // eax
  unsigned __int64 v34; // r14
  __int64 v35; // r9
  __int64 v36; // rdx
  unsigned __int64 v37; // r8
  unsigned __int64 v38; // r14
  __int64 v39; // rdx
  __int64 v40; // rsi
  __int64 v41; // rdi
  __int32 v42; // edx
  __int64 v43; // r9
  __int16 v44; // dx
  int v45; // eax
  __int64 v46; // rdx
  unsigned __int64 v47; // r8
  const __m128i *v48; // rdi
  int v49; // eax
  __int64 v50; // r9
  __int64 v51; // rdx
  unsigned __int64 v52; // r8
  const __m128i *v53; // rdi
  __int64 v54; // r8
  __int64 v55; // rax
  __m128i *v56; // rax
  __m128i *v57; // r14
  __m128i *v59; // rax
  __int32 v60; // r15d
  __int16 v61; // ax
  __int64 v62; // rcx
  unsigned __int64 v63; // rax
  __int128 v64; // [rsp-20h] [rbp-220h]
  __int128 v65; // [rsp-20h] [rbp-220h]
  __int32 v66; // [rsp+10h] [rbp-1F0h]
  __int32 v67; // [rsp+20h] [rbp-1E0h]
  __int16 v68; // [rsp+20h] [rbp-1E0h]
  __int16 v69; // [rsp+20h] [rbp-1E0h]
  int v70; // [rsp+20h] [rbp-1E0h]
  __int32 v71; // [rsp+20h] [rbp-1E0h]
  int v72; // [rsp+20h] [rbp-1E0h]
  int v73; // [rsp+20h] [rbp-1E0h]
  __int16 v74; // [rsp+2Ah] [rbp-1D6h]
  char v75; // [rsp+2Ch] [rbp-1D4h]
  __m128i v76; // [rsp+30h] [rbp-1D0h] BYREF
  int v77; // [rsp+40h] [rbp-1C0h]
  unsigned __int16 v78; // [rsp+46h] [rbp-1BAh]
  __int64 v79; // [rsp+48h] [rbp-1B8h]
  unsigned __int8 *v80; // [rsp+58h] [rbp-1A8h] BYREF
  unsigned __int64 v81[2]; // [rsp+60h] [rbp-1A0h] BYREF
  __m128i v82; // [rsp+70h] [rbp-190h]
  __int64 v83; // [rsp+80h] [rbp-180h]
  __int64 v84; // [rsp+88h] [rbp-178h]
  __m128i v85; // [rsp+90h] [rbp-170h]
  __m128i v86; // [rsp+A0h] [rbp-160h]
  __m128i v87; // [rsp+B0h] [rbp-150h]
  __m128i v88[2]; // [rsp+C0h] [rbp-140h] BYREF
  __int16 v89; // [rsp+E0h] [rbp-120h]
  __int64 v90[6]; // [rsp+110h] [rbp-F0h] BYREF
  _BYTE *v91; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v92; // [rsp+148h] [rbp-B8h]
  _BYTE v93[176]; // [rsp+150h] [rbp-B0h] BYREF

  v20 = a12;
  v76.m128i_i64[1] = a6;
  v76.m128i_i64[0] = a5;
  v79 = a13;
  v78 = a12;
  v77 = a16;
  v75 = a17;
  if ( a15 )
  {
    v21 = (unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8);
    v22 = sub_33E5110(a1, *v21, *((_QWORD *)v21 + 1), 1, 0);
    v24 = a7;
    v25 = a8;
    v26 = a2;
    v27 = v22;
  }
  else
  {
    v59 = sub_33ED250((__int64)a1, 1, 0);
    v25 = a8;
    v24 = a7;
    v27 = (unsigned __int64)v59;
    v26 = a2;
  }
  v67 = v23;
  v28 = _mm_loadu_si128((const __m128i *)&a9);
  v29 = _mm_load_si128(&v76);
  v91 = v93;
  v76.m128i_i64[0] = (__int64)v93;
  v30 = _mm_loadu_si128((const __m128i *)&a10);
  v81[1] = a3;
  v31 = _mm_loadu_si128((const __m128i *)&a11);
  v92 = 0x2000000000LL;
  v83 = v24;
  v81[0] = v26;
  v84 = v25;
  v82 = v29;
  v85 = v28;
  v86 = v30;
  v87 = v31;
  sub_33C9670((__int64)&v91, 465, v27, v81, 6, v24);
  v32 = v20;
  if ( !v20 )
    v32 = v79;
  v33 = v67;
  v34 = v32;
  v35 = (unsigned int)v32;
  v36 = (unsigned int)v92;
  v37 = (unsigned int)v92 + 1LL;
  if ( v37 > HIDWORD(v92) )
  {
    sub_C8D5F0((__int64)&v91, (const void *)v76.m128i_i64[0], (unsigned int)v92 + 1LL, 4u, v37, v35);
    v36 = (unsigned int)v92;
    v33 = v67;
    v35 = (unsigned int)v34;
  }
  v38 = HIDWORD(v34);
  *(_DWORD *)&v91[4 * v36] = v35;
  LODWORD(v92) = v92 + 1;
  v39 = (unsigned int)v92;
  if ( (unsigned __int64)(unsigned int)v92 + 1 > HIDWORD(v92) )
  {
    v71 = v33;
    sub_C8D5F0((__int64)&v91, (const void *)v76.m128i_i64[0], (unsigned int)v92 + 1LL, 4u, (unsigned int)v92 + 1LL, v35);
    v39 = (unsigned int)v92;
    v33 = v71;
  }
  v40 = v78;
  v41 = v79;
  *(_DWORD *)&v91[4 * v39] = v38;
  v42 = *(_DWORD *)(a4 + 8);
  *((_QWORD *)&v64 + 1) = v41;
  *(_QWORD *)&v64 = v40;
  LODWORD(v92) = v92 + 1;
  v66 = v33;
  v80 = 0;
  sub_33CF750(v88, 465, v42, &v80, v27, v33, v64, (__int64)a14);
  v74 = a15 & 7;
  v44 = (v74 << 7) | v89 & 0xFC7F;
  LOBYTE(v89) = ((a15 & 7) << 7) | v89 & 0x7F;
  HIBYTE(v89) = (8 * (v75 & 1)) | (4 * (v77 & 1)) | HIBYTE(v44) & 0xF3;
  LOWORD(v45) = v89 & 0xFFFA;
  if ( v90[0] )
  {
    v68 = v89 & 0xFFFA;
    sub_B91220((__int64)v90, v90[0]);
    LOWORD(v45) = v68;
  }
  if ( v80 )
  {
    v69 = v45;
    sub_B91220((__int64)&v80, (__int64)v80);
    LOWORD(v45) = v69;
  }
  v46 = (unsigned int)v92;
  v45 = (unsigned __int16)v45;
  v47 = (unsigned int)v92 + 1LL;
  if ( v47 > HIDWORD(v92) )
  {
    v72 = (unsigned __int16)v45;
    sub_C8D5F0((__int64)&v91, (const void *)v76.m128i_i64[0], (unsigned int)v92 + 1LL, 4u, v47, v43);
    v46 = (unsigned int)v92;
    v45 = v72;
  }
  v48 = a14;
  *(_DWORD *)&v91[4 * v46] = v45;
  LODWORD(v92) = v92 + 1;
  v49 = sub_2EAC1E0((__int64)v48);
  v51 = (unsigned int)v92;
  v52 = (unsigned int)v92 + 1LL;
  if ( v52 > HIDWORD(v92) )
  {
    v73 = v49;
    sub_C8D5F0((__int64)&v91, (const void *)v76.m128i_i64[0], (unsigned int)v92 + 1LL, 4u, v52, v50);
    v51 = (unsigned int)v92;
    v49 = v73;
  }
  v53 = a14;
  *(_DWORD *)&v91[4 * v51] = v49;
  v54 = v53[2].m128i_u16[0];
  LODWORD(v92) = v92 + 1;
  v55 = (unsigned int)v92;
  if ( (unsigned __int64)(unsigned int)v92 + 1 > HIDWORD(v92) )
  {
    v70 = v54;
    sub_C8D5F0((__int64)&v91, (const void *)v76.m128i_i64[0], (unsigned int)v92 + 1LL, 4u, v54, v50);
    v55 = (unsigned int)v92;
    LODWORD(v54) = v70;
  }
  *(_DWORD *)&v91[4 * v55] = v54;
  LODWORD(v92) = v92 + 1;
  v88[0].m128i_i64[0] = 0;
  v56 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v91, a4, v88[0].m128i_i64);
  v57 = v56;
  if ( v56 )
  {
    sub_2EAC4C0((__m128i *)v56[7].m128i_i64[0], a14);
    goto LABEL_21;
  }
  v57 = (__m128i *)a1[52];
  v60 = *(_DWORD *)(a4 + 8);
  if ( v57 )
  {
    a1[52] = v57->m128i_i64[0];
  }
  else
  {
    v62 = a1[53];
    a1[63] += 120;
    v63 = (v62 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v63 + 120 && v62 )
    {
      a1[53] = v63 + 120;
      if ( !v63 )
        goto LABEL_28;
    }
    else
    {
      v63 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    }
    v57 = (__m128i *)v63;
  }
  *((_QWORD *)&v65 + 1) = v79;
  *(_QWORD *)&v65 = v78;
  sub_33CF750(v57, 465, v60, (unsigned __int8 **)a4, v27, v66, v65, (__int64)a14);
  v61 = v57[2].m128i_i16[0] & 0xFC7F | (v74 << 7);
  v57[2].m128i_i16[0] = v61;
  v57[2].m128i_i8[1] = HIBYTE(v61) & 0xF3 | (4 * (v77 & 1)) | (8 * (v75 & 1));
LABEL_28:
  sub_33E4EC0((__int64)a1, (__int64)v57, (__int64)v81, 6);
  sub_C657C0(a1 + 65, v57->m128i_i64, (__int64 *)v88[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v57);
LABEL_21:
  if ( v91 != (_BYTE *)v76.m128i_i64[0] )
    _libc_free((unsigned __int64)v91);
  return v57;
}
