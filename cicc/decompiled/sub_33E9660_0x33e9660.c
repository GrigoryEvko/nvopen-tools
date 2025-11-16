// Function: sub_33E9660
// Address: 0x33e9660
//
__m128i *__fastcall sub_33E9660(
        __int64 *a1,
        int a2,
        char a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        __int64 a9,
        __int128 a10,
        __int128 a11,
        __int128 a12,
        __int64 a13,
        __int64 a14,
        const __m128i *a15,
        char a16)
{
  unsigned __int16 v18; // r14
  __int64 v19; // r15
  unsigned __int16 *v20; // rax
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  __m128i v23; // xmm0
  __m128i v24; // xmm1
  __m128i v25; // xmm2
  __m128i v26; // xmm3
  __int64 v27; // rax
  __int64 v28; // r9
  unsigned __int64 v29; // r14
  __int64 v30; // r8
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // r14
  __int64 v34; // rax
  __int32 v35; // edx
  __int64 v36; // r9
  __int16 v37; // bx
  char v38; // cl
  unsigned __int16 v39; // bx
  __int64 v40; // rdx
  int v41; // eax
  unsigned __int64 v42; // r8
  const __m128i *v43; // rdi
  int v44; // eax
  __int64 v45; // r9
  __int64 v46; // rdx
  unsigned __int64 v47; // r8
  const __m128i *v48; // rdi
  __int64 v49; // r8
  __int64 v50; // rax
  __m128i *v51; // rax
  __m128i *v52; // r14
  __int32 v54; // r15d
  __int16 v55; // ax
  __int64 v56; // rcx
  unsigned __int64 v57; // rax
  __int128 v58; // [rsp-20h] [rbp-200h]
  __int128 v59; // [rsp-20h] [rbp-200h]
  __int32 v60; // [rsp+0h] [rbp-1E0h]
  __int64 v61; // [rsp+10h] [rbp-1D0h]
  int v62; // [rsp+10h] [rbp-1D0h]
  int v63; // [rsp+10h] [rbp-1D0h]
  int v64; // [rsp+10h] [rbp-1D0h]
  __int16 v65; // [rsp+1Ah] [rbp-1C6h]
  char v67; // [rsp+1Ch] [rbp-1C4h]
  char v68; // [rsp+20h] [rbp-1C0h]
  unsigned __int16 v69; // [rsp+26h] [rbp-1BAh]
  __int64 v70; // [rsp+28h] [rbp-1B8h]
  __int64 v71; // [rsp+30h] [rbp-1B0h]
  __int64 v72; // [rsp+38h] [rbp-1A8h]
  unsigned __int8 *v73; // [rsp+48h] [rbp-198h] BYREF
  __m128i v74; // [rsp+50h] [rbp-190h] BYREF
  __int64 v75; // [rsp+60h] [rbp-180h]
  __int64 v76; // [rsp+68h] [rbp-178h]
  __m128i v77; // [rsp+70h] [rbp-170h]
  __m128i v78; // [rsp+80h] [rbp-160h]
  __m128i v79; // [rsp+90h] [rbp-150h]
  __m128i v80[2]; // [rsp+A0h] [rbp-140h] BYREF
  __int16 v81; // [rsp+C0h] [rbp-120h]
  __int64 v82[6]; // [rsp+F0h] [rbp-F0h] BYREF
  _BYTE *v83; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v84; // [rsp+128h] [rbp-B8h]
  _BYTE v85[176]; // [rsp+130h] [rbp-B0h] BYREF

  v18 = a13;
  v19 = a9;
  v71 = a14;
  v69 = a13;
  v68 = a16;
  v70 = a8;
  if ( a2 )
  {
    v20 = (unsigned __int16 *)(*(_QWORD *)(a8 + 48) + 16LL * (unsigned int)a9);
    v21 = sub_33E5B50(a1, a4, a5, *v20, *((_QWORD *)v20 + 1), a6, 1, 0);
  }
  else
  {
    v21 = sub_33E5110(a1, a4, a5, 1, 0);
  }
  v72 = v21;
  v23 = _mm_loadu_si128((const __m128i *)&a7);
  v24 = _mm_loadu_si128((const __m128i *)&a10);
  v25 = _mm_loadu_si128((const __m128i *)&a11);
  v26 = _mm_loadu_si128((const __m128i *)&a12);
  v83 = v85;
  v76 = v19;
  v84 = 0x2000000000LL;
  v61 = v22;
  v75 = v70;
  v74 = v23;
  v77 = v24;
  v78 = v25;
  v79 = v26;
  sub_33C9670((__int64)&v83, 468, v21, (unsigned __int64 *)&v74, 5, v22);
  v27 = v18;
  if ( !v18 )
    v27 = v71;
  v28 = v61;
  v29 = v27;
  v30 = (unsigned int)v27;
  v31 = (unsigned int)v84;
  v32 = (unsigned int)v84 + 1LL;
  if ( v32 > HIDWORD(v84) )
  {
    sub_C8D5F0((__int64)&v83, v85, v32, 4u, v30, v61);
    v31 = (unsigned int)v84;
    v28 = v61;
    v30 = (unsigned int)v29;
  }
  v33 = HIDWORD(v29);
  *(_DWORD *)&v83[4 * v31] = v30;
  LODWORD(v84) = v84 + 1;
  v34 = (unsigned int)v84;
  if ( (unsigned __int64)(unsigned int)v84 + 1 > HIDWORD(v84) )
  {
    v63 = v28;
    sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 4u, v30, v28);
    v34 = (unsigned int)v84;
    LODWORD(v28) = v63;
  }
  v60 = v28;
  *(_DWORD *)&v83[4 * v34] = v33;
  *((_QWORD *)&v58 + 1) = v71;
  *(_QWORD *)&v58 = v69;
  v35 = *(_DWORD *)(a6 + 8);
  LODWORD(v84) = v84 + 1;
  v73 = 0;
  sub_33CF750(v80, 468, v35, &v73, v72, v28, v58, (__int64)a15);
  v65 = a2 & 7;
  v37 = v81 & 0xFC7F | (v65 << 7);
  LOBYTE(v81) = v81 & 0x7F | ((_BYTE)v65 << 7);
  v38 = 4 * (a3 & 3);
  v67 = a3 & 3;
  HIBYTE(v81) = (16 * (v68 & 1)) | v38 | HIBYTE(v37) & 0xE3;
  v39 = v81 & 0xFFFA;
  if ( v82[0] )
    sub_B91220((__int64)v82, v82[0]);
  if ( v73 )
    sub_B91220((__int64)&v73, (__int64)v73);
  v40 = (unsigned int)v84;
  v41 = v39;
  v42 = (unsigned int)v84 + 1LL;
  if ( v42 > HIDWORD(v84) )
  {
    sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 4u, v42, v36);
    v40 = (unsigned int)v84;
    v41 = v39;
  }
  v43 = a15;
  *(_DWORD *)&v83[4 * v40] = v41;
  LODWORD(v84) = v84 + 1;
  v44 = sub_2EAC1E0((__int64)v43);
  v46 = (unsigned int)v84;
  v47 = (unsigned int)v84 + 1LL;
  if ( v47 > HIDWORD(v84) )
  {
    v64 = v44;
    sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 4u, v47, v45);
    v46 = (unsigned int)v84;
    v44 = v64;
  }
  v48 = a15;
  *(_DWORD *)&v83[4 * v46] = v44;
  v49 = v48[2].m128i_u16[0];
  LODWORD(v84) = v84 + 1;
  v50 = (unsigned int)v84;
  if ( (unsigned __int64)(unsigned int)v84 + 1 > HIDWORD(v84) )
  {
    v62 = v49;
    sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 4u, v49, v45);
    v50 = (unsigned int)v84;
    LODWORD(v49) = v62;
  }
  *(_DWORD *)&v83[4 * v50] = v49;
  LODWORD(v84) = v84 + 1;
  v80[0].m128i_i64[0] = 0;
  v51 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v83, a6, v80[0].m128i_i64);
  v52 = v51;
  if ( v51 )
  {
    sub_2EAC4C0((__m128i *)v51[7].m128i_i64[0], a15);
    goto LABEL_21;
  }
  v52 = (__m128i *)a1[52];
  v54 = *(_DWORD *)(a6 + 8);
  if ( v52 )
  {
    a1[52] = v52->m128i_i64[0];
  }
  else
  {
    v56 = a1[53];
    a1[63] += 120;
    v57 = (v56 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v57 + 120 && v56 )
    {
      a1[53] = v57 + 120;
      if ( !v57 )
        goto LABEL_28;
    }
    else
    {
      v57 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    }
    v52 = (__m128i *)v57;
  }
  *((_QWORD *)&v59 + 1) = v71;
  *(_QWORD *)&v59 = v69;
  sub_33CF750(v52, 468, v54, (unsigned __int8 **)a6, v72, v60, v59, (__int64)a15);
  v55 = v52[2].m128i_i16[0] & 0xFC7F | (v65 << 7);
  v52[2].m128i_i16[0] = v55;
  v52[2].m128i_i8[1] = HIBYTE(v55) & 0xE3 | (4 * v67) | (16 * (v68 & 1));
LABEL_28:
  sub_33E4EC0((__int64)a1, (__int64)v52, (__int64)&v74, 5);
  sub_C657C0(a1 + 65, v52->m128i_i64, (__int64 *)v80[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v52);
LABEL_21:
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
  return v52;
}
