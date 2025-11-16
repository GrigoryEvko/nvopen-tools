// Function: sub_33E8F60
// Address: 0x33e8f60
//
__m128i *__fastcall sub_33E8F60(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        unsigned __int64 a7,
        unsigned __int64 a8,
        __int128 a9,
        __int128 a10,
        __int128 a11,
        __int64 a12,
        __int64 a13,
        const __m128i *a14,
        int a15,
        char a16,
        char a17)
{
  unsigned __int16 v21; // r13
  unsigned __int16 *v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // r11
  unsigned __int64 v26; // r10
  __m128i v27; // xmm0
  __m128i v28; // xmm1
  __m128i v29; // xmm2
  __int64 v30; // rax
  __int64 v31; // r9
  unsigned __int64 v32; // r13
  __int64 v33; // r8
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // r13
  __int64 v37; // rax
  __int32 v38; // edx
  __int64 v39; // r9
  __int16 v40; // dx
  int v41; // eax
  __int64 v42; // rdx
  unsigned __int64 v43; // r8
  const __m128i *v44; // rdi
  int v45; // eax
  __int64 v46; // r9
  __int64 v47; // rdx
  unsigned __int64 v48; // r8
  const __m128i *v49; // rdi
  __int64 v50; // r8
  __int64 v51; // rax
  __m128i *v52; // rax
  __m128i *v53; // r13
  __int32 v55; // r14d
  __int16 v56; // ax
  __int64 v57; // rcx
  unsigned __int64 v58; // rax
  __int128 v59; // [rsp-20h] [rbp-200h]
  __int128 v60; // [rsp-20h] [rbp-200h]
  __int32 v61; // [rsp+Ch] [rbp-1D4h]
  char v62; // [rsp+10h] [rbp-1D0h]
  __int64 v63; // [rsp+20h] [rbp-1C0h]
  __int16 v64; // [rsp+20h] [rbp-1C0h]
  __int16 v65; // [rsp+20h] [rbp-1C0h]
  int v66; // [rsp+20h] [rbp-1C0h]
  int v67; // [rsp+20h] [rbp-1C0h]
  int v68; // [rsp+20h] [rbp-1C0h]
  int v69; // [rsp+20h] [rbp-1C0h]
  char v70; // [rsp+28h] [rbp-1B8h]
  __int16 v71; // [rsp+2Ch] [rbp-1B4h]
  unsigned __int16 v72; // [rsp+2Eh] [rbp-1B2h]
  __int64 v73; // [rsp+30h] [rbp-1B0h]
  __int64 v74; // [rsp+38h] [rbp-1A8h]
  unsigned __int8 *v75; // [rsp+48h] [rbp-198h] BYREF
  unsigned __int64 v76[4]; // [rsp+50h] [rbp-190h] BYREF
  __m128i v77; // [rsp+70h] [rbp-170h]
  __m128i v78; // [rsp+80h] [rbp-160h]
  __m128i v79; // [rsp+90h] [rbp-150h]
  __m128i v80[2]; // [rsp+A0h] [rbp-140h] BYREF
  __int16 v81; // [rsp+C0h] [rbp-120h]
  __int64 v82[6]; // [rsp+F0h] [rbp-F0h] BYREF
  _BYTE *v83; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v84; // [rsp+128h] [rbp-B8h]
  _BYTE v85[176]; // [rsp+130h] [rbp-B0h] BYREF

  v21 = a12;
  v73 = a13;
  v72 = a12;
  v70 = a17;
  if ( a15 )
  {
    v22 = (unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8);
    v23 = sub_33E5B50(a1, a2, a3, *v22, *((_QWORD *)v22 + 1), a6, 1, 0);
    v25 = a7;
    v26 = a8;
  }
  else
  {
    v23 = sub_33E5110(a1, a2, a3, 1, 0);
    v26 = a8;
    v25 = a7;
  }
  v74 = v23;
  v27 = _mm_loadu_si128((const __m128i *)&a9);
  v28 = _mm_loadu_si128((const __m128i *)&a10);
  v76[0] = a5;
  v84 = 0x2000000000LL;
  v29 = _mm_loadu_si128((const __m128i *)&a11);
  v76[1] = a6;
  v63 = v24;
  v76[2] = v25;
  v76[3] = v26;
  v83 = v85;
  v77 = v27;
  v78 = v28;
  v79 = v29;
  sub_33C9670((__int64)&v83, 362, v23, v76, 5, v24);
  v30 = v21;
  if ( !v21 )
    v30 = v73;
  v31 = v63;
  v32 = v30;
  v33 = (unsigned int)v30;
  v34 = (unsigned int)v84;
  v35 = (unsigned int)v84 + 1LL;
  if ( v35 > HIDWORD(v84) )
  {
    sub_C8D5F0((__int64)&v83, v85, v35, 4u, v33, v63);
    v34 = (unsigned int)v84;
    v31 = v63;
    v33 = (unsigned int)v32;
  }
  v36 = HIDWORD(v32);
  *(_DWORD *)&v83[4 * v34] = v33;
  LODWORD(v84) = v84 + 1;
  v37 = (unsigned int)v84;
  if ( (unsigned __int64)(unsigned int)v84 + 1 > HIDWORD(v84) )
  {
    v67 = v31;
    sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 4u, v33, v31);
    v37 = (unsigned int)v84;
    LODWORD(v31) = v67;
  }
  v61 = v31;
  *(_DWORD *)&v83[4 * v37] = v36;
  *((_QWORD *)&v59 + 1) = v73;
  *(_QWORD *)&v59 = v72;
  v38 = *(_DWORD *)(a4 + 8);
  LODWORD(v84) = v84 + 1;
  v75 = 0;
  sub_33CF750(v80, 362, v38, &v75, v74, v31, v59, (__int64)a14);
  v71 = a15 & 7;
  v62 = a16 & 3;
  v40 = (v71 << 7) | v81 & 0xFC7F;
  LOBYTE(v81) = ((a15 & 7) << 7) | v81 & 0x7F;
  HIBYTE(v81) = (16 * (v70 & 1)) | (4 * (a16 & 3)) | HIBYTE(v40) & 0xE3;
  LOWORD(v41) = v81 & 0xFFFA;
  if ( v82[0] )
  {
    v64 = v81 & 0xFFFA;
    sub_B91220((__int64)v82, v82[0]);
    LOWORD(v41) = v64;
  }
  if ( v75 )
  {
    v65 = v41;
    sub_B91220((__int64)&v75, (__int64)v75);
    LOWORD(v41) = v65;
  }
  v42 = (unsigned int)v84;
  v41 = (unsigned __int16)v41;
  v43 = (unsigned int)v84 + 1LL;
  if ( v43 > HIDWORD(v84) )
  {
    v68 = (unsigned __int16)v41;
    sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 4u, v43, v39);
    v42 = (unsigned int)v84;
    v41 = v68;
  }
  v44 = a14;
  *(_DWORD *)&v83[4 * v42] = v41;
  LODWORD(v84) = v84 + 1;
  v45 = sub_2EAC1E0((__int64)v44);
  v47 = (unsigned int)v84;
  v48 = (unsigned int)v84 + 1LL;
  if ( v48 > HIDWORD(v84) )
  {
    v69 = v45;
    sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 4u, v48, v46);
    v47 = (unsigned int)v84;
    v45 = v69;
  }
  v49 = a14;
  *(_DWORD *)&v83[4 * v47] = v45;
  v50 = v49[2].m128i_u16[0];
  LODWORD(v84) = v84 + 1;
  v51 = (unsigned int)v84;
  if ( (unsigned __int64)(unsigned int)v84 + 1 > HIDWORD(v84) )
  {
    v66 = v50;
    sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 4u, v50, v46);
    v51 = (unsigned int)v84;
    LODWORD(v50) = v66;
  }
  *(_DWORD *)&v83[4 * v51] = v50;
  LODWORD(v84) = v84 + 1;
  v80[0].m128i_i64[0] = 0;
  v52 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v83, a4, v80[0].m128i_i64);
  v53 = v52;
  if ( v52 )
  {
    sub_2EAC4C0((__m128i *)v52[7].m128i_i64[0], a14);
    goto LABEL_21;
  }
  v53 = (__m128i *)a1[52];
  v55 = *(_DWORD *)(a4 + 8);
  if ( v53 )
  {
    a1[52] = v53->m128i_i64[0];
  }
  else
  {
    v57 = a1[53];
    a1[63] += 120;
    v58 = (v57 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v58 + 120 && v57 )
    {
      a1[53] = v58 + 120;
      if ( !v58 )
        goto LABEL_28;
    }
    else
    {
      v58 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    }
    v53 = (__m128i *)v58;
  }
  *((_QWORD *)&v60 + 1) = v73;
  *(_QWORD *)&v60 = v72;
  sub_33CF750(v53, 362, v55, (unsigned __int8 **)a4, v74, v61, v60, (__int64)a14);
  v56 = v53[2].m128i_i16[0] & 0xFC7F | (v71 << 7);
  v53[2].m128i_i16[0] = v56;
  v53[2].m128i_i8[1] = HIBYTE(v56) & 0xE3 | (4 * v62) | (16 * (v70 & 1));
LABEL_28:
  sub_33E4EC0((__int64)a1, (__int64)v53, (__int64)v76, 5);
  sub_C657C0(a1 + 65, v53->m128i_i64, (__int64 *)v80[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v53);
LABEL_21:
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
  return v53;
}
