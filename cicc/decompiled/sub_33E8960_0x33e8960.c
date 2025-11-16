// Function: sub_33E8960
// Address: 0x33e8960
//
__m128i *__fastcall sub_33E8960(
        __int64 *a1,
        int a2,
        char a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        __int64 a9,
        __int128 a10,
        __int128 a11,
        __int128 a12,
        __int128 a13,
        __int64 a14,
        __int64 a15,
        const __m128i *a16,
        char a17)
{
  unsigned __int64 v18; // r14
  unsigned __int16 v19; // r13
  unsigned __int16 *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  int v26; // r13d
  unsigned __int64 v27; // r14
  __int64 v28; // rax
  __int32 v29; // edx
  __int64 v30; // r9
  __int16 v31; // bx
  char v32; // bl
  __int64 v33; // rdx
  int v34; // eax
  unsigned __int64 v35; // r8
  const __m128i *v36; // rdi
  int v37; // eax
  __int64 v38; // r9
  __int64 v39; // rdx
  unsigned __int64 v40; // r8
  __m128i *v41; // rax
  __m128i *v42; // r13
  __int32 v44; // r10d
  __int16 v45; // ax
  __int64 v46; // rcx
  unsigned __int64 v47; // rax
  __int64 v48; // rax
  __int128 v49; // [rsp-20h] [rbp-220h]
  __int128 v50; // [rsp-20h] [rbp-220h]
  __int64 v51; // [rsp+10h] [rbp-1F0h]
  __int32 v52; // [rsp+10h] [rbp-1F0h]
  int v53; // [rsp+10h] [rbp-1F0h]
  unsigned __int16 v54; // [rsp+18h] [rbp-1E8h]
  int v56; // [rsp+20h] [rbp-1E0h]
  __int64 v57; // [rsp+28h] [rbp-1D8h]
  __int16 v58; // [rsp+30h] [rbp-1D0h]
  char v59; // [rsp+34h] [rbp-1CCh]
  __int32 v60; // [rsp+40h] [rbp-1C0h]
  __int64 v61; // [rsp+48h] [rbp-1B8h]
  unsigned __int8 *v62; // [rsp+58h] [rbp-1A8h] BYREF
  __m128i v63; // [rsp+60h] [rbp-1A0h] BYREF
  __int64 v64; // [rsp+70h] [rbp-190h]
  __int64 v65; // [rsp+78h] [rbp-188h]
  __m128i v66; // [rsp+80h] [rbp-180h]
  __m128i v67; // [rsp+90h] [rbp-170h]
  __m128i v68; // [rsp+A0h] [rbp-160h]
  __m128i v69; // [rsp+B0h] [rbp-150h]
  __m128i v70[2]; // [rsp+C0h] [rbp-140h] BYREF
  __int16 v71; // [rsp+E0h] [rbp-120h]
  __int64 v72[6]; // [rsp+110h] [rbp-F0h] BYREF
  _BYTE *v73; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v74; // [rsp+148h] [rbp-B8h]
  _BYTE v75[176]; // [rsp+150h] [rbp-B0h] BYREF

  v18 = a5;
  v19 = a4;
  v54 = a14;
  v57 = a15;
  v59 = a17;
  v64 = a8;
  v65 = a9;
  v63 = _mm_loadu_si128((const __m128i *)&a7);
  v66 = _mm_loadu_si128((const __m128i *)&a10);
  v67 = _mm_loadu_si128((const __m128i *)&a11);
  v68 = _mm_loadu_si128((const __m128i *)&a12);
  v69 = _mm_loadu_si128((const __m128i *)&a13);
  if ( a2 )
  {
    v21 = (unsigned __int16 *)(*(_QWORD *)(a8 + 48) + 16LL * (unsigned int)a9);
    v61 = sub_33E5B50(a1, a4, a5, *v21, *((_QWORD *)v21 + 1), a6, 1, 0);
  }
  else
  {
    v61 = sub_33E5110(a1, a4, a5, 1, 0);
  }
  v73 = v75;
  v74 = 0x2000000000LL;
  v51 = v22;
  sub_33C9670((__int64)&v73, 469, v61, (unsigned __int64 *)&v63, 6, v22);
  v23 = (unsigned int)v74;
  v24 = v19;
  if ( v19 )
    v18 = v19;
  v25 = v51;
  v26 = v18;
  if ( (unsigned __int64)(unsigned int)v74 + 1 > HIDWORD(v74) )
  {
    sub_C8D5F0((__int64)&v73, v75, (unsigned int)v74 + 1LL, 4u, v24, v51);
    v23 = (unsigned int)v74;
    v25 = v51;
  }
  v27 = HIDWORD(v18);
  *(_DWORD *)&v73[4 * v23] = v26;
  LODWORD(v74) = v74 + 1;
  v28 = (unsigned int)v74;
  if ( (unsigned __int64)(unsigned int)v74 + 1 > HIDWORD(v74) )
  {
    v53 = v25;
    sub_C8D5F0((__int64)&v73, v75, (unsigned int)v74 + 1LL, 4u, v24, v25);
    v28 = (unsigned int)v74;
    LODWORD(v25) = v53;
  }
  v52 = v25;
  *(_DWORD *)&v73[4 * v28] = v27;
  *((_QWORD *)&v49 + 1) = v57;
  *(_QWORD *)&v49 = v54;
  v29 = *(_DWORD *)(a6 + 8);
  LODWORD(v74) = v74 + 1;
  v62 = 0;
  sub_33CF750(v70, 469, v29, &v62, v61, v25, v49, (__int64)a16);
  v58 = a2 & 7;
  v31 = v71 & 0xFC7F | (v58 << 7);
  LOBYTE(v71) = v71 & 0x7F | ((_BYTE)v58 << 7);
  v32 = a3 & 3;
  HIBYTE(v71) = (16 * (v59 & 1)) | (4 * (a3 & 3)) | HIBYTE(v31) & 0xE3;
  LOWORD(v27) = v71 & 0xFFFA;
  if ( v72[0] )
    sub_B91220((__int64)v72, v72[0]);
  if ( v62 )
    sub_B91220((__int64)&v62, (__int64)v62);
  v33 = (unsigned int)v74;
  v34 = (unsigned __int16)v27;
  v35 = (unsigned int)v74 + 1LL;
  if ( v35 > HIDWORD(v74) )
  {
    sub_C8D5F0((__int64)&v73, v75, (unsigned int)v74 + 1LL, 4u, v35, v30);
    v33 = (unsigned int)v74;
    v34 = (unsigned __int16)v27;
  }
  v36 = a16;
  *(_DWORD *)&v73[4 * v33] = v34;
  LODWORD(v74) = v74 + 1;
  v37 = sub_2EAC1E0((__int64)v36);
  v39 = (unsigned int)v74;
  v40 = (unsigned int)v74 + 1LL;
  if ( v40 > HIDWORD(v74) )
  {
    v56 = v37;
    sub_C8D5F0((__int64)&v73, v75, (unsigned int)v74 + 1LL, 4u, v40, v38);
    v39 = (unsigned int)v74;
    v37 = v56;
  }
  *(_DWORD *)&v73[4 * v39] = v37;
  LODWORD(v74) = v74 + 1;
  v70[0].m128i_i64[0] = 0;
  v41 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v73, a6, v70[0].m128i_i64);
  v42 = v41;
  if ( v41 )
  {
    sub_2EAC4C0((__m128i *)v41[7].m128i_i64[0], a16);
    goto LABEL_19;
  }
  v42 = (__m128i *)a1[52];
  v44 = *(_DWORD *)(a6 + 8);
  if ( v42 )
  {
    a1[52] = v42->m128i_i64[0];
  }
  else
  {
    v46 = a1[53];
    a1[63] += 120;
    v47 = (v46 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v47 + 120 && v46 )
    {
      a1[53] = v47 + 120;
      if ( !v47 )
        goto LABEL_26;
      v42 = (__m128i *)((v46 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    }
    else
    {
      v60 = v44;
      v48 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
      v44 = v60;
      v42 = (__m128i *)v48;
    }
  }
  *((_QWORD *)&v50 + 1) = v57;
  *(_QWORD *)&v50 = v54;
  sub_33CF750(v42, 469, v44, (unsigned __int8 **)a6, v61, v52, v50, (__int64)a16);
  v45 = (v58 << 7) | v42[2].m128i_i16[0] & 0xFC7F;
  v42[2].m128i_i16[0] = v45;
  v42[2].m128i_i8[1] = HIBYTE(v45) & 0xE3 | (4 * v32) | (16 * (v59 & 1));
LABEL_26:
  sub_33E4EC0((__int64)a1, (__int64)v42, (__int64)&v63, 6);
  sub_C657C0(a1 + 65, v42->m128i_i64, (__int64 *)v70[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v42);
LABEL_19:
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  return v42;
}
