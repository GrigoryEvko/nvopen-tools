// Function: sub_30E0710
// Address: 0x30e0710
//
__m128i *__fastcall sub_30E0710(
        __m128i *a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v11; // rdi
  int v12; // eax
  int v13; // r15d
  int v14; // eax
  int v15; // eax
  int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rbx
  __int64 v19; // r12
  unsigned __int64 v20; // rdi
  __m128i v22; // xmm0
  __m128i v23; // xmm1
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __m128i v26; // xmm4
  __m128i v27; // xmm5
  __int32 v28; // eax
  __int64 (__fastcall **v29)(); // [rsp+0h] [rbp-350h] BYREF
  __int64 *v30; // [rsp+8h] [rbp-348h]
  __int64 v31; // [rsp+10h] [rbp-340h]
  __int64 v32; // [rsp+18h] [rbp-338h]
  __int64 v33; // [rsp+20h] [rbp-330h]
  __int64 v34; // [rsp+28h] [rbp-328h]
  __int64 v35; // [rsp+30h] [rbp-320h]
  __int64 v36; // [rsp+38h] [rbp-318h]
  __int64 v37; // [rsp+40h] [rbp-310h]
  __int64 v38; // [rsp+48h] [rbp-308h]
  __int64 v39; // [rsp+50h] [rbp-300h]
  __int64 v40; // [rsp+58h] [rbp-2F8h]
  __int64 v41; // [rsp+60h] [rbp-2F0h]
  __int64 v42; // [rsp+68h] [rbp-2E8h]
  char v43; // [rsp+70h] [rbp-2E0h]
  __int64 v44; // [rsp+78h] [rbp-2D8h]
  __int64 v45; // [rsp+80h] [rbp-2D0h]
  __int64 v46; // [rsp+88h] [rbp-2C8h]
  __int64 v47; // [rsp+90h] [rbp-2C0h]
  __int64 v48; // [rsp+98h] [rbp-2B8h]
  unsigned int v49; // [rsp+A0h] [rbp-2B0h]
  __int64 v50; // [rsp+A8h] [rbp-2A8h]
  __int64 v51; // [rsp+B0h] [rbp-2A0h]
  __int64 v52; // [rsp+B8h] [rbp-298h]
  unsigned int v53; // [rsp+C0h] [rbp-290h]
  __int64 v54; // [rsp+C8h] [rbp-288h]
  __int64 v55; // [rsp+D0h] [rbp-280h]
  __int64 v56; // [rsp+D8h] [rbp-278h]
  __int64 v57; // [rsp+E0h] [rbp-270h]
  __int64 v58; // [rsp+E8h] [rbp-268h]
  __int64 v59; // [rsp+F0h] [rbp-260h]
  __int64 v60; // [rsp+F8h] [rbp-258h]
  unsigned int v61; // [rsp+100h] [rbp-250h]
  __int64 v62; // [rsp+108h] [rbp-248h]
  char *v63; // [rsp+110h] [rbp-240h]
  __int64 v64; // [rsp+118h] [rbp-238h]
  int v65; // [rsp+120h] [rbp-230h]
  char v66; // [rsp+124h] [rbp-22Ch]
  char v67; // [rsp+128h] [rbp-228h] BYREF
  __int64 v68; // [rsp+1A8h] [rbp-1A8h]
  __int64 v69; // [rsp+1B0h] [rbp-1A0h]
  __int64 v70; // [rsp+1B8h] [rbp-198h]
  unsigned int v71; // [rsp+1C0h] [rbp-190h]
  __int16 v72; // [rsp+1C8h] [rbp-188h]
  __int64 v73; // [rsp+1D0h] [rbp-180h]
  char *v74; // [rsp+1D8h] [rbp-178h]
  __int64 v75; // [rsp+1E0h] [rbp-170h]
  int v76; // [rsp+1E8h] [rbp-168h]
  char v77; // [rsp+1ECh] [rbp-164h]
  char v78; // [rsp+1F0h] [rbp-160h] BYREF
  __int64 v79; // [rsp+270h] [rbp-E0h]
  __int64 v80; // [rsp+278h] [rbp-D8h]
  __int64 v81; // [rsp+280h] [rbp-D0h]
  __m128i v82[6]; // [rsp+288h] [rbp-C8h] BYREF
  __int32 v83; // [rsp+2E8h] [rbp-68h]
  __int64 v84; // [rsp+2ECh] [rbp-64h]
  __int64 v85; // [rsp+2F4h] [rbp-5Ch]
  __int64 v86; // [rsp+300h] [rbp-50h]
  __int64 v87; // [rsp+308h] [rbp-48h]
  __int64 v88; // [rsp+310h] [rbp-40h]
  unsigned int v89; // [rsp+318h] [rbp-38h]

  v11 = *(_QWORD *)(a2 - 32);
  if ( v11 )
  {
    if ( *(_BYTE *)v11 )
    {
      v11 = 0;
    }
    else if ( *(_QWORD *)(v11 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v11 = 0;
    }
  }
  v32 = a5;
  v33 = a7;
  v34 = a8;
  v37 = a6;
  v30 = a3;
  v31 = a4;
  v35 = a9;
  v38 = v11;
  v36 = a10;
  v41 = a2;
  v39 = sub_B2BEC0(v11);
  v63 = &v67;
  v40 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v64 = 16;
  v65 = 0;
  v66 = 1;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 1;
  v74 = &v78;
  memset(v82, 0, sizeof(v82));
  v73 = 0;
  v75 = 16;
  v76 = 0;
  v77 = 1;
  v29 = off_49D8A00;
  v79 = 0;
  v83 = 0;
  v80 = 0;
  v81 = 0;
  v84 = 0;
  v85 = 0x500000000LL;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v12 = sub_30D4FE0(v30, (unsigned __int8 *)a2, v39);
  v82[4].m128i_i32[2] -= v12;
  v82[4].m128i_i32[3] = ((*(_WORD *)(v11 + 2) >> 4) & 0x3FF) == 9;
  v82[5].m128i_i32[0] = (unsigned __int8)sub_30D14D0(a2, v11);
  v13 = sub_DF94D0((__int64)v30);
  v14 = sub_DF9470((__int64)v30);
  HIDWORD(v85) += v14;
  v15 = sub_DF93B0((__int64)v30);
  v16 = HIDWORD(v85) * v15;
  LODWORD(v85) = 50 * v16 / 100;
  HIDWORD(v84) = v16 * v13 / 100;
  HIDWORD(v85) = v16 + HIDWORD(v84) + v85;
  if ( v38 + 72 == (*(_QWORD *)(v38 + 72) & 0xFFFFFFFFFFFFFFF8LL) || !sub_30DC7E0(&v29) )
  {
    v22 = _mm_loadu_si128(v82);
    v23 = _mm_loadu_si128(&v82[1]);
    a1[6].m128i_i8[4] = 1;
    v24 = _mm_loadu_si128(&v82[2]);
    v25 = _mm_loadu_si128(&v82[3]);
    v26 = _mm_loadu_si128(&v82[4]);
    v27 = _mm_loadu_si128(&v82[5]);
    *a1 = v22;
    v28 = v83;
    a1[1] = v23;
    a1[2] = v24;
    a1[6].m128i_i32[0] = v28;
    a1[3] = v25;
    a1[4] = v26;
    a1[5] = v27;
  }
  else
  {
    a1[6].m128i_i8[4] = 0;
  }
  v29 = off_49D8A00;
  sub_C7D6A0(v87, 16LL * v89, 8);
  v29 = off_49D8850;
  if ( !v77 )
    _libc_free((unsigned __int64)v74);
  sub_C7D6A0(v69, 16LL * v71, 8);
  if ( !v66 )
    _libc_free((unsigned __int64)v63);
  v17 = v61;
  if ( v61 )
  {
    v18 = v59;
    v19 = v59 + 32LL * v61;
    do
    {
      if ( *(_QWORD *)v18 != -8192 && *(_QWORD *)v18 != -4096 && *(_DWORD *)(v18 + 24) > 0x40u )
      {
        v20 = *(_QWORD *)(v18 + 16);
        if ( v20 )
          j_j___libc_free_0_0(v20);
      }
      v18 += 32;
    }
    while ( v19 != v18 );
    v17 = v61;
  }
  sub_C7D6A0(v59, 32 * v17, 8);
  sub_C7D6A0(v55, 8LL * (unsigned int)v57, 8);
  sub_C7D6A0(v51, 16LL * v53, 8);
  sub_C7D6A0(v47, 16LL * v49, 8);
  return a1;
}
