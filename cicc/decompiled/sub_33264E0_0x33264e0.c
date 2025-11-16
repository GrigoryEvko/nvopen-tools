// Function: sub_33264E0
// Address: 0x33264e0
//
__m128i *__fastcall sub_33264E0(__m128i *a1, __int64 a2, int a3, __int64 a4, unsigned __int64 *a5, unsigned __int8 a6)
{
  __int64 v9; // r14
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int); // r15
  __int64 v11; // rax
  int v12; // edx
  unsigned __int16 v13; // ax
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  int v19; // edx
  unsigned __int16 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 *v26; // rax
  __int64 v27; // r15
  __m128i *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 (__fastcall *v31)(__int64, __int64, unsigned int); // rax
  __int64 v32; // rsi
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rdi
  unsigned int v35; // r12d
  unsigned __int64 v36; // rdx
  unsigned __int64 v37; // rax
  void (***v38)(); // rdi
  void (*v39)(); // rax
  __m128i v40; // xmm2
  __int64 v42; // r15
  int v43; // edx
  int v44; // r14d
  __int64 v45; // r15
  __m128i v46; // xmm0
  __int64 v47; // [rsp+18h] [rbp-11E8h]
  int v49; // [rsp+2Ch] [rbp-11D4h]
  __int64 v50; // [rsp+30h] [rbp-11D0h]
  unsigned __int8 v51; // [rsp+3Ah] [rbp-11C6h]
  _BYTE v52[5]; // [rsp+3Bh] [rbp-11C5h]
  const char *v53; // [rsp+40h] [rbp-11C0h]
  __m128i *v55; // [rsp+50h] [rbp-11B0h]
  __int16 v57; // [rsp+60h] [rbp-11A0h] BYREF
  __int64 v58; // [rsp+68h] [rbp-1198h]
  const char *v59; // [rsp+70h] [rbp-1190h] BYREF
  int v60; // [rsp+78h] [rbp-1188h]
  __m128i v61; // [rsp+80h] [rbp-1180h] BYREF
  __m128i v62; // [rsp+90h] [rbp-1170h] BYREF
  const char *v63; // [rsp+A0h] [rbp-1160h] BYREF
  __int64 v64; // [rsp+A8h] [rbp-1158h]
  __m128i *v65; // [rsp+B0h] [rbp-1150h]
  unsigned __int64 v66; // [rsp+B8h] [rbp-1148h]
  __int64 v67; // [rsp+C0h] [rbp-1140h]
  __int64 v68; // [rsp+C8h] [rbp-1138h]
  __int64 v69; // [rsp+D0h] [rbp-1130h]
  unsigned __int64 v70; // [rsp+D8h] [rbp-1128h] BYREF
  unsigned __int64 v71; // [rsp+E0h] [rbp-1120h]
  unsigned __int64 v72; // [rsp+E8h] [rbp-1118h]
  __int64 v73; // [rsp+F0h] [rbp-1110h]
  __int64 v74; // [rsp+F8h] [rbp-1108h] BYREF
  __int32 v75; // [rsp+100h] [rbp-1100h]
  __int64 v76; // [rsp+108h] [rbp-10F8h]
  _BYTE *v77; // [rsp+110h] [rbp-10F0h]
  __int64 v78; // [rsp+118h] [rbp-10E8h]
  _BYTE v79[1792]; // [rsp+120h] [rbp-10E0h] BYREF
  _BYTE *v80; // [rsp+820h] [rbp-9E0h]
  __int64 v81; // [rsp+828h] [rbp-9D8h]
  _BYTE v82[512]; // [rsp+830h] [rbp-9D0h] BYREF
  _BYTE *v83; // [rsp+A30h] [rbp-7D0h]
  __int64 v84; // [rsp+A38h] [rbp-7C8h]
  _BYTE v85[1792]; // [rsp+A40h] [rbp-7C0h] BYREF
  _BYTE *v86; // [rsp+1140h] [rbp-C0h]
  __int64 v87; // [rsp+1148h] [rbp-B8h]
  _BYTE v88[64]; // [rsp+1150h] [rbp-B0h] BYREF
  __int64 v89; // [rsp+1190h] [rbp-70h]
  __int64 v90; // [rsp+1198h] [rbp-68h]
  int v91; // [rsp+11A0h] [rbp-60h]
  char v92; // [rsp+11C0h] [rbp-40h]

  v9 = *(_QWORD *)(a2 + 8);
  v51 = a6;
  v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v9 + 32LL);
  v11 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a2 + 16) + 40LL));
  if ( v10 == sub_2D42F30 )
  {
    v12 = sub_AE2980(v11, 0)[1];
    v13 = 2;
    if ( v12 != 1 )
    {
      v13 = 3;
      if ( v12 != 2 )
      {
        v13 = 4;
        if ( v12 != 4 )
        {
          v13 = 5;
          if ( v12 != 8 )
          {
            v13 = 6;
            if ( v12 != 16 )
            {
              v13 = 7;
              if ( v12 != 32 )
              {
                v13 = 8;
                if ( v12 != 64 )
                  v13 = 9 * (v12 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v13 = v10(v9, v11, 0);
  }
  v14 = *(_QWORD *)(a2 + 16);
  v15 = *(_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL * a3 + 525288);
  v47 = a3;
  if ( v15 )
  {
    v50 = sub_33EED90(v14, v15, v13, 0);
    v49 = v19;
  }
  else
  {
    v63 = 0;
    LODWORD(v64) = 0;
    v42 = sub_33F17F0(v14, 51, &v63, v13, 0);
    v44 = v43;
    if ( v63 )
      sub_B91220((__int64)&v63, (__int64)v63);
    v49 = v44;
    v50 = v42;
    v45 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 64LL);
    sub_3418C90(&v61, a4);
    v65 = &v61;
    v63 = "no libcall available for ";
    LOWORD(v67) = 1027;
    sub_B6ECE0(v45, (__int64)&v63);
    if ( (__m128i *)v61.m128i_i64[0] != &v62 )
      j_j___libc_free_0(v61.m128i_u64[0]);
  }
  v20 = *(unsigned __int16 **)(a4 + 48);
  v21 = *v20;
  v58 = *((_QWORD *)v20 + 1);
  v22 = *(_QWORD *)(a2 + 16);
  v57 = v21;
  v23 = sub_3007410((__int64)&v57, *(__int64 **)(v22 + 64), v21, v16, v17, v18);
  v24 = *(_QWORD *)(a2 + 16);
  v25 = *(_QWORD *)(a2 + 8);
  v55 = (__m128i *)v23;
  v26 = *(__int64 **)(v24 + 40);
  v60 = 0;
  v59 = (const char *)(v24 + 288);
  v53 = (const char *)(v24 + 288);
  v27 = *v26;
  v52[4] = 0;
  *(_DWORD *)v52 = (unsigned __int8)sub_3446F60(v25, v24, a4, &v59);
  if ( v52[0] )
  {
    v28 = **(__m128i ***)(*(_QWORD *)(v27 + 24) + 16LL);
    if ( v55 == v28 || v28->m128i_i8[8] == 7 )
    {
      v53 = v59;
      *(_DWORD *)&v52[1] = v60;
    }
    else
    {
      v52[0] = 0;
    }
  }
  v29 = *(_QWORD *)(a2 + 16);
  v30 = *(_QWORD *)(a2 + 8);
  v66 = 0xFFFFFFFF00000020LL;
  v73 = v29;
  v77 = v79;
  v78 = 0x2000000000LL;
  v81 = 0x2000000000LL;
  v84 = 0x2000000000LL;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v80 = v82;
  v83 = v85;
  v86 = v88;
  v87 = 0x400000000LL;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v31 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v30 + 1128LL);
  if ( v31 != sub_2FE3330 )
    v51 = v31(v30, (__int64)v55, a6);
  v32 = *(_QWORD *)(a4 + 80);
  v61.m128i_i64[0] = v32;
  if ( v32 )
    sub_B96E90((__int64)&v61, v32, 1);
  v61.m128i_i32[2] = *(_DWORD *)(a4 + 72);
  if ( v74 )
    sub_B91220((__int64)&v74, v74);
  v74 = v61.m128i_i64[0];
  if ( v61.m128i_i64[0] )
    sub_B96E90((__int64)&v74, v61.m128i_i64[0], 1);
  v33 = *a5;
  v34 = v70;
  v75 = v61.m128i_i32[2];
  v63 = v53;
  LODWORD(v64) = *(_DWORD *)&v52[1];
  v35 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 4 * v47 + 531128);
  v70 = v33;
  v36 = a5[1];
  *a5 = 0;
  v65 = v55;
  LODWORD(v67) = v35;
  v68 = v50;
  v71 = v36;
  LODWORD(v69) = v49;
  a5[1] = 0;
  HIDWORD(v66) = -1431655765 * ((__int64)(v36 - v33) >> 4);
  v37 = a5[2];
  a5[2] = 0;
  v72 = v37;
  if ( v34 )
    j_j___libc_free_0(v34);
  v38 = *(void (****)())(v73 + 16);
  v39 = **v38;
  if ( v39 != nullsub_1688 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v39)(v38, *(_QWORD *)(v73 + 40), v35, &v70);
  BYTE3(v66) = 1;
  BYTE2(v66) = v52[0];
  LOBYTE(v66) = v51 & 1 | v66 & 0xFC | (2 * ((v51 ^ 1) & 1));
  if ( v61.m128i_i64[0] )
    sub_B91220((__int64)&v61, v61.m128i_i64[0]);
  sub_3377410(&v61, *(_QWORD *)(a2 + 8), &v63);
  if ( v62.m128i_i64[0] )
  {
    v40 = _mm_loadu_si128(&v62);
    *a1 = _mm_loadu_si128(&v61);
    a1[1] = v40;
  }
  else
  {
    v46 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 16) + 384LL));
    *a1 = v46;
    a1[1] = v46;
  }
  if ( v86 != v88 )
    _libc_free((unsigned __int64)v86);
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
  if ( v80 != v82 )
    _libc_free((unsigned __int64)v80);
  if ( v77 != v79 )
    _libc_free((unsigned __int64)v77);
  if ( v74 )
    sub_B91220((__int64)&v74, v74);
  if ( v70 )
    j_j___libc_free_0(v70);
  return a1;
}
