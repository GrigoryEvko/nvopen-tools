// Function: sub_304FBD0
// Address: 0x304fbd0
//
__int64 __fastcall sub_304FBD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // r8
  _QWORD *v8; // rax
  char v9; // al
  int v10; // r9d
  __int64 v11; // rcx
  unsigned int v12; // edi
  unsigned int v13; // r11d
  __int64 v14; // rcx
  unsigned int v15; // edi
  int v16; // edx
  __int64 v18; // r9
  _QWORD *v19; // r11
  __int16 v20; // r14
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r9
  unsigned __int64 v25; // rdx
  __int64 *v26; // rax
  int v27; // edx
  unsigned __int8 v28; // cl
  __int64 v29; // rax
  unsigned int v30; // ecx
  __m128i v31; // xmm0
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // rax
  __m128i v35; // xmm0
  unsigned int v36; // eax
  int v37; // eax
  int v38; // edx
  int v39; // r9d
  int v40; // r9d
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  unsigned __int64 v45; // r10
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r12
  __int64 v49; // r9
  _QWORD *v50; // rax
  int v51; // r14d
  unsigned __int8 v52; // al
  __int64 v53; // r8
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r9
  unsigned __int64 v57; // rdx
  __int64 *v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // rax
  __m128i v62; // xmm0
  int v63; // eax
  int v64; // edx
  int v65; // r9d
  int v66; // r9d
  __int64 v67; // rax
  unsigned __int64 v68; // rdx
  __int64 v69; // r12
  __int64 v70; // rcx
  int v71; // r8d
  __int64 v72; // rax
  __m128i v73; // xmm0
  __int64 v74; // rdx
  __int64 v75; // rax
  __m128i v76; // xmm0
  __int64 v77; // rcx
  int v78; // r8d
  __int64 v79; // [rsp-20h] [rbp-F0h]
  __int64 v80; // [rsp-20h] [rbp-F0h]
  __m128i v81; // [rsp+0h] [rbp-D0h] BYREF
  __m128i v82; // [rsp+10h] [rbp-C0h] BYREF
  _BYTE **v83; // [rsp+28h] [rbp-A8h]
  __int64 v84; // [rsp+30h] [rbp-A0h]
  __int64 v85; // [rsp+38h] [rbp-98h]
  __int64 v86; // [rsp+40h] [rbp-90h] BYREF
  int v87; // [rsp+48h] [rbp-88h]
  _BYTE *v88; // [rsp+50h] [rbp-80h] BYREF
  __int64 v89; // [rsp+58h] [rbp-78h]
  _BYTE v90[112]; // [rsp+60h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  v7 = *(unsigned int *)(v6 + 32);
  v8 = *(_QWORD **)(v6 + 24);
  if ( (unsigned int)v7 > 0x40 )
    v8 = (_QWORD *)*v8;
  v10 = (int)v8;
  v9 = (unsigned __int8)v8 & 0xF;
  LOBYTE(v10) = (unsigned __int8)v10 >> 4;
  v11 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 16LL);
  v12 = *(_DWORD *)(v11 + 340);
  if ( v12 <= 0x383 || (v13 = *(_DWORD *)(v11 + 336), v13 <= 0x4D) )
    sub_C64ED0("cvt_packfloat intrinsic needs atleast SM90 and PTX >= 78", 1u);
  if ( v12 > 0x3E7 && ((v15 = v12 % 0xA, v15 == 1) || v15 == 2 && v13 > 0x57) )
  {
    v14 = (unsigned int)(unsigned __int8)v10 - 6;
  }
  else
  {
    if ( (unsigned __int8)v10 == 8 || v9 == 8 )
      sub_C64ED0(
        "ue8m0x2 type in cvt_packfloat intrinsic supported only in arch-conditional or family-conditional variants from SM100 onwards.",
        1u);
    v14 = (unsigned int)(unsigned __int8)v10 - 6;
    if ( (unsigned int)v14 <= 1 || ((v9 + 11) & 0xFu) <= 2 || (unsigned __int8)v10 == 5 )
      sub_C64ED0(
        "{fp6/fp4}x2 types in cvt_packfloat intrinsic supported only in arch-conditional variants from SM100 onwards.",
        1u);
  }
  if ( !(_BYTE)v10 )
  {
    v18 = *(_QWORD *)(a2 + 80);
    v86 = v18;
    if ( v18 )
    {
      sub_B96E90((__int64)&v86, v18, 1);
      v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
      LODWORD(v7) = *(_DWORD *)(v6 + 32);
    }
    v87 = *(_DWORD *)(a2 + 72);
    v19 = *(_QWORD **)(v6 + 24);
    if ( (unsigned int)v7 > 0x40 )
      v19 = (_QWORD *)*v19;
    v20 = (__int16)v19;
    v83 = &v88;
    v88 = v90;
    v89 = 0x400000000LL;
    v21 = sub_3400BD0(a4, (_DWORD)v19, (unsigned int)&v86, 8, 0, 1, 0);
    v22 = (unsigned int)v89;
    v24 = v23;
    v25 = (unsigned int)v89 + 1LL;
    if ( v25 > HIDWORD(v89) )
    {
      v81.m128i_i64[0] = v21;
      v81.m128i_i64[1] = v24;
      sub_C8D5F0((__int64)v83, v90, v25, 0x10u, v21, v24);
      v22 = (unsigned int)v89;
      v24 = v81.m128i_i64[1];
      v21 = v81.m128i_i64[0];
    }
    v26 = (__int64 *)&v88[16 * v22];
    *v26 = v21;
    v27 = v20 & 0xF;
    v28 = v20 & 0xF;
    v26[1] = v24;
    v29 = (unsigned int)(v89 + 1);
    LODWORD(v89) = v89 + 1;
    if ( v27 == 1 || v27 == 4 )
    {
      v73 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 80LL));
      if ( v29 + 1 > (unsigned __int64)HIDWORD(v89) )
      {
        v81 = v73;
        sub_C8D5F0((__int64)v83, v90, v29 + 1, 0x10u, v21, v24);
        v29 = (unsigned int)v89;
        v73 = _mm_load_si128(&v81);
      }
      *(__m128i *)&v88[16 * v29] = v73;
      v74 = *(_QWORD *)(a2 + 40);
      LODWORD(v89) = v89 + 1;
      v75 = (unsigned int)v89;
      v76 = _mm_loadu_si128((const __m128i *)(v74 + 120));
      if ( (unsigned __int64)(unsigned int)v89 + 1 > HIDWORD(v89) )
      {
        v81 = v76;
        sub_C8D5F0((__int64)v83, v90, (unsigned int)v89 + 1LL, 0x10u, v21, v24);
        v75 = (unsigned int)v89;
        v76 = _mm_load_si128(&v81);
      }
      *(__m128i *)&v88[16 * v75] = v76;
      v77 = *(_QWORD *)(a2 + 48);
      v78 = *(_DWORD *)(a2 + 68);
      LODWORD(v89) = v89 + 1;
      v48 = sub_33E66D0(a4, 1054, (unsigned int)&v86, v77, v78, v24, (__int64)v88, (unsigned int)v89);
      goto LABEL_47;
    }
    if ( v28 > 3u )
    {
      if ( v28 != 8 )
      {
        v30 = v27 - 6;
        if ( v27 == 5 )
        {
LABEL_37:
          LOBYTE(v21) = v30 <= 1;
LABEL_40:
          v31 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 80LL));
          if ( v29 + 1 > (unsigned __int64)HIDWORD(v89) )
          {
            v82.m128i_i8[0] = v21;
            v81 = v31;
            sub_C8D5F0((__int64)v83, v90, v29 + 1, 0x10u, v21, v24);
            v29 = (unsigned int)v89;
            v31 = _mm_load_si128(&v81);
            v21 = v82.m128i_u8[0];
          }
          *(__m128i *)&v88[16 * v29] = v31;
          v32 = *(_QWORD *)(a2 + 40);
          v33 = HIDWORD(v89);
          LODWORD(v89) = v89 + 1;
          v34 = (unsigned int)v89;
          v35 = _mm_loadu_si128((const __m128i *)(v32 + 120));
          if ( (unsigned __int64)(unsigned int)v89 + 1 > HIDWORD(v89) )
          {
            v82.m128i_i8[0] = v21;
            v81 = v35;
            sub_C8D5F0((__int64)v83, v90, (unsigned int)v89 + 1LL, 0x10u, v21, v24);
            v34 = (unsigned int)v89;
            v35 = _mm_load_si128(&v81);
            LOBYTE(v21) = v82.m128i_i8[0];
          }
          *(__m128i *)&v88[16 * v34] = v35;
          v36 = v89 + 1;
          LODWORD(v89) = v89 + 1;
          if ( (_BYTE)v21 )
          {
            v37 = sub_33ED250(a4, 6, 0, v33);
            v79 = sub_33E66D0(a4, 1053, (unsigned int)&v86, v37, v38, v39, (__int64)v88, (unsigned int)v89);
            v41 = sub_33FAF80(a4, 214, (unsigned int)&v86, 7, 0, v40, (unsigned __int64)v79);
            v43 = v42;
            v84 = v41;
            v44 = v41;
            v85 = v43;
            v45 = (unsigned int)v43;
          }
          else
          {
            v70 = *(_QWORD *)(a2 + 48);
            v71 = *(_DWORD *)(a2 + 68);
            v83 = 0;
            v72 = sub_33E66D0(a4, 1057, (unsigned int)&v86, v70, v71, v24, (__int64)v88, v36);
            v45 = (unsigned __int64)v83;
            v44 = v72;
          }
          v46 = sub_302EAB0(
                  a4,
                  (int)&v86,
                  v44,
                  v45,
                  *(_QWORD *)(*(_QWORD *)(a2 + 40) + 160LL),
                  *(_QWORD *)(*(_QWORD *)(a2 + 40) + 168LL),
                  HIBYTE(v20) & 1);
          v82.m128i_i64[0] = v47;
          v48 = v46;
LABEL_47:
          if ( v88 != v90 )
            _libc_free((unsigned __int64)v88);
          if ( v86 )
            sub_B91220((__int64)&v86, v86);
          return v48;
        }
        if ( v30 > 1 )
          goto LABEL_80;
      }
    }
    else if ( v28 <= 1u )
    {
      v30 = v27 - 6;
      if ( v27 != 5 )
        goto LABEL_80;
      goto LABEL_37;
    }
    v21 = 1;
    goto LABEL_40;
  }
  if ( (unsigned __int8)v10 != 1 && (unsigned __int8)v10 != 4 )
  {
    if ( (unsigned __int8)v10 <= 3u || (_BYTE)v10 == 8 || (unsigned int)v14 <= 1 )
    {
      v16 = 1056;
      return sub_30315D0(a2, a4, v16, v14, v7, v10);
    }
    if ( (unsigned __int8)v10 == 5 )
    {
      v16 = 1055;
      return sub_30315D0(a2, a4, v16, v14, v7, v10);
    }
LABEL_80:
    BUG();
  }
  v49 = *(_QWORD *)(a2 + 80);
  v86 = v49;
  if ( v49 )
  {
    sub_B96E90((__int64)&v86, v49, 1);
    v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
    LODWORD(v7) = *(_DWORD *)(v6 + 32);
  }
  v87 = *(_DWORD *)(a2 + 72);
  v50 = *(_QWORD **)(v6 + 24);
  if ( (unsigned int)v7 > 0x40 )
    v50 = (_QWORD *)*v50;
  v51 = (int)v50;
  v52 = (unsigned __int8)v50 & 0xF;
  if ( v52 <= 3u )
  {
    if ( v52 <= 1u )
      goto LABEL_80;
  }
  else if ( v52 != 8 )
  {
    goto LABEL_80;
  }
  v83 = &v88;
  v88 = v90;
  v89 = 0x400000000LL;
  v53 = sub_3400BD0(a4, v51, (unsigned int)&v86, 8, 0, 1, 0);
  v54 = (unsigned int)v89;
  v56 = v55;
  v57 = (unsigned int)v89 + 1LL;
  if ( v57 > HIDWORD(v89) )
  {
    v82.m128i_i64[0] = v53;
    v82.m128i_i64[1] = v56;
    sub_C8D5F0((__int64)v83, v90, v57, 0x10u, v53, v56);
    v54 = (unsigned int)v89;
    v56 = v82.m128i_i64[1];
    v53 = v82.m128i_i64[0];
  }
  v58 = (__int64 *)&v88[16 * v54];
  *v58 = v53;
  v58[1] = v56;
  v59 = *(_QWORD *)(a2 + 40);
  v60 = HIDWORD(v89);
  LODWORD(v89) = v89 + 1;
  v61 = (unsigned int)v89;
  v62 = _mm_loadu_si128((const __m128i *)(v59 + 120));
  if ( (unsigned __int64)(unsigned int)v89 + 1 > HIDWORD(v89) )
  {
    v82 = v62;
    sub_C8D5F0((__int64)v83, v90, (unsigned int)v89 + 1LL, 0x10u, v53, v56);
    v61 = (unsigned int)v89;
    v62 = _mm_load_si128(&v82);
  }
  *(__m128i *)&v88[16 * v61] = v62;
  LODWORD(v89) = v89 + 1;
  v63 = sub_33ED250(a4, 6, 0, v60);
  v80 = sub_33E66D0(a4, 1052, (unsigned int)&v86, v63, v64, v65, (__int64)v88, (unsigned int)v89);
  v67 = sub_33FAF80(a4, 214, (unsigned int)&v86, 7, 0, v66, (unsigned __int64)v80);
  v69 = sub_302EAB0(
          a4,
          (int)&v86,
          v67,
          v68,
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 160LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 168LL),
          BYTE1(v51) & 1);
  if ( v88 != v90 )
    _libc_free((unsigned __int64)v88);
  if ( v86 )
    sub_B91220((__int64)&v86, v86);
  return v69;
}
