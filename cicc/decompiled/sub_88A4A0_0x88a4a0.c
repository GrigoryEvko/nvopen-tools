// Function: sub_88A4A0
// Address: 0x88a4a0
//
unsigned int *sub_88A4A0()
{
  _QWORD *v0; // rsi
  _QWORD *v1; // rsi
  _QWORD *v2; // rsi
  int v3; // eax
  int v4; // eax
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // rsi
  unsigned int *result; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // rsi
  char **v23; // rbx
  int v24; // r13d
  unsigned __int16 v25; // r9
  char *v26; // rdx
  unsigned __int16 v27; // cx
  char **v28; // rbx
  int v29; // r13d
  unsigned __int16 v30; // r9
  unsigned __int16 v31; // si
  unsigned __int16 v32; // cx
  unsigned __int16 v33; // ax
  char **v34; // r13
  int v35; // ebx
  unsigned __int16 v36; // r9
  unsigned __int16 v37; // si
  unsigned __int16 v38; // cx
  unsigned __int16 v39; // ax
  int v40; // r15d
  int v41; // r13d
  __int64 j; // rbx
  unsigned __int16 v43; // r9
  unsigned __int16 v44; // si
  unsigned __int16 v45; // cx
  unsigned __int16 v46; // ax
  __int64 v47; // r13
  _QWORD *v48; // rdi
  char **v49; // rbx
  int v50; // r13d
  unsigned __int16 v51; // si
  unsigned __int16 v52; // r9
  unsigned __int16 v53; // cx
  unsigned __int16 v54; // ax
  int v55; // r13d
  int v56; // ebx
  __int64 i; // r14
  unsigned __int16 v58; // r9
  unsigned __int16 v59; // si
  unsigned __int16 v60; // ax
  unsigned __int16 v61; // cx
  _QWORD *v62; // rax
  _QWORD *v63; // rax
  _QWORD *v64; // rax
  _QWORD *v65; // rax
  _QWORD *v66; // rax
  _QWORD *v67; // rax
  _QWORD *v68; // rax
  _QWORD *v69; // rax
  _QWORD *v70; // rax
  _QWORD *v71; // rax
  _QWORD *v72; // rax
  _QWORD *v73; // rax
  char *v74; // rax
  _QWORD *v75; // rax
  _QWORD *v76; // rax
  _QWORD *v77; // rax
  __int64 v78; // rax
  _QWORD *v79; // rsi
  __int64 v80; // rax
  _QWORD *v81; // rax
  __int64 v82; // rax
  _QWORD *v83; // rax
  _QWORD *v84; // rsi
  _QWORD *v85; // rax
  _QWORD *v86; // rax
  _QWORD *v87; // rax
  _QWORD *v88; // rax
  _QWORD *v89; // rax
  _QWORD *v90; // rax
  _QWORD *v91; // rax
  _QWORD *v92; // rax
  _QWORD *v93; // rax
  _QWORD *v94; // rax
  _QWORD *v95; // rax
  _QWORD *v96; // rax
  _QWORD *v97; // rax
  _QWORD *v98; // rax
  _QWORD *v99; // rsi
  __int64 v100; // rsi
  __int64 v101; // rdi
  _QWORD *v102; // rax
  _QWORD *v103; // rax
  _QWORD *v104; // rax
  _QWORD *v105; // rsi
  _QWORD *v106; // rax
  __int64 v107; // r14
  _QWORD *v108; // rax
  _QWORD *v109; // rax
  _QWORD *v110; // rax
  _QWORD *v111; // rax
  _QWORD *v112; // rax
  char *v113; // [rsp-8h] [rbp-68h]
  char *v114; // [rsp+10h] [rbp-50h] BYREF
  const char *v115; // [rsp+18h] [rbp-48h]
  const char *v116; // [rsp+20h] [rbp-40h]
  const char *v117; // [rsp+28h] [rbp-38h]

  if ( unk_4D0428C )
  {
    if ( !HIDWORD(qword_4F077B4) || (_DWORD)qword_4F077B4 || qword_4F077A8 <= 0x1FBCFu )
      v0 = sub_72C610(0);
    else
      v0 = sub_72C610(0xAu);
    sub_888EB0("_Float16", (__int64)v0);
  }
  if ( dword_4D04288 )
  {
    v64 = sub_72C610(unk_4B6D467);
    sub_888EB0("__float80", (__int64)v64);
  }
  if ( dword_4D04284 )
  {
    v62 = sub_72C610(unk_4B6D466);
    sub_888EB0("__float128", (__int64)v62);
    if ( unk_4D046C0 )
    {
      v63 = sub_72C610(unk_4B6D466);
      sub_888EB0("__ieee128", (__int64)v63);
    }
  }
  if ( HIDWORD(qword_4F077B4) )
  {
    if ( dword_4D0455C && unk_4D04600 > 0x30DA3u )
    {
      if ( !unk_4D04290 )
        goto LABEL_15;
    }
    else
    {
      sub_88A1F0();
      if ( !unk_4D04290 )
        goto LABEL_15;
    }
    v21 = sub_72BA30(0xBu);
    sub_888EB0("__int128_t", (__int64)v21);
    v22 = sub_72BA30(0xCu);
    sub_888EB0("__uint128_t", (__int64)v22);
LABEL_15:
    if ( qword_4F077A8 > 0x9C3Fu )
    {
      if ( (_DWORD)qword_4F077B4 )
      {
LABEL_17:
        v1 = sub_72C610(1u);
        sub_888EB0("__fp16", (__int64)v1);
        if ( (_DWORD)qword_4F077B4 )
        {
          if ( qword_4F077A0 <= 0x1ADAFu )
            goto LABEL_52;
          goto LABEL_19;
        }
        if ( !HIDWORD(qword_4F077B4) )
          goto LABEL_25;
LABEL_81:
        if ( qword_4F077A8 <= 0x1869Fu )
        {
          v3 = qword_4F077B4;
LABEL_20:
          if ( v3 )
            goto LABEL_52;
          if ( qword_4F077A8 <= 0xC34Fu )
            goto LABEL_27;
          if ( (_DWORD)qword_4F06A7C )
          {
            sub_889530(byte_4B7E000);
            sub_8894B0(byte_4B7DE60, 2);
            sub_8894B0(byte_4B7DCC0, 3);
            v102 = sub_72BA30(2u);
            sub_888EB0("__Poly8_t", (__int64)v102);
            v103 = sub_72BA30(4u);
            sub_888EB0("__Poly16_t", (__int64)v103);
            v104 = sub_72BA30(8u);
            sub_888EB0("__Poly64_t", (__int64)v104);
            v105 = sub_72BA30(0xCu);
            sub_888EB0("__Poly128_t", (__int64)v105);
            v4 = qword_4F077B4;
LABEL_26:
            if ( !v4 )
            {
LABEL_27:
              if ( !HIDWORD(qword_4F077B4) )
                goto LABEL_40;
              if ( qword_4F077A8 <= 0x1869Fu )
                goto LABEL_31;
              goto LABEL_29;
            }
LABEL_52:
            if ( qword_4F077A0 <= 0x1869Fu )
            {
              if ( !dword_4D03FE8[0] )
                goto LABEL_41;
              goto LABEL_54;
            }
LABEL_29:
            if ( (_DWORD)qword_4F06A7C )
            {
              v114 = "__SVInt8_t";
              v115 = "__clang_svint8x2_t";
              v116 = "__clang_svint8x3_t";
              v117 = "__clang_svint8x4_t";
              v65 = sub_72BA30(1u);
              sub_888F80((__int64)v65, &v114);
              v114 = "__SVUint8_t";
              v115 = "__clang_svuint8x2_t";
              v116 = "__clang_svuint8x3_t";
              v117 = "__clang_svuint8x4_t";
              v66 = sub_72BA30(2u);
              sub_888F80((__int64)v66, &v114);
              v114 = "__SVInt16_t";
              v115 = "__clang_svint16x2_t";
              v116 = "__clang_svint16x3_t";
              v117 = "__clang_svint16x4_t";
              v67 = sub_72BA30(3u);
              sub_888F80((__int64)v67, &v114);
              v114 = "__SVUint16_t";
              v115 = "__clang_svuint16x2_t";
              v116 = "__clang_svuint16x3_t";
              v117 = "__clang_svuint16x4_t";
              v68 = sub_72BA30(4u);
              sub_888F80((__int64)v68, &v114);
              v114 = "__SVInt32_t";
              v115 = "__clang_svint32x2_t";
              v116 = "__clang_svint32x3_t";
              v117 = "__clang_svint32x4_t";
              v69 = sub_72BA30(5u);
              sub_888F80((__int64)v69, &v114);
              v114 = "__SVUint32_t";
              v115 = "__clang_svuint32x2_t";
              v116 = "__clang_svuint32x3_t";
              v117 = "__clang_svuint32x4_t";
              v70 = sub_72BA30(6u);
              sub_888F80((__int64)v70, &v114);
              v114 = "__SVInt64_t";
              v115 = "__clang_svint64x2_t";
              v116 = "__clang_svint64x3_t";
              v117 = "__clang_svint64x4_t";
              v71 = sub_72BA30(7u);
              sub_888F80((__int64)v71, &v114);
              v114 = "__SVUint64_t";
              v115 = "__clang_svuint64x2_t";
              v116 = "__clang_svuint64x3_t";
              v117 = "__clang_svuint64x4_t";
              v72 = sub_72BA30(8u);
              sub_888F80((__int64)v72, &v114);
              v114 = "__SVFloat16_t";
              v115 = "__clang_svfloat16x2_t";
              v116 = "__clang_svfloat16x3_t";
              v117 = "__clang_svfloat16x4_t";
              v73 = sub_72C610(1u);
              sub_888F80((__int64)v73, &v114);
              v74 = "__SVBfloat16_t";
              if ( (_DWORD)qword_4F077B4 && qword_4F077A0 <= 0x2BF1Fu )
                v74 = "__SVBFloat16_t";
              v114 = v74;
              v115 = "__clang_svbfloat16x2_t";
              v116 = "__clang_svbfloat16x3_t";
              v117 = "__clang_svbfloat16x4_t";
              v75 = sub_72C610(9u);
              sub_888F80((__int64)v75, &v114);
              v114 = "__SVFloat32_t";
              v115 = "__clang_svfloat32x2_t";
              v116 = "__clang_svfloat32x3_t";
              v117 = "__clang_svfloat32x4_t";
              v76 = sub_72C610(2u);
              sub_888F80((__int64)v76, &v114);
              v114 = "__SVFloat64_t";
              v115 = "__clang_svfloat64x2_t";
              v116 = "__clang_svfloat64x3_t";
              v117 = "__clang_svfloat64x4_t";
              v77 = sub_72C610(4u);
              sub_888F80((__int64)v77, &v114);
              v78 = sub_72C390();
              v79 = sub_72B620(v78, 1);
              sub_888EB0("__SVBool_t", (__int64)v79);
              if ( !(_DWORD)qword_4F077B4 )
                goto LABEL_115;
              if ( qword_4F077A0 <= 0x2980Fu )
                goto LABEL_40;
              v80 = sub_72C390();
              v81 = sub_72B620(v80, 2);
              sub_888EB0("__clang_svboolx2_t", (__int64)v81);
              v82 = sub_72C390();
              v83 = sub_72B620(v82, 4);
              sub_888EB0("__clang_svboolx4_t", (__int64)v83);
              v84 = sub_72B660();
              sub_888EB0("__SVCount_t", (__int64)v84);
              if ( (_DWORD)qword_4F077B4 )
              {
                if ( qword_4F077A0 <= 0x30D3Fu )
                  goto LABEL_40;
              }
              else
              {
LABEL_115:
                if ( !HIDWORD(qword_4F077B4) )
                  goto LABEL_40;
                if ( qword_4F077A8 <= 0x249EFu )
                  goto LABEL_31;
              }
              v100 = sub_72B690();
              sub_888EB0("__mfp8", v100);
              v114 = "__SVMfloat8_t";
              v115 = "__clang_svmfloat8x2_t";
              v116 = "__clang_svmfloat8x3_t";
              v117 = "__clang_svmfloat8x4_t";
              v101 = sub_72B690();
              sub_888F80(v101, &v114);
            }
            if ( !HIDWORD(qword_4F077B4) )
              goto LABEL_40;
LABEL_31:
            if ( !(_DWORD)qword_4F077B4 && qword_4F077A8 > 0x9F5Fu )
            {
              if ( unk_4F06A78 )
              {
                if ( qword_4F077A8 > 0x1869Fu )
                {
                  v106 = sub_72C610(9u);
                  sub_888EB0("__builtin_neon_bf", (__int64)v106);
                }
                else if ( qword_4F077A8 <= 0xEA5Fu )
                {
                  v110 = sub_72C610(1u);
                  sub_888EB0("__builtin_neon_hf", (__int64)v110);
                }
                v85 = sub_72C610(2u);
                sub_888EB0("__builtin_neon_sf", (__int64)v85);
                v86 = sub_72C610(4u);
                sub_888EB0("__builtin_neon_df", (__int64)v86);
                v87 = sub_72BA30(1u);
                sub_888EB0("__builtin_neon_qi", (__int64)v87);
                v88 = sub_72BA30(2u);
                sub_888EB0("__builtin_neon_uqi", (__int64)v88);
                v89 = sub_72BA30(3u);
                sub_888EB0("__builtin_neon_hi", (__int64)v89);
                v90 = sub_72BA30(4u);
                sub_888EB0("__builtin_neon_uhi", (__int64)v90);
                v91 = sub_72BA30(5u);
                sub_888EB0("__builtin_neon_si", (__int64)v91);
                v92 = sub_72BA30(6u);
                sub_888EB0("__builtin_neon_usi", (__int64)v92);
                v93 = sub_72BA30(9u);
                sub_888EB0("__builtin_neon_di", (__int64)v93);
                v94 = sub_72BA30(0xAu);
                sub_888EB0("__builtin_neon_udi", (__int64)v94);
                v95 = sub_72BA30(1u);
                sub_888EB0("__builtin_neon_poly8", (__int64)v95);
                v96 = sub_72BA30(3u);
                sub_888EB0("__builtin_neon_poly16", (__int64)v96);
                v97 = sub_72BA30(0xAu);
                sub_888EB0("__builtin_neon_poly64", (__int64)v97);
                v98 = sub_72BA30(0xCu);
                sub_888EB0("__builtin_neon_poly128", (__int64)v98);
                v99 = sub_72BA30(0xCu);
                sub_888EB0("__builtin_neon_uti", (__int64)v99);
                sub_888F20(byte_4B7DBA0);
              }
              else if ( (_DWORD)qword_4F06A7C )
              {
                if ( qword_4F077A8 <= 0x249EFu
                  || (v107 = sub_72B690(),
                      v108 = sub_72B5A0(v107, 8, 2),
                      sub_888EB0("__Mfloat8x8_t", (__int64)v108),
                      v109 = sub_72B5A0(v107, 16, 2),
                      sub_888EB0("__Mfloat8x16_t", (__int64)v109),
                      HIDWORD(qword_4F077B4))
                  && !(_DWORD)qword_4F077B4 )
                {
                  if ( qword_4F077A8 <= 0x1869Fu
                    || (v111 = sub_72C610(9u),
                        sub_888EB0("__builtin_aarch64_simd_bf", (__int64)v111),
                        HIDWORD(qword_4F077B4))
                    && !(_DWORD)qword_4F077B4 )
                  {
                    if ( qword_4F077A8 > 0xEA5Fu )
                    {
                      v112 = sub_72C610(1u);
                      sub_888EB0("__builtin_aarch64_simd_hf", (__int64)v112);
                    }
                  }
                }
                v5 = sub_72C610(2u);
                sub_888EB0("__builtin_aarch64_simd_sf", (__int64)v5);
                v6 = sub_72C610(4u);
                sub_888EB0("__builtin_aarch64_simd_df", (__int64)v6);
                v7 = sub_72BA30(1u);
                sub_888EB0("__builtin_aarch64_simd_qi", (__int64)v7);
                v8 = sub_72BA30(2u);
                sub_888EB0("__builtin_aarch64_simd_uqi", (__int64)v8);
                v9 = sub_72BA30(3u);
                sub_888EB0("__builtin_aarch64_simd_hi", (__int64)v9);
                v10 = sub_72BA30(4u);
                sub_888EB0("__builtin_aarch64_simd_uhi", (__int64)v10);
                v11 = sub_72BA30(5u);
                sub_888EB0("__builtin_aarch64_simd_si", (__int64)v11);
                v12 = sub_72BA30(6u);
                sub_888EB0("__builtin_aarch64_simd_usi", (__int64)v12);
                v13 = sub_72BA30(7u);
                sub_888EB0("__builtin_aarch64_simd_di", (__int64)v13);
                v14 = sub_72BA30(8u);
                sub_888EB0("__builtin_aarch64_simd_udi", (__int64)v14);
                v15 = sub_72BA30(2u);
                sub_888EB0("__builtin_aarch64_simd_poly8", (__int64)v15);
                v16 = sub_72BA30(4u);
                sub_888EB0("__builtin_aarch64_simd_poly16", (__int64)v16);
                v17 = sub_72BA30(8u);
                sub_888EB0("__builtin_aarch64_simd_poly64", (__int64)v17);
                v18 = sub_72BA30(0xCu);
                sub_888EB0("__builtin_aarch64_simd_poly128", (__int64)v18);
                v19 = sub_72BA30(0xBu);
                sub_888EB0("__builtin_aarch64_simd_ti", (__int64)v19);
                sub_888F20(byte_4B7DB40);
              }
            }
            goto LABEL_40;
          }
          if ( !unk_4F06A78 )
            goto LABEL_27;
          sub_889530(byte_4B7DF80);
          sub_8894B0(byte_4B7DD40, 2);
          sub_8894B0(byte_4B7DC40, 3);
LABEL_25:
          v4 = qword_4F077B4;
          goto LABEL_26;
        }
LABEL_19:
        v2 = sub_72C610(9u);
        sub_888EB0("__bf16", (__int64)v2);
        v3 = qword_4F077B4;
        if ( !HIDWORD(qword_4F077B4) )
        {
          if ( !(_DWORD)qword_4F077B4 )
            goto LABEL_40;
          goto LABEL_52;
        }
        goto LABEL_20;
      }
      v48 = sub_736B10(10, (__int64)"_IO_FILE");
      sub_736B60((__int64)v48, 0, &dword_4F077C8);
    }
    if ( !(_DWORD)qword_4F077B4 )
    {
      if ( !HIDWORD(qword_4F077B4) )
        goto LABEL_25;
      if ( qword_4F077A8 <= 0xEA5Fu )
        goto LABEL_81;
    }
    goto LABEL_17;
  }
LABEL_40:
  if ( !dword_4D03FE8[0] )
    goto LABEL_41;
LABEL_54:
  if ( dword_4D0455C && unk_4D04600 > 0x2E693u && unk_4D045F8 )
    qsort(&unk_4B7E0A0, 0x6C6u, 2u, (__compar_fn_t)sub_888600);
  v23 = off_4A52080;
  v24 = 0;
  while ( *v23 )
  {
    while ( !(unsigned int)sub_8891F0(0, v23[1], 0, *((_WORD *)v23 + 12)) )
    {
      v23 += 4;
      ++v24;
      if ( !*v23 )
        goto LABEL_60;
    }
    v25 = *((_WORD *)v23 + 12);
    v26 = v23[1];
    v113 = v23[2];
    v27 = v24;
    v23 += 4;
    ++v24;
    sub_889BE0(*(v23 - 4), 0, v26, v27, 9u, v25, 0, v113);
  }
LABEL_60:
  v28 = &off_4ADADA0;
  v29 = 0;
  if ( off_4ADADA0 )
  {
    do
    {
      while ( !(unsigned int)sub_8891F0(*((_WORD *)v28 + 4), 0, 0, *((_WORD *)v28 + 6)) )
      {
        v28 += 2;
        ++v29;
        if ( !*v28 )
          goto LABEL_65;
      }
      v30 = *((_WORD *)v28 + 6);
      v31 = *((_WORD *)v28 + 4);
      v32 = v29;
      v33 = *((_WORD *)v28 + 5);
      v28 += 2;
      ++v29;
      sub_889BE0(*(v28 - 2), v31, 0, v32, 1u, v30, v33, 0);
    }
    while ( *v28 );
  }
LABEL_65:
  if ( !unk_4F06A78 )
    goto LABEL_66;
  v49 = &off_4AD51E0;
  v50 = 0;
  if ( off_4AD51E0 )
  {
    do
    {
      while ( !(unsigned int)sub_8891F0(*((_WORD *)v49 + 4), 0, 0, *((_WORD *)v49 + 6)) )
      {
        v49 += 2;
        ++v50;
        if ( !*v49 )
          goto LABEL_88;
      }
      v51 = *((_WORD *)v49 + 4);
      v52 = *((_WORD *)v49 + 6);
      v53 = v50;
      v54 = *((_WORD *)v49 + 5);
      v49 += 2;
      ++v50;
      sub_889BE0(*(v49 - 2), v51, 0, v53, 2u, v52, v54, 0);
    }
    while ( *v49 );
  }
LABEL_88:
  v55 = 3 - ((qword_4F06A7C == 0) - 1);
  v56 = 0;
  for ( i = qword_4A598E0[v55]; *(_QWORD *)i; sub_889BE0(*(char **)(i - 16), v59, 0, v61, v55, v58, v60, 0) )
  {
    while ( !(unsigned int)sub_8891F0(*(_WORD *)(i + 8), 0, 0, *(_WORD *)(i + 12)) )
    {
      i += 16;
      ++v56;
      if ( !*(_QWORD *)i )
        goto LABEL_93;
    }
    v58 = *(_WORD *)(i + 12);
    v59 = *(_WORD *)(i + 8);
    v60 = *(_WORD *)(i + 10);
    v61 = v56;
    i += 16;
    ++v56;
  }
LABEL_93:
  if ( !unk_4F06A78 && !(_DWORD)qword_4F06A7C )
  {
LABEL_66:
    v34 = &off_4A5ABE0;
    v35 = 0;
    if ( off_4A5ABE0 )
    {
      do
      {
        while ( !(unsigned int)sub_8891F0(*((_WORD *)v34 + 4), 0, 0, *((_WORD *)v34 + 6)) )
        {
          v34 += 2;
          ++v35;
          if ( !*v34 )
            goto LABEL_71;
        }
        v36 = *((_WORD *)v34 + 6);
        v37 = *((_WORD *)v34 + 4);
        v38 = v35;
        v39 = *((_WORD *)v34 + 5);
        ++v35;
        v34 += 2;
        sub_889BE0(*(v34 - 2), v37, 0, v38, 5u, v36, v39, 0);
      }
      while ( *v34 );
    }
LABEL_71:
    v40 = 6 - ((qword_4F06A7C == 0) - 1);
    v41 = 0;
    for ( j = qword_4A598E0[v40]; *(_QWORD *)j; sub_889BE0(*(char **)(j - 16), v44, 0, v45, v40, v43, v46, 0) )
    {
      while ( !(unsigned int)sub_8891F0(*(_WORD *)(j + 8), 0, 0, *(_WORD *)(j + 12)) )
      {
        j += 16;
        ++v41;
        if ( !*(_QWORD *)j )
          goto LABEL_76;
      }
      v43 = *(_WORD *)(j + 12);
      v44 = *(_WORD *)(j + 8);
      v45 = v41;
      v46 = *(_WORD *)(j + 10);
      j += 16;
      ++v41;
    }
  }
LABEL_76:
  unk_4D041F8 = 1;
  v47 = qword_4D03FF0;
  sub_8D0910(qword_4D03FD0);
  sub_889FF0("__cxa_vec_ctor");
  sub_889FF0("__cxa_vec_cctor");
  sub_889FF0("__cxa_vec_dtor");
  sub_889FF0("__cxa_vec_new2");
  sub_889FF0("__cxa_vec_new");
  sub_889FF0("__cxa_vec_new3");
  sub_889FF0("__cxa_vec_delete2");
  sub_889FF0("__cxa_vec_delete");
  sub_889FF0("__cxa_vec_delete3");
  sub_889FF0("__gen_nvvm_memcpy_aligned1");
  sub_889FF0("__gen_nvvm_memcpy_aligned2");
  sub_889FF0("__gen_nvvm_memcpy_aligned4");
  sub_889FF0("__gen_nvvm_memcpy_aligned8");
  sub_889FF0("__gen_nvvm_memcpy_aligned16");
  sub_889FF0("__gen_nvvm_memcpy");
  sub_889FF0("__gen_nvvm_memset");
  sub_8D0910(v47);
LABEL_41:
  result = &dword_4D04408;
  if ( dword_4D04408 )
  {
    if ( (_DWORD)qword_4F077B4 && (result = (unsigned int *)sub_879B10(), (_DWORD)qword_4F077B4) )
    {
      result = &dword_4F077C4;
      if ( dword_4F077C4 == 2 )
      {
        result = (unsigned int *)&qword_4F077A0;
        if ( qword_4F077A0 > 0x78B3u )
          return (unsigned int *)sub_879B90();
      }
    }
    else if ( HIDWORD(qword_4F077B4) )
    {
      result = (unsigned int *)&qword_4F077A8;
      if ( qword_4F077A8 > 0x222DFu )
        return (unsigned int *)sub_879B90();
    }
  }
  return result;
}
