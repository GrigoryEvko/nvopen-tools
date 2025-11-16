// Function: sub_CD9990
// Address: 0xcd9990
//
void __fastcall sub_CD9990(__int64 a1, __int64 a2)
{
  char v2; // al
  _BOOL8 v3; // rcx
  char v4; // al
  _BOOL8 v5; // rcx
  char v6; // al
  char v7; // al
  char v8; // al
  char v9; // al
  char v10; // al
  char v11; // al
  char v12; // al
  unsigned __int8 v13; // al
  char v14; // al
  unsigned __int8 v15; // al
  char v16; // al
  char v17; // al
  unsigned __int8 v18; // al
  char v19; // al
  char v20; // al
  char v21; // al
  char v22; // al
  char v23; // al
  unsigned __int8 v24; // al
  char v25; // al
  char v26; // al
  char v27; // al
  unsigned __int8 v28; // al
  char v29; // al
  unsigned __int8 v30; // al
  char v31; // al
  char v32; // al
  char v33; // al
  char v34; // al
  char v35; // al
  unsigned __int8 v36; // al
  int v37; // r15d
  unsigned __int8 v38; // al
  char v39; // al
  _BOOL8 v40; // rcx
  char v41; // al
  _BOOL8 v42; // rcx
  char v43; // al
  _BOOL8 v44; // rcx
  __int8 v45; // dl
  int v46; // eax
  char v47; // al
  _BOOL8 v48; // rcx
  __int32 v49; // r15d
  char v50; // al
  _BOOL8 v51; // rcx
  char v52; // al
  _BOOL8 v53; // rcx
  char v54; // al
  _BOOL8 v55; // rcx
  char v56; // al
  _BOOL8 v57; // rcx
  __int64 v58; // r15
  __int64 v59; // r15
  unsigned __int8 v60; // al
  bool v61; // zf
  __int64 v62; // rax
  char v63; // al
  _BOOL8 v64; // rdx
  char v65; // al
  __int64 v66; // rax
  char v67; // al
  _BOOL8 v68; // rdx
  char v69; // al
  __int64 v70; // rax
  char v71; // al
  _BOOL8 v72; // rdx
  char v73; // al
  __int64 v74; // rax
  char v75; // al
  _BOOL8 v76; // rdx
  char v77; // al
  char v78; // al
  _BOOL8 v79; // rdx
  char v80; // al
  _BOOL8 v81; // rdx
  char v82; // al
  _BOOL8 v83; // rcx
  char v84; // al
  char v85; // al
  char v86; // al
  _BOOL8 v87; // rcx
  char v88; // al
  char v89; // al
  char v90; // al
  _BOOL8 v91; // rcx
  char v92; // al
  char v93; // al
  _BOOL8 v94; // rcx
  unsigned int v95; // eax
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // rax
  __m128i v99; // xmm1
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // rax
  __m128i v103; // xmm4
  unsigned __int8 (__fastcall *v104)(__int64, const char *, _QWORD); // [rsp+0h] [rbp-B0h]
  __int64 (__fastcall *v105)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  __int64 (__fastcall *v106)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  __int64 (__fastcall *v107)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  __int64 (__fastcall *v108)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  unsigned __int8 (__fastcall *v109)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  unsigned __int8 (__fastcall *v110)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  char v111; // [rsp+2Eh] [rbp-82h] BYREF
  char v112; // [rsp+2Fh] [rbp-81h] BYREF
  __int32 v113; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v114; // [rsp+34h] [rbp-7Ch] BYREF
  __int64 v115; // [rsp+38h] [rbp-78h] BYREF
  __m128i v116; // [rsp+40h] [rbp-70h] BYREF
  __m128i v117; // [rsp+50h] [rbp-60h] BYREF
  __m128i v118; // [rsp+60h] [rbp-50h] BYREF
  __int64 v119; // [rsp+70h] [rbp-40h]

  sub_CD8610(a1, a2);
  sub_CCE100(a1, a2 + 48);
  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = *(_DWORD *)(a2 + 432) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CooperativeVectorInfo",
         0,
         v3,
         &v112) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
    v114 = *(_BYTE *)(a2 + 432) & 1;
    v82 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v83 = 0;
    if ( v82 )
      v83 = v114 == 0;
    v84 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int32 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
            a1,
            "U.DisableSAMRAM",
            0,
            v83,
            &v113,
            &v116);
    if ( v84 )
    {
      sub_CCC2C0(a1, &v114);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v116.m128i_i64[0]);
      v84 = v114 & 1;
    }
    else if ( !(_BYTE)v113 )
    {
      v84 = v114 & 1;
    }
    v85 = v84 & 1 | *(_BYTE *)(a2 + 432) & 0xFE;
    *(_BYTE *)(a2 + 432) = v85;
    v114 = (v85 & 2) != 0;
    v86 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v87 = 0;
    if ( v86 )
      v87 = v114 == 0;
    v88 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int32 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
            a1,
            "U.DisableLayerFusion",
            0,
            v87,
            &v113,
            &v116);
    if ( v88 )
    {
      sub_CCC2C0(a1, &v114);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v116.m128i_i64[0]);
      v88 = v114 & 1;
    }
    else if ( !(_BYTE)v113 )
    {
      v88 = v114 & 1;
    }
    v89 = (2 * (v88 & 1)) | *(_BYTE *)(a2 + 432) & 0xFD;
    *(_BYTE *)(a2 + 432) = v89;
    v114 = (v89 & 4) != 0;
    v90 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v91 = 0;
    if ( v90 )
      v91 = v114 == 0;
    v92 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int32 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
            a1,
            "U.DisableI32MatrixLayout",
            0,
            v91,
            &v113,
            &v116);
    if ( v92 )
    {
      sub_CCC2C0(a1, &v114);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v116.m128i_i64[0]);
      v92 = v114 & 1;
    }
    else if ( !(_BYTE)v113 )
    {
      v92 = v114 & 1;
    }
    *(_BYTE *)(a2 + 432) = *(_BYTE *)(a2 + 432) & 0xFB | (4 * (v92 & 1));
    v114 = *(_DWORD *)(a2 + 432) >> 3;
    v93 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v94 = 0;
    if ( v93 )
      v94 = v114 == 0;
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int32 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
           a1,
           "U.Reserved",
           0,
           v94,
           &v113,
           &v116) )
    {
      sub_CCC2C0(a1, &v114);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v116.m128i_i64[0]);
      v95 = v114 & 0x1FFFFFFF;
    }
    else
    {
      v95 = 0;
      if ( !(_BYTE)v113 )
        v95 = v114 & 0x1FFFFFFF;
    }
    *(_DWORD *)(a2 + 432) = (8 * v95) | *(_DWORD *)(a2 + 432) & 7;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v115);
  }
  else if ( v112 )
  {
    *(_DWORD *)(a2 + 432) = 0;
  }
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = *(_DWORD *)(a2 + 64) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CodegenSel",
         0,
         v5,
         &v115,
         &v116) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v109 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v78 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v79 = 0;
    if ( v78 )
      v79 = *(_DWORD *)(a2 + 64) == 0;
    if ( v109(a1, "D2IR", v79) )
      *(_DWORD *)(a2 + 64) = 0;
    v110 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v80 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v81 = 0;
    if ( v80 )
      v81 = *(_DWORD *)(a2 + 64) == 1;
    if ( v110(a1, "Omega", v81) )
      *(_DWORD *)(a2 + 64) = 1;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v116.m128i_i64[0]);
  }
  else if ( (_BYTE)v115 )
  {
    *(_DWORD *)(a2 + 64) = 0;
  }
  v6 = *(_BYTE *)(a2 + 68);
  v113 = 0;
  v114 = 1;
  v111 = 0;
  v116.m128i_i32[0] = (v6 & 2) != 0;
  sub_CCC4A0(a1, (__int64)"PromoteHalf", (unsigned int *)&v116, &v114, 0);
  v7 = (2 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 68) & 0xFD;
  *(_BYTE *)(a2 + 68) = v7;
  v116.m128i_i32[0] = (v7 & 4) != 0;
  sub_CCC4A0(a1, (__int64)"IgnoreRndFtzOnF32F16Conv", (unsigned int *)&v116, &v113, 0);
  v8 = (4 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 68) & 0xFB;
  *(_BYTE *)(a2 + 68) = v8;
  v116.m128i_i32[0] = v8 & 1;
  sub_CCC4A0(a1, (__int64)"PromoteFixed", (unsigned int *)&v116, &v113, 0);
  v9 = v116.m128i_i8[0] & 1 | *(_BYTE *)(a2 + 68) & 0xFE;
  *(_BYTE *)(a2 + 68) = v9;
  v116.m128i_i32[0] = (v9 & 8) != 0;
  sub_CCC4A0(a1, (__int64)"UsePIXBAR", (unsigned int *)&v116, &v113, 0);
  v10 = (8 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 68) & 0xF7;
  *(_BYTE *)(a2 + 68) = v10;
  v116.m128i_i32[0] = (v10 & 0x10) != 0;
  sub_CCC4A0(a1, (__int64)"TLDUsesTLD4CompatibleSampler", (unsigned int *)&v116, &v113, 0);
  v11 = (16 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 68) & 0xEF;
  *(_BYTE *)(a2 + 68) = v11;
  v116.m128i_i32[0] = (v11 & 0x20) != 0;
  sub_CCC4A0(a1, (__int64)"VSIsVREnabled", (unsigned int *)&v116, &v113, 0);
  v12 = (32 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 68) & 0xDF;
  *(_BYTE *)(a2 + 68) = v12;
  v116.m128i_i32[0] = (v12 & 0x40) != 0;
  sub_CCC4A0(a1, (__int64)"VSIsLastVTGStage", (unsigned int *)&v116, &v113, 0);
  v13 = ((v116.m128i_i8[0] & 1) << 6) | *(_BYTE *)(a2 + 68) & 0xBF;
  *(_BYTE *)(a2 + 68) = v13;
  v116.m128i_i32[0] = v13 >> 7;
  sub_CCC4A0(a1, (__int64)"EnableZeroCoverageKill", (unsigned int *)&v116, &v113, 0);
  *(_BYTE *)(a2 + 68) = (v116.m128i_i8[0] << 7) | *(_BYTE *)(a2 + 68) & 0x7F;
  v116.m128i_i32[0] = *(_BYTE *)(a2 + 69) & 1;
  sub_CCC4A0(a1, (__int64)"EnablePartialBindlessTextures", (unsigned int *)&v116, &v113, 0);
  v14 = v116.m128i_i8[0] & 1 | *(_BYTE *)(a2 + 69) & 0xFE;
  *(_BYTE *)(a2 + 69) = v14;
  v116.m128i_i32[0] = (v14 & 4) != 0;
  sub_CCC4A0(a1, (__int64)"DisableKeplerLUWar", (unsigned int *)&v116, &v113, 0);
  v15 = (4 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 69) & 0xFB;
  *(_BYTE *)(a2 + 69) = v15;
  v116.m128i_i32[0] = (v15 >> 3) & 3;
  sub_CCC4A0(a1, (__int64)"ReorderCSE", (unsigned int *)&v116, &v113, 0);
  v16 = (8 * (v116.m128i_i8[0] & 3)) | *(_BYTE *)(a2 + 69) & 0xE7;
  *(_BYTE *)(a2 + 69) = v16;
  v116.m128i_i32[0] = (v16 & 0x20) != 0;
  sub_CCC4A0(a1, (__int64)"InitUninitialized", (unsigned int *)&v116, &v113, 0);
  v17 = (32 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 69) & 0xDF;
  *(_BYTE *)(a2 + 69) = v17;
  v116.m128i_i32[0] = (v17 & 0x40) != 0;
  sub_CCC4A0(a1, (__int64)"DisablePredication", (unsigned int *)&v116, &v113, 0);
  v18 = ((v116.m128i_i8[0] & 1) << 6) | *(_BYTE *)(a2 + 69) & 0xBF;
  *(_BYTE *)(a2 + 69) = v18;
  v116.m128i_i32[0] = v18 >> 7;
  sub_CCC4A0(a1, (__int64)"DisableXBlockSched", (unsigned int *)&v116, &v113, 0);
  *(_BYTE *)(a2 + 69) = (v116.m128i_i8[0] << 7) | *(_BYTE *)(a2 + 69) & 0x7F;
  v116.m128i_i32[0] = *(_BYTE *)(a2 + 70) & 3;
  sub_CCC4A0(a1, (__int64)"FP16Mode", (unsigned int *)&v116, &v113, 0);
  v19 = v116.m128i_i8[0] & 3 | *(_BYTE *)(a2 + 70) & 0xFC;
  *(_BYTE *)(a2 + 70) = v19;
  v116.m128i_i32[0] = (v19 & 4) != 0;
  sub_CCC4A0(a1, (__int64)"AllowComputeDerivatives", (unsigned int *)&v116, &v113, 0);
  v20 = (4 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 70) & 0xFB;
  *(_BYTE *)(a2 + 70) = v20;
  v116.m128i_i32[0] = (v20 & 8) != 0;
  sub_CCC4A0(a1, (__int64)"AllowDerivatives", (unsigned int *)&v116, &v113, 0);
  v21 = (8 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 70) & 0xF7;
  *(_BYTE *)(a2 + 70) = v21;
  v116.m128i_i32[0] = (v21 & 0x10) != 0;
  sub_CCC4A0(a1, (__int64)"UseOneForTrue", (unsigned int *)&v116, &v113, 0);
  v22 = (16 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 70) & 0xEF;
  *(_BYTE *)(a2 + 70) = v22;
  v116.m128i_i32[0] = (v22 & 0x20) != 0;
  sub_CCC4A0(a1, (__int64)"DisablePartialHalfVectorWrites", (unsigned int *)&v116, &v113, 0);
  v23 = (32 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 70) & 0xDF;
  *(_BYTE *)(a2 + 70) = v23;
  v116.m128i_i32[0] = (v23 & 0x40) != 0;
  sub_CCC4A0(a1, (__int64)"EnableNonUniformQuadDerivatives", (unsigned int *)&v116, &v113, 0);
  v24 = ((v116.m128i_i8[0] & 1) << 6) | *(_BYTE *)(a2 + 70) & 0xBF;
  *(_BYTE *)(a2 + 70) = v24;
  v116.m128i_i32[0] = v24 >> 7;
  sub_CCC4A0(a1, (__int64)"ManageAPICallDepth", (unsigned int *)&v116, &v113, 0);
  *(_BYTE *)(a2 + 70) = (v116.m128i_i8[0] << 7) | *(_BYTE *)(a2 + 70) & 0x7F;
  v116.m128i_i32[0] = *(_BYTE *)(a2 + 71) & 1;
  sub_CCC4A0(a1, (__int64)"DoMMACoalescing", (unsigned int *)&v116, &v113, 0);
  v25 = v116.m128i_i8[0] & 1 | *(_BYTE *)(a2 + 71) & 0xFE;
  *(_BYTE *)(a2 + 71) = v25;
  v116.m128i_i32[0] = (v25 & 2) != 0;
  sub_CCC4A0(a1, (__int64)"DumpPerfStats", (unsigned int *)&v116, &v113, 0);
  v26 = (2 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 71) & 0xFD;
  *(_BYTE *)(a2 + 71) = v26;
  v116.m128i_i32[0] = (v26 & 4) != 0;
  sub_CCC4A0(a1, (__int64)"ForceNTZ", (unsigned int *)&v116, &v113, 0);
  v27 = (4 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 71) & 0xFB;
  *(_BYTE *)(a2 + 71) = v27;
  v116.m128i_i32[0] = (v27 & 8) != 0;
  sub_CCC4A0(a1, (__int64)"ForceRELA", (unsigned int *)&v116, &v113, 0);
  v28 = (8 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 71) & 0xF7;
  *(_BYTE *)(a2 + 71) = v28;
  v116.m128i_i32[0] = (v28 >> 4) & 3;
  sub_CCC4A0(a1, (__int64)"AdvancedRemat", (unsigned int *)&v116, &v113, 0);
  v29 = (16 * (v116.m128i_i8[0] & 3)) | *(_BYTE *)(a2 + 71) & 0xCF;
  *(_BYTE *)(a2 + 71) = v29;
  v116.m128i_i32[0] = (v29 & 0x40) != 0;
  sub_CCC4A0(a1, (__int64)"CSSACoalescing", (unsigned int *)&v116, &v113, 0);
  v30 = ((v116.m128i_i8[0] & 1) << 6) | *(_BYTE *)(a2 + 71) & 0xBF;
  *(_BYTE *)(a2 + 71) = v30;
  v116.m128i_i32[0] = v30 >> 7;
  sub_CCC4A0(a1, (__int64)"DisableERRBARAfterMEMBAR", (unsigned int *)&v116, &v113, 0);
  *(_BYTE *)(a2 + 71) = (v116.m128i_i8[0] << 7) | *(_BYTE *)(a2 + 71) & 0x7F;
  v116.m128i_i32[0] = *(_BYTE *)(a2 + 72) & 1;
  sub_CCC4A0(a1, (__int64)"GenConvBranchForWarpSync", (unsigned int *)&v116, &v113, 0);
  v31 = v116.m128i_i8[0] & 1 | *(_BYTE *)(a2 + 72) & 0xFE;
  *(_BYTE *)(a2 + 72) = v31;
  v116.m128i_i32[0] = (v31 & 2) != 0;
  sub_CCC4A0(a1, (__int64)"DisableConvertMemoryToRegEstRegPresCodeSizeHeur", (unsigned int *)&v116, &v113, 0);
  v32 = (2 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 72) & 0xFD;
  *(_BYTE *)(a2 + 72) = v32;
  v116.m128i_i32[0] = (v32 & 4) != 0;
  sub_CCC4A0(a1, (__int64)"AssumeConvertMemoryToRegProfitable", (unsigned int *)&v116, &v113, 0);
  v33 = (4 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 72) & 0xFB;
  *(_BYTE *)(a2 + 72) = v33;
  v116.m128i_i32[0] = (v33 & 8) != 0;
  sub_CCC4A0(a1, (__int64)"MSTSForceOneCTAPerSMForSmemEmu", (unsigned int *)&v116, &v113, 0);
  *(_BYTE *)(a2 + 72) = (8 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 72) & 0xF7;
  v116.m128i_i32[0] = *(_DWORD *)(a2 + 284);
  sub_CCC4A0(a1, (__int64)"NumNopsAtStart", (unsigned int *)&v116, &v113, 0);
  *(_DWORD *)(a2 + 284) = v116.m128i_i32[0];
  v116.m128i_i32[0] = (*(_BYTE *)(a2 + 72) & 0x10) != 0;
  sub_CCC4A0(a1, (__int64)"EnableJumpTable", (unsigned int *)&v116, &v113, 0);
  v34 = (16 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 72) & 0xEF;
  *(_BYTE *)(a2 + 72) = v34;
  v116.m128i_i32[0] = (v34 & 0x20) != 0;
  sub_CCC4A0(a1, (__int64)"ScheduleKils", (unsigned int *)&v116, &v113, 0);
  v35 = (32 * (v116.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 72) & 0xDF;
  *(_BYTE *)(a2 + 72) = v35;
  v116.m128i_i32[0] = (v35 & 0x40) != 0;
  sub_CCC4A0(a1, (__int64)"IncludeEmulationFunctions", (unsigned int *)&v116, &v114, 0);
  v36 = ((v116.m128i_i8[0] & 1) << 6) | *(_BYTE *)(a2 + 72) & 0xBF;
  *(_BYTE *)(a2 + 72) = v36;
  v116.m128i_i32[0] = v36 >> 7;
  sub_CCC4A0(a1, (__int64)"UseEmulibForNoABI", (unsigned int *)&v116, &v113, 0);
  *(_BYTE *)(a2 + 72) = (v116.m128i_i8[0] << 7) | *(_BYTE *)(a2 + 72) & 0x7F;
  v116.m128i_i8[0] = *(_BYTE *)(a2 + 288);
  sub_CCCFB0(a1, (__int64)"DisableLegalizeIntegers", &v116, &v111, 0);
  *(_BYTE *)(a2 + 288) = v116.m128i_i8[0];
  v116.m128i_i8[0] = *(_BYTE *)(a2 + 289);
  sub_CCCFB0(a1, (__int64)"AddDepFromGlobalMembarToCB", &v116, &v111, 0);
  *(_BYTE *)(a2 + 289) = v116.m128i_i8[0];
  v116.m128i_i8[0] = *(_BYTE *)(a2 + 290);
  sub_CCCFB0(a1, (__int64)"CBUWAR5048043", &v116, &v111, 0);
  *(_BYTE *)(a2 + 290) = v116.m128i_i8[0];
  v116.m128i_i32[0] = *(_DWORD *)(a2 + 76) & 0xFFFFFF;
  sub_CCC4A0(a1, (__int64)"Reserved", (unsigned int *)&v116, &v113, 0);
  *(_WORD *)(a2 + 76) = v116.m128i_i16[0];
  *(_BYTE *)(a2 + 78) = v116.m128i_i8[2];
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    *(_QWORD *)(a2 + 80) = 0;
    *(_QWORD *)(a2 + 88) = 0;
  }
  v37 = *(_DWORD *)(a2 + 344);
  v38 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "PGOProfileKind",
         0,
         (v37 == 0) & v38,
         &v115,
         &v116) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v104 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD))(*(_QWORD *)a1 + 168LL);
    v60 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v61 = v104(a1, "NV_PIECEMEAL_PROFILER_DISABLED", (v37 == 0) & v60) == 0;
    v62 = *(_QWORD *)a1;
    v105 = *(__int64 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    if ( v61 )
    {
      v63 = (*(__int64 (__fastcall **)(__int64))(v62 + 16))(a1);
      v64 = 0;
      if ( v63 )
        v64 = v37 == 1;
      v65 = v105(a1, "NV_PIECEMEAL_PROFILER_ZEROP", v64);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v62 + 16))(a1);
      v37 = 0;
      v65 = v105(a1, "NV_PIECEMEAL_PROFILER_ZEROP", 0);
    }
    v61 = v65 == 0;
    v66 = *(_QWORD *)a1;
    v106 = *(__int64 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    if ( v61 )
    {
      v67 = (*(__int64 (__fastcall **)(__int64))(v66 + 16))(a1);
      v68 = 0;
      if ( v67 )
        v68 = v37 == 2;
      v69 = v106(a1, "NV_PIECEMEAL_PROFILER_ALPHA_BETA", v68);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v66 + 16))(a1);
      v37 = 1;
      v69 = v106(a1, "NV_PIECEMEAL_PROFILER_ALPHA_BETA", 0);
    }
    v61 = v69 == 0;
    v70 = *(_QWORD *)a1;
    v107 = *(__int64 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    if ( v61 )
    {
      v71 = (*(__int64 (__fastcall **)(__int64))(v70 + 16))(a1);
      v72 = 0;
      if ( v71 )
        v72 = v37 == 3;
      v73 = v107(a1, "NV_PIECEMEAL_PROFILER_SANITY", v72);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v70 + 16))(a1);
      v37 = 2;
      v73 = v107(a1, "NV_PIECEMEAL_PROFILER_SANITY", 0);
    }
    v61 = v73 == 0;
    v74 = *(_QWORD *)a1;
    v108 = *(__int64 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    if ( v61 )
    {
      v75 = (*(__int64 (__fastcall **)(__int64))(v74 + 16))(a1);
      v76 = 0;
      if ( v75 )
        v76 = v37 == 4;
      v77 = v108(a1, "NV_PIECEMEAL_PROFILER_BB_WEIGHTS", v76);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v74 + 16))(a1);
      v37 = 3;
      v77 = v108(a1, "NV_PIECEMEAL_PROFILER_BB_WEIGHTS", 0);
    }
    if ( v77 )
      v37 = 4;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v116.m128i_i64[0]);
  }
  else if ( (_BYTE)v115 )
  {
    v37 = 0;
  }
  *(_DWORD *)(a2 + 344) = v37;
  v116.m128i_i32[0] = *(_DWORD *)(a2 + 348);
  sub_CCC4A0(a1, (__int64)"PGOEpoch", (unsigned int *)&v116, &v113, 0);
  *(_DWORD *)(a2 + 348) = v116.m128i_i32[0];
  v116.m128i_i32[0] = *(_DWORD *)(a2 + 352);
  sub_CCC4A0(a1, (__int64)"PGOBatchSize", (unsigned int *)&v116, &v113, 0);
  *(_DWORD *)(a2 + 352) = v116.m128i_i32[0];
  v116.m128i_i32[0] = *(_DWORD *)(a2 + 356);
  sub_CCC4A0(a1, (__int64)"PGOCounterMemBaseVAIndex", (unsigned int *)&v116, &v113, 0);
  *(_DWORD *)(a2 + 356) = v116.m128i_i32[0];
  v116.m128i_i32[0] = *(_DWORD *)(a2 + 360);
  sub_CCC4A0(a1, (__int64)"PGOCounterMemOffsetIndex", (unsigned int *)&v116, &v113, 0);
  *(_DWORD *)(a2 + 360) = v116.m128i_i32[0];
  v116.m128i_i32[0] = *(_DWORD *)(a2 + 364);
  sub_CCC4A0(a1, (__int64)"PGONumReplicatedCopies", (unsigned int *)&v116, &v113, 0);
  *(_DWORD *)(a2 + 364) = v116.m128i_i32[0];
  v116.m128i_i32[0] = *(_DWORD *)(a2 + 368);
  sub_CCC4A0(a1, (__int64)"PGOBatchIdInCurAllocation", (unsigned int *)&v116, &v113, 0);
  *(_DWORD *)(a2 + 368) = v116.m128i_i32[0];
  v116.m128i_i32[0] = *(_DWORD *)(a2 + 372);
  sub_CCC4A0(a1, (__int64)"PGONumBatchesPerAllocation", (unsigned int *)&v116, &v113, 0);
  *(_DWORD *)(a2 + 372) = v116.m128i_i32[0];
  v116.m128i_i64[0] = *(_QWORD *)(a2 + 376);
  v116.m128i_i64[1] = a2 + 376;
  v39 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v40 = 0;
  if ( v39 )
    v40 = v116.m128i_i64[0] == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "PGOAppHash",
         0,
         v40,
         &v112,
         &v115) )
  {
    sub_CCC650(a1, v116.m128i_i64);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v115);
  }
  else if ( v112 )
  {
    v116.m128i_i64[0] = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_QWORD *)v116.m128i_i64[1] = v116.m128i_i64[0];
  v116.m128i_i64[0] = *(_QWORD *)(a2 + 384);
  v116.m128i_i64[1] = a2 + 384;
  v41 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v42 = 0;
  if ( v41 )
    v42 = v116.m128i_i64[0] == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "PGOProfileHash",
         0,
         v42,
         &v112,
         &v115) )
  {
    sub_CCC650(a1, v116.m128i_i64);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v115);
  }
  else if ( v112 )
  {
    v116.m128i_i64[0] = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_QWORD *)v116.m128i_i64[1] = v116.m128i_i64[0];
  v116.m128i_i64[0] = *(_QWORD *)(a2 + 392);
  v116.m128i_i64[1] = a2 + 392;
  v43 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v44 = 0;
  if ( v43 )
    v44 = v116.m128i_i64[0] == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "PGOOptionsHash",
         0,
         v44,
         &v112,
         &v115) )
  {
    sub_CCC650(a1, v116.m128i_i64);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v115);
  }
  else if ( v112 )
  {
    v116.m128i_i64[0] = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_QWORD *)v116.m128i_i64[1] = v116.m128i_i64[0];
  v116.m128i_i32[0] = *(_DWORD *)(a2 + 400);
  sub_CCC4A0(a1, (__int64)"PGOFlags", (unsigned int *)&v116, &v113, 0);
  v45 = v116.m128i_i8[0];
  v46 = *(_DWORD *)(a2 + 344);
  *(_DWORD *)(a2 + 400) = v116.m128i_i32[0];
  if ( v46 == 1 )
  {
    if ( (v45 & 0x10) == 0 )
      goto LABEL_39;
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    {
      v59 = *(_QWORD *)(a2 + 408);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
              a1,
              "ZeroPPGODataPtr",
              0,
              0,
              &v115,
              &v116) )
        goto LABEL_39;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      sub_CD58A0(a1, v59);
      goto LABEL_77;
    }
    v96 = *(_QWORD *)a1;
    v119 = 0;
    v116 = 0;
    v117 = 0;
    v118 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(v96 + 120))(
           a1,
           "ZeroPPGODataPtr",
           0,
           0,
           &v112,
           &v115) )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      sub_CD58A0(a1, (__int64)&v116);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v115);
    }
    v97 = sub_CB0A70(a1);
    v98 = sub_CD1D40(*(__int64 **)(v97 + 8), 56, 3);
    v99 = _mm_loadu_si128(&v116);
    *(_QWORD *)(a2 + 408) = v98;
    *(__m128i *)v98 = v99;
    *(__m128i *)(v98 + 16) = _mm_loadu_si128(&v117);
    *(__m128i *)(v98 + 32) = _mm_loadu_si128(&v118);
    *(_QWORD *)(v98 + 48) = v119;
  }
  else if ( v46 == 4 && (v45 & 0x20) != 0 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    {
      v58 = *(_QWORD *)(a2 + 416);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
              a1,
              "BBweightPGODataPtr",
              0,
              0,
              &v115,
              &v116) )
        goto LABEL_39;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      sub_CD64F0(a1, v58);
LABEL_77:
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v116.m128i_i64[0]);
      goto LABEL_39;
    }
    v100 = *(_QWORD *)a1;
    v118.m128i_i64[0] = 0;
    v116 = 0;
    v117 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(v100 + 120))(
           a1,
           "BBweightPGODataPtr",
           0,
           0,
           &v112,
           &v115) )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      sub_CD64F0(a1, (__int64)&v116);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v115);
    }
    v101 = sub_CB0A70(a1);
    v102 = sub_CD1D40(*(__int64 **)(v101 + 8), 40, 3);
    v103 = _mm_loadu_si128(&v116);
    *(_QWORD *)(a2 + 416) = v102;
    *(__m128i *)v102 = v103;
    *(__m128i *)(v102 + 16) = _mm_loadu_si128(&v117);
    *(_QWORD *)(v102 + 32) = v118.m128i_i64[0];
  }
LABEL_39:
  sub_CD5150(a1, (__int64)"OCGKnobs", (char **)(a2 + 312));
  sub_CD5150(a1, (__int64)"OCGKnobsFile", (char **)(a2 + 320));
  sub_CD5150(a1, (__int64)"NVVMKnobsString", (char **)(a2 + 296));
  sub_CD5150(a1, (__int64)"OmegaKnobs", (char **)(a2 + 304));
  sub_CD5150(a1, (__int64)"FinalizerKnobs", (char **)(a2 + 328));
  v116.m128i_i64[0] = *(_QWORD *)(a2 + 336);
  v116.m128i_i64[1] = a2 + 336;
  v47 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v48 = 0;
  if ( v47 )
    v48 = v116.m128i_i64[0] == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ProgramHash",
         0,
         v48,
         &v112,
         &v115) )
  {
    sub_CCC650(a1, v116.m128i_i64);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v115);
  }
  else if ( v112 )
  {
    v116.m128i_i64[0] = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_QWORD *)v116.m128i_i64[1] = v116.m128i_i64[0];
  sub_CD9000(a1, (__int64)"AlignEntries", (_QWORD *)(a2 + 200));
  sub_CD7780(a1, (__int64)"RegTargets", a2 + 168);
  sub_CD7780(a1, (__int64)"PerfStatsRegTargets", a2 + 184);
  sub_CCC4A0(a1, (__int64)"PSIGSCBOffset", (unsigned int *)(a2 + 248), &v113, 0);
  sub_CCC4A0(a1, (__int64)"PSIThreadMaskBaseOffset", (unsigned int *)(a2 + 252), &v113, 0);
  sub_CCC4A0(a1, (__int64)"MSTSSharedMemBaseGSCBByteOffset", (unsigned int *)(a2 + 256), &v113, 0);
  sub_CCC4A0(a1, (__int64)"MaxMSTSSharedMemSizePerCTA", (unsigned int *)(a2 + 260), &v113, 0);
  v49 = v113;
  v116.m128i_i32[0] = *(_DWORD *)(a2 + 264);
  v116.m128i_i64[1] = a2 + 264;
  v50 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v51 = 0;
  if ( v50 )
    v51 = v116.m128i_i32[0] == v49;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "DisableCBSpeculateMask",
         0,
         v51,
         &v112,
         &v115) )
  {
    sub_CCCA10(a1, v116.m128i_i32);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v115);
  }
  else if ( v112 )
  {
    v116.m128i_i32[0] = v49;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_DWORD *)v116.m128i_i64[1] = v116.m128i_i32[0];
  sub_CCC4A0(a1, (__int64)"CTASizeX", (unsigned int *)(a2 + 96), &v113, 0);
  sub_CCC4A0(a1, (__int64)"CTASizeY", (unsigned int *)(a2 + 100), &v113, 0);
  sub_CCC4A0(a1, (__int64)"CTASizeZ", (unsigned int *)(a2 + 104), &v113, 0);
  sub_CCC4A0(a1, (__int64)"SMemScratchBase", (unsigned int *)(a2 + 112), &v113, 0);
  sub_CCC4A0(a1, (__int64)"SharedMemorySize", (unsigned int *)(a2 + 112), (_DWORD *)(a2 + 112), 0);
  sub_CCC4A0(a1, (__int64)"MaxFlatSMemScratchPerThread", (unsigned int *)(a2 + 116), &v113, 0);
  sub_CD7780(a1, (__int64)"FlatSMemScratchPerThread", a2 + 120);
  sub_CD7780(a1, (__int64)"PerfStatsFlatSMemScratchPerThread", a2 + 136);
  sub_CCC4A0(a1, (__int64)"SMemScratchWarpStride", (unsigned int *)(a2 + 152), &v113, 0);
  v52 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v53 = 0;
  if ( v52 )
    v53 = *(_DWORD *)(a2 + 156) == -1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SMemPerSM",
         0,
         v53,
         &v115,
         &v116) )
  {
    sub_CCD060(a1, (int *)(a2 + 156));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v116.m128i_i64[0]);
  }
  else if ( (_BYTE)v115 )
  {
    *(_DWORD *)(a2 + 156) = -1;
  }
  v54 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v55 = 0;
  if ( v54 )
    v55 = *(_DWORD *)(a2 + 160) == -1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "LMemHIReservation",
         0,
         v55,
         &v115,
         &v116) )
  {
    sub_CCD060(a1, (int *)(a2 + 160));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v116.m128i_i64[0]);
  }
  else if ( (_BYTE)v115 )
  {
    *(_DWORD *)(a2 + 160) = -1;
  }
  v56 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v57 = 0;
  if ( v56 )
    v57 = *(_DWORD *)(a2 + 268) == -1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "NumReservedUReg",
         0,
         v57,
         &v115,
         &v116) )
  {
    sub_CCD060(a1, (int *)(a2 + 268));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v116.m128i_i64[0]);
  }
  else if ( (_BYTE)v115 )
  {
    *(_DWORD *)(a2 + 268) = -1;
  }
  sub_CCC4A0(a1, (__int64)"NumScratchURegs", (unsigned int *)(a2 + 272), &v113, 0);
  sub_CCC4A0(a1, (__int64)"MaxActiveWarpsPerSM", (unsigned int *)(a2 + 280), &v113, 0);
  sub_CCC4A0(a1, (__int64)"NumNopsAtStart", (unsigned int *)(a2 + 284), &v113, 0);
  sub_CD9640(a1, (__int64)"OptimizerConstBankConstants", a2 + 216);
}
