// Function: sub_1C20170
// Address: 0x1c20170
//
void __fastcall sub_1C20170(__int64 a1, __int64 a2)
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
  char v37; // al
  _BOOL8 v38; // rcx
  char v39; // al
  int v40; // r15d
  unsigned __int8 v41; // al
  char v42; // al
  _BOOL8 v43; // rcx
  char v44; // al
  _BOOL8 v45; // rcx
  char v46; // al
  _BOOL8 v47; // rcx
  __int8 v48; // dl
  int v49; // eax
  char v50; // al
  _BOOL8 v51; // rcx
  __int32 v52; // r15d
  char v53; // al
  _BOOL8 v54; // rcx
  char v55; // al
  _BOOL8 v56; // rcx
  char v57; // al
  _BOOL8 v58; // rcx
  char v59; // al
  _BOOL8 v60; // rcx
  __int64 v61; // r15
  unsigned __int8 v62; // al
  bool v63; // zf
  __int64 v64; // rax
  char v65; // al
  _BOOL8 v66; // rdx
  char v67; // al
  __int64 v68; // rax
  char v69; // al
  _BOOL8 v70; // rdx
  char v71; // al
  __int64 v72; // rax
  char v73; // al
  _BOOL8 v74; // rdx
  char v75; // al
  __int64 v76; // rax
  char v77; // al
  _BOOL8 v78; // rdx
  char v79; // al
  char v80; // al
  _BOOL8 v81; // rdx
  char v82; // al
  _BOOL8 v83; // rdx
  char v84; // al
  _BOOL8 v85; // rcx
  char v86; // al
  char v87; // al
  char v88; // al
  _BOOL8 v89; // rcx
  char v90; // al
  char v91; // al
  char v92; // al
  _BOOL8 v93; // rcx
  char v94; // al
  char v95; // al
  _BOOL8 v96; // rcx
  unsigned int v97; // eax
  __int64 v98; // r15
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rax
  __m128i v102; // xmm1
  __m128i v103; // xmm2
  __m128i v104; // xmm3
  __int64 v105; // rdx
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rax
  __m128i v109; // xmm4
  __m128i v110; // xmm5
  __int64 v111; // rdx
  unsigned __int8 (__fastcall *v112)(__int64, const char *, _QWORD); // [rsp+0h] [rbp-B0h]
  __int64 (__fastcall *v113)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  __int64 (__fastcall *v114)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  __int64 (__fastcall *v115)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  __int64 (__fastcall *v116)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  unsigned __int8 (__fastcall *v117)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  unsigned __int8 (__fastcall *v118)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-A8h]
  char v119; // [rsp+2Eh] [rbp-82h] BYREF
  char v120; // [rsp+2Fh] [rbp-81h] BYREF
  __int32 v121; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v122; // [rsp+34h] [rbp-7Ch] BYREF
  __int64 v123; // [rsp+38h] [rbp-78h] BYREF
  __m128i v124; // [rsp+40h] [rbp-70h] BYREF
  __m128i v125; // [rsp+50h] [rbp-60h] BYREF
  __m128i v126; // [rsp+60h] [rbp-50h] BYREF
  __int64 v127; // [rsp+70h] [rbp-40h]

  sub_1C1EDC0(a1, a2);
  sub_1C15590(a1, a2 + 48);
  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = *(_DWORD *)(a2 + 432) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CooperativeVectorInfo",
         0,
         v3,
         &v120) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
    v122 = *(_BYTE *)(a2 + 432) & 1;
    v84 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v85 = 0;
    if ( v84 )
      v85 = v122 == 0;
    v86 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int32 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
            a1,
            "U.DisableSAMRAM",
            0,
            v85,
            &v121,
            &v124);
    if ( v86 )
    {
      sub_1C14710(a1, &v122);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v124.m128i_i64[0]);
      v86 = v122 & 1;
    }
    else if ( !(_BYTE)v121 )
    {
      v86 = v122 & 1;
    }
    v87 = v86 & 1 | *(_BYTE *)(a2 + 432) & 0xFE;
    *(_BYTE *)(a2 + 432) = v87;
    v122 = (v87 & 2) != 0;
    v88 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v89 = 0;
    if ( v88 )
      v89 = v122 == 0;
    v90 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int32 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
            a1,
            "U.DisableLayerFusion",
            0,
            v89,
            &v121,
            &v124);
    if ( v90 )
    {
      sub_1C14710(a1, &v122);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v124.m128i_i64[0]);
      v90 = v122 & 1;
    }
    else if ( !(_BYTE)v121 )
    {
      v90 = v122 & 1;
    }
    v91 = (2 * (v90 & 1)) | *(_BYTE *)(a2 + 432) & 0xFD;
    *(_BYTE *)(a2 + 432) = v91;
    v122 = (v91 & 4) != 0;
    v92 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v93 = 0;
    if ( v92 )
      v93 = v122 == 0;
    v94 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int32 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
            a1,
            "U.DisableI32MatrixLayout",
            0,
            v93,
            &v121,
            &v124);
    if ( v94 )
    {
      sub_1C14710(a1, &v122);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v124.m128i_i64[0]);
      v94 = v122 & 1;
    }
    else if ( !(_BYTE)v121 )
    {
      v94 = v122 & 1;
    }
    *(_BYTE *)(a2 + 432) = *(_BYTE *)(a2 + 432) & 0xFB | (4 * (v94 & 1));
    v122 = *(_DWORD *)(a2 + 432) >> 3;
    v95 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v96 = 0;
    if ( v95 )
      v96 = v122 == 0;
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int32 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
           a1,
           "U.Reserved",
           0,
           v96,
           &v121,
           &v124) )
    {
      sub_1C14710(a1, &v122);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v124.m128i_i64[0]);
      v97 = v122 & 0x1FFFFFFF;
    }
    else
    {
      v97 = 0;
      if ( !(_BYTE)v121 )
        v97 = v122 & 0x1FFFFFFF;
    }
    *(_DWORD *)(a2 + 432) = (8 * v97) | *(_DWORD *)(a2 + 432) & 7;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v123);
  }
  else if ( v120 )
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
         &v123,
         &v124) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v117 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v80 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v81 = 0;
    if ( v80 )
      v81 = *(_DWORD *)(a2 + 64) == 0;
    if ( v117(a1, "D2IR", v81) )
      *(_DWORD *)(a2 + 64) = 0;
    v118 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v82 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v83 = 0;
    if ( v82 )
      v83 = *(_DWORD *)(a2 + 64) == 1;
    if ( v118(a1, "Omega", v83) )
      *(_DWORD *)(a2 + 64) = 1;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v124.m128i_i64[0]);
  }
  else if ( (_BYTE)v123 )
  {
    *(_DWORD *)(a2 + 64) = 0;
  }
  v6 = *(_BYTE *)(a2 + 68);
  v121 = 0;
  v122 = 1;
  v119 = 0;
  v124.m128i_i32[0] = (v6 & 2) != 0;
  sub_1C14890(a1, (__int64)"PromoteHalf", (unsigned int *)&v124, &v122, 0);
  v7 = (2 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 68) & 0xFD;
  *(_BYTE *)(a2 + 68) = v7;
  v124.m128i_i32[0] = (v7 & 4) != 0;
  sub_1C14890(a1, (__int64)"IgnoreRndFtzOnF32F16Conv", (unsigned int *)&v124, &v121, 0);
  v8 = (4 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 68) & 0xFB;
  *(_BYTE *)(a2 + 68) = v8;
  v124.m128i_i32[0] = v8 & 1;
  sub_1C14890(a1, (__int64)"PromoteFixed", (unsigned int *)&v124, &v121, 0);
  v9 = v124.m128i_i8[0] & 1 | *(_BYTE *)(a2 + 68) & 0xFE;
  *(_BYTE *)(a2 + 68) = v9;
  v124.m128i_i32[0] = (v9 & 8) != 0;
  sub_1C14890(a1, (__int64)"UsePIXBAR", (unsigned int *)&v124, &v121, 0);
  v10 = (8 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 68) & 0xF7;
  *(_BYTE *)(a2 + 68) = v10;
  v124.m128i_i32[0] = (v10 & 0x10) != 0;
  sub_1C14890(a1, (__int64)"TLDUsesTLD4CompatibleSampler", (unsigned int *)&v124, &v121, 0);
  v11 = (16 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 68) & 0xEF;
  *(_BYTE *)(a2 + 68) = v11;
  v124.m128i_i32[0] = (v11 & 0x20) != 0;
  sub_1C14890(a1, (__int64)"VSIsVREnabled", (unsigned int *)&v124, &v121, 0);
  v12 = (32 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 68) & 0xDF;
  *(_BYTE *)(a2 + 68) = v12;
  v124.m128i_i32[0] = (v12 & 0x40) != 0;
  sub_1C14890(a1, (__int64)"VSIsLastVTGStage", (unsigned int *)&v124, &v121, 0);
  v13 = ((v124.m128i_i8[0] & 1) << 6) | *(_BYTE *)(a2 + 68) & 0xBF;
  *(_BYTE *)(a2 + 68) = v13;
  v124.m128i_i32[0] = v13 >> 7;
  sub_1C14890(a1, (__int64)"EnableZeroCoverageKill", (unsigned int *)&v124, &v121, 0);
  *(_BYTE *)(a2 + 68) = (v124.m128i_i8[0] << 7) | *(_BYTE *)(a2 + 68) & 0x7F;
  v124.m128i_i32[0] = *(_BYTE *)(a2 + 69) & 1;
  sub_1C14890(a1, (__int64)"EnablePartialBindlessTextures", (unsigned int *)&v124, &v121, 0);
  v14 = v124.m128i_i8[0] & 1 | *(_BYTE *)(a2 + 69) & 0xFE;
  *(_BYTE *)(a2 + 69) = v14;
  v124.m128i_i32[0] = (v14 & 4) != 0;
  sub_1C14890(a1, (__int64)"DisableKeplerLUWar", (unsigned int *)&v124, &v121, 0);
  v15 = (4 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 69) & 0xFB;
  *(_BYTE *)(a2 + 69) = v15;
  v124.m128i_i32[0] = (v15 >> 3) & 3;
  sub_1C14890(a1, (__int64)"ReorderCSE", (unsigned int *)&v124, &v121, 0);
  v16 = (8 * (v124.m128i_i8[0] & 3)) | *(_BYTE *)(a2 + 69) & 0xE7;
  *(_BYTE *)(a2 + 69) = v16;
  v124.m128i_i32[0] = (v16 & 0x20) != 0;
  sub_1C14890(a1, (__int64)"InitUninitialized", (unsigned int *)&v124, &v121, 0);
  v17 = (32 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 69) & 0xDF;
  *(_BYTE *)(a2 + 69) = v17;
  v124.m128i_i32[0] = (v17 & 0x40) != 0;
  sub_1C14890(a1, (__int64)"DisablePredication", (unsigned int *)&v124, &v121, 0);
  v18 = ((v124.m128i_i8[0] & 1) << 6) | *(_BYTE *)(a2 + 69) & 0xBF;
  *(_BYTE *)(a2 + 69) = v18;
  v124.m128i_i32[0] = v18 >> 7;
  sub_1C14890(a1, (__int64)"DisableXBlockSched", (unsigned int *)&v124, &v121, 0);
  *(_BYTE *)(a2 + 69) = (v124.m128i_i8[0] << 7) | *(_BYTE *)(a2 + 69) & 0x7F;
  v124.m128i_i32[0] = *(_BYTE *)(a2 + 70) & 3;
  sub_1C14890(a1, (__int64)"FP16Mode", (unsigned int *)&v124, &v121, 0);
  v19 = v124.m128i_i8[0] & 3 | *(_BYTE *)(a2 + 70) & 0xFC;
  *(_BYTE *)(a2 + 70) = v19;
  v124.m128i_i32[0] = (v19 & 4) != 0;
  sub_1C14890(a1, (__int64)"AllowComputeDerivatives", (unsigned int *)&v124, &v121, 0);
  v20 = (4 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 70) & 0xFB;
  *(_BYTE *)(a2 + 70) = v20;
  v124.m128i_i32[0] = (v20 & 8) != 0;
  sub_1C14890(a1, (__int64)"AllowDerivatives", (unsigned int *)&v124, &v121, 0);
  v21 = (8 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 70) & 0xF7;
  *(_BYTE *)(a2 + 70) = v21;
  v124.m128i_i32[0] = (v21 & 0x10) != 0;
  sub_1C14890(a1, (__int64)"UseOneForTrue", (unsigned int *)&v124, &v121, 0);
  v22 = (16 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 70) & 0xEF;
  *(_BYTE *)(a2 + 70) = v22;
  v124.m128i_i32[0] = (v22 & 0x20) != 0;
  sub_1C14890(a1, (__int64)"DisablePartialHalfVectorWrites", (unsigned int *)&v124, &v121, 0);
  v23 = (32 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 70) & 0xDF;
  *(_BYTE *)(a2 + 70) = v23;
  v124.m128i_i32[0] = (v23 & 0x40) != 0;
  sub_1C14890(a1, (__int64)"EnableNonUniformQuadDerivatives", (unsigned int *)&v124, &v121, 0);
  v24 = ((v124.m128i_i8[0] & 1) << 6) | *(_BYTE *)(a2 + 70) & 0xBF;
  *(_BYTE *)(a2 + 70) = v24;
  v124.m128i_i32[0] = v24 >> 7;
  sub_1C14890(a1, (__int64)"ManageAPICallDepth", (unsigned int *)&v124, &v121, 0);
  *(_BYTE *)(a2 + 70) = (v124.m128i_i8[0] << 7) | *(_BYTE *)(a2 + 70) & 0x7F;
  v124.m128i_i32[0] = *(_BYTE *)(a2 + 71) & 1;
  sub_1C14890(a1, (__int64)"DoMMACoalescing", (unsigned int *)&v124, &v121, 0);
  v25 = v124.m128i_i8[0] & 1 | *(_BYTE *)(a2 + 71) & 0xFE;
  *(_BYTE *)(a2 + 71) = v25;
  v124.m128i_i32[0] = (v25 & 2) != 0;
  sub_1C14890(a1, (__int64)"DumpPerfStats", (unsigned int *)&v124, &v121, 0);
  v26 = (2 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 71) & 0xFD;
  *(_BYTE *)(a2 + 71) = v26;
  v124.m128i_i32[0] = (v26 & 4) != 0;
  sub_1C14890(a1, (__int64)"ForceNTZ", (unsigned int *)&v124, &v121, 0);
  v27 = (4 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 71) & 0xFB;
  *(_BYTE *)(a2 + 71) = v27;
  v124.m128i_i32[0] = (v27 & 8) != 0;
  sub_1C14890(a1, (__int64)"ForceRELA", (unsigned int *)&v124, &v121, 0);
  v28 = (8 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 71) & 0xF7;
  *(_BYTE *)(a2 + 71) = v28;
  v124.m128i_i32[0] = (v28 >> 4) & 3;
  sub_1C14890(a1, (__int64)"AdvancedRemat", (unsigned int *)&v124, &v121, 0);
  v29 = (16 * (v124.m128i_i8[0] & 3)) | *(_BYTE *)(a2 + 71) & 0xCF;
  *(_BYTE *)(a2 + 71) = v29;
  v124.m128i_i32[0] = (v29 & 0x40) != 0;
  sub_1C14890(a1, (__int64)"CSSACoalescing", (unsigned int *)&v124, &v121, 0);
  v30 = ((v124.m128i_i8[0] & 1) << 6) | *(_BYTE *)(a2 + 71) & 0xBF;
  *(_BYTE *)(a2 + 71) = v30;
  v124.m128i_i32[0] = v30 >> 7;
  sub_1C14890(a1, (__int64)"DisableERRBARAfterMEMBAR", (unsigned int *)&v124, &v121, 0);
  *(_BYTE *)(a2 + 71) = (v124.m128i_i8[0] << 7) | *(_BYTE *)(a2 + 71) & 0x7F;
  v124.m128i_i32[0] = *(_BYTE *)(a2 + 72) & 1;
  sub_1C14890(a1, (__int64)"GenConvBranchForWarpSync", (unsigned int *)&v124, &v121, 0);
  v31 = v124.m128i_i8[0] & 1 | *(_BYTE *)(a2 + 72) & 0xFE;
  *(_BYTE *)(a2 + 72) = v31;
  v124.m128i_i32[0] = (v31 & 2) != 0;
  sub_1C14890(a1, (__int64)"DisableConvertMemoryToRegEstRegPresCodeSizeHeur", (unsigned int *)&v124, &v121, 0);
  v32 = (2 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 72) & 0xFD;
  *(_BYTE *)(a2 + 72) = v32;
  v124.m128i_i32[0] = (v32 & 4) != 0;
  sub_1C14890(a1, (__int64)"AssumeConvertMemoryToRegProfitable", (unsigned int *)&v124, &v121, 0);
  v33 = (4 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 72) & 0xFB;
  *(_BYTE *)(a2 + 72) = v33;
  v124.m128i_i32[0] = (v33 & 8) != 0;
  sub_1C14890(a1, (__int64)"MSTSForceOneCTAPerSMForSmemEmu", (unsigned int *)&v124, &v121, 0);
  *(_BYTE *)(a2 + 72) = (8 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 72) & 0xF7;
  v124.m128i_i32[0] = *(_DWORD *)(a2 + 284);
  sub_1C14890(a1, (__int64)"NumNopsAtStart", (unsigned int *)&v124, &v121, 0);
  *(_DWORD *)(a2 + 284) = v124.m128i_i32[0];
  v124.m128i_i32[0] = (*(_BYTE *)(a2 + 72) & 0x10) != 0;
  sub_1C14890(a1, (__int64)"EnableJumpTable", (unsigned int *)&v124, &v121, 0);
  v34 = (16 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 72) & 0xEF;
  *(_BYTE *)(a2 + 72) = v34;
  v124.m128i_i32[0] = (v34 & 0x20) != 0;
  sub_1C14890(a1, (__int64)"ScheduleKils", (unsigned int *)&v124, &v121, 0);
  v35 = (32 * (v124.m128i_i8[0] & 1)) | *(_BYTE *)(a2 + 72) & 0xDF;
  *(_BYTE *)(a2 + 72) = v35;
  v124.m128i_i32[0] = (v35 & 0x40) != 0;
  sub_1C14890(a1, (__int64)"IncludeEmulationFunctions", (unsigned int *)&v124, &v122, 0);
  v36 = ((v124.m128i_i8[0] & 1) << 6) | *(_BYTE *)(a2 + 72) & 0xBF;
  *(_BYTE *)(a2 + 72) = v36;
  v124.m128i_i32[0] = v36 >> 7;
  sub_1C14890(a1, (__int64)"UseEmulibForNoABI", (unsigned int *)&v124, &v121, 0);
  *(_BYTE *)(a2 + 72) = (v124.m128i_i8[0] << 7) | *(_BYTE *)(a2 + 72) & 0x7F;
  v124.m128i_i8[0] = *(_BYTE *)(a2 + 288);
  sub_1C144E0(a1, (__int64)"DisableLegalizeIntegers", &v124, &v119, 0);
  *(_BYTE *)(a2 + 288) = v124.m128i_i8[0];
  v124.m128i_i8[0] = *(_BYTE *)(a2 + 289);
  sub_1C144E0(a1, (__int64)"AddDepFromGlobalMembarToCB", &v124, &v119, 0);
  *(_BYTE *)(a2 + 289) = v124.m128i_i8[0];
  v120 = *(_BYTE *)(a2 + 290);
  v37 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v38 = 0;
  if ( v37 )
    v38 = v120 == v119;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CBUWAR5048043",
         0,
         v38,
         &v123,
         &v124) )
  {
    sub_1C14360(a1, &v120);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v124.m128i_i64[0]);
    v39 = v120;
  }
  else if ( (_BYTE)v123 )
  {
    v39 = v119;
  }
  else
  {
    v39 = v120;
  }
  *(_BYTE *)(a2 + 290) = v39;
  v124.m128i_i32[0] = *(_DWORD *)(a2 + 76) & 0xFFFFFF;
  sub_1C14890(a1, (__int64)"Reserved", (unsigned int *)&v124, &v121, 0);
  *(_WORD *)(a2 + 76) = v124.m128i_i16[0];
  *(_BYTE *)(a2 + 78) = v124.m128i_i8[2];
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    *(_QWORD *)(a2 + 80) = 0;
    *(_QWORD *)(a2 + 88) = 0;
  }
  v40 = *(_DWORD *)(a2 + 344);
  v41 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "PGOProfileKind",
         0,
         (v40 == 0) & v41,
         &v123,
         &v124) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v112 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD))(*(_QWORD *)a1 + 168LL);
    v62 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v63 = v112(a1, "NV_PIECEMEAL_PROFILER_DISABLED", (v40 == 0) & v62) == 0;
    v64 = *(_QWORD *)a1;
    v113 = *(__int64 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    if ( v63 )
    {
      v65 = (*(__int64 (__fastcall **)(__int64))(v64 + 16))(a1);
      v66 = 0;
      if ( v65 )
        v66 = v40 == 1;
      v67 = v113(a1, "NV_PIECEMEAL_PROFILER_ZEROP", v66);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v64 + 16))(a1);
      v40 = 0;
      v67 = v113(a1, "NV_PIECEMEAL_PROFILER_ZEROP", 0);
    }
    v63 = v67 == 0;
    v68 = *(_QWORD *)a1;
    v114 = *(__int64 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    if ( v63 )
    {
      v69 = (*(__int64 (__fastcall **)(__int64))(v68 + 16))(a1);
      v70 = 0;
      if ( v69 )
        v70 = v40 == 2;
      v71 = v114(a1, "NV_PIECEMEAL_PROFILER_ALPHA_BETA", v70);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v68 + 16))(a1);
      v40 = 1;
      v71 = v114(a1, "NV_PIECEMEAL_PROFILER_ALPHA_BETA", 0);
    }
    v63 = v71 == 0;
    v72 = *(_QWORD *)a1;
    v115 = *(__int64 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    if ( v63 )
    {
      v73 = (*(__int64 (__fastcall **)(__int64))(v72 + 16))(a1);
      v74 = 0;
      if ( v73 )
        v74 = v40 == 3;
      v75 = v115(a1, "NV_PIECEMEAL_PROFILER_SANITY", v74);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v72 + 16))(a1);
      v40 = 2;
      v75 = v115(a1, "NV_PIECEMEAL_PROFILER_SANITY", 0);
    }
    v63 = v75 == 0;
    v76 = *(_QWORD *)a1;
    v116 = *(__int64 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    if ( v63 )
    {
      v77 = (*(__int64 (__fastcall **)(__int64))(v76 + 16))(a1);
      v78 = 0;
      if ( v77 )
        v78 = v40 == 4;
      v79 = v116(a1, "NV_PIECEMEAL_PROFILER_BB_WEIGHTS", v78);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v76 + 16))(a1);
      v40 = 3;
      v79 = v116(a1, "NV_PIECEMEAL_PROFILER_BB_WEIGHTS", 0);
    }
    if ( v79 )
      v40 = 4;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v124.m128i_i64[0]);
  }
  else if ( (_BYTE)v123 )
  {
    v40 = 0;
  }
  *(_DWORD *)(a2 + 344) = v40;
  v124.m128i_i32[0] = *(_DWORD *)(a2 + 348);
  sub_1C14890(a1, (__int64)"PGOEpoch", (unsigned int *)&v124, &v121, 0);
  *(_DWORD *)(a2 + 348) = v124.m128i_i32[0];
  v124.m128i_i32[0] = *(_DWORD *)(a2 + 352);
  sub_1C14890(a1, (__int64)"PGOBatchSize", (unsigned int *)&v124, &v121, 0);
  *(_DWORD *)(a2 + 352) = v124.m128i_i32[0];
  v124.m128i_i32[0] = *(_DWORD *)(a2 + 356);
  sub_1C14890(a1, (__int64)"PGOCounterMemBaseVAIndex", (unsigned int *)&v124, &v121, 0);
  *(_DWORD *)(a2 + 356) = v124.m128i_i32[0];
  v124.m128i_i32[0] = *(_DWORD *)(a2 + 360);
  sub_1C14890(a1, (__int64)"PGOCounterMemOffsetIndex", (unsigned int *)&v124, &v121, 0);
  *(_DWORD *)(a2 + 360) = v124.m128i_i32[0];
  v124.m128i_i32[0] = *(_DWORD *)(a2 + 364);
  sub_1C14890(a1, (__int64)"PGONumReplicatedCopies", (unsigned int *)&v124, &v121, 0);
  *(_DWORD *)(a2 + 364) = v124.m128i_i32[0];
  v124.m128i_i32[0] = *(_DWORD *)(a2 + 368);
  sub_1C14890(a1, (__int64)"PGOBatchIdInCurAllocation", (unsigned int *)&v124, &v121, 0);
  *(_DWORD *)(a2 + 368) = v124.m128i_i32[0];
  v124.m128i_i32[0] = *(_DWORD *)(a2 + 372);
  sub_1C14890(a1, (__int64)"PGONumBatchesPerAllocation", (unsigned int *)&v124, &v121, 0);
  *(_DWORD *)(a2 + 372) = v124.m128i_i32[0];
  v124.m128i_i64[0] = *(_QWORD *)(a2 + 376);
  v124.m128i_i64[1] = a2 + 376;
  v42 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v43 = 0;
  if ( v42 )
    v43 = v124.m128i_i64[0] == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "PGOAppHash",
         0,
         v43,
         &v120,
         &v123) )
  {
    sub_1C141E0(a1, v124.m128i_i64);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v123);
  }
  else if ( v120 )
  {
    v124.m128i_i64[0] = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_QWORD *)v124.m128i_i64[1] = v124.m128i_i64[0];
  v124.m128i_i64[0] = *(_QWORD *)(a2 + 384);
  v124.m128i_i64[1] = a2 + 384;
  v44 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v45 = 0;
  if ( v44 )
    v45 = v124.m128i_i64[0] == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "PGOProfileHash",
         0,
         v45,
         &v120,
         &v123) )
  {
    sub_1C141E0(a1, v124.m128i_i64);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v123);
  }
  else if ( v120 )
  {
    v124.m128i_i64[0] = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_QWORD *)v124.m128i_i64[1] = v124.m128i_i64[0];
  v124.m128i_i64[0] = *(_QWORD *)(a2 + 392);
  v124.m128i_i64[1] = a2 + 392;
  v46 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v47 = 0;
  if ( v46 )
    v47 = v124.m128i_i64[0] == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "PGOOptionsHash",
         0,
         v47,
         &v120,
         &v123) )
  {
    sub_1C141E0(a1, v124.m128i_i64);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v123);
  }
  else if ( v120 )
  {
    v124.m128i_i64[0] = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_QWORD *)v124.m128i_i64[1] = v124.m128i_i64[0];
  v124.m128i_i32[0] = *(_DWORD *)(a2 + 400);
  sub_1C14890(a1, (__int64)"PGOFlags", (unsigned int *)&v124, &v121, 0);
  v48 = v124.m128i_i8[0];
  v49 = *(_DWORD *)(a2 + 344);
  *(_DWORD *)(a2 + 400) = v124.m128i_i32[0];
  if ( v49 == 1 )
  {
    if ( (v48 & 0x10) == 0 )
      goto LABEL_44;
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    {
      v61 = *(_QWORD *)(a2 + 408);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
              a1,
              "ZeroPPGODataPtr",
              0,
              0,
              &v123,
              &v124) )
        goto LABEL_44;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      sub_1C1C390(a1, v61);
      goto LABEL_79;
    }
    v99 = *(_QWORD *)a1;
    v127 = 0;
    v124 = 0;
    v125 = 0;
    v126 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(v99 + 120))(
           a1,
           "ZeroPPGODataPtr",
           0,
           0,
           &v120,
           &v123) )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      sub_1C1C390(a1, (__int64)&v124);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v123);
    }
    v100 = sub_16E4080(a1);
    v101 = sub_145CBF0(*(__int64 **)(v100 + 8), 56, 8);
    v102 = _mm_loadu_si128(&v124);
    v103 = _mm_loadu_si128(&v125);
    v104 = _mm_loadu_si128(&v126);
    v105 = v127;
    *(_QWORD *)(a2 + 408) = v101;
    *(__m128i *)v101 = v102;
    *(_QWORD *)(v101 + 48) = v105;
    *(__m128i *)(v101 + 16) = v103;
    *(__m128i *)(v101 + 32) = v104;
  }
  else if ( v49 == 4 && (v48 & 0x20) != 0 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    {
      v98 = *(_QWORD *)(a2 + 416);
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "BBweightPGODataPtr",
             0,
             0,
             &v123,
             &v124) )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
        sub_1C1CDE0(a1, v98);
LABEL_79:
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v124.m128i_i64[0]);
      }
    }
    else
    {
      v106 = *(_QWORD *)a1;
      v126.m128i_i64[0] = 0;
      v124 = 0;
      v125 = 0;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(v106 + 120))(
             a1,
             "BBweightPGODataPtr",
             0,
             0,
             &v120,
             &v123) )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
        sub_1C1CDE0(a1, (__int64)&v124);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v123);
      }
      v107 = sub_16E4080(a1);
      v108 = sub_145CBF0(*(__int64 **)(v107 + 8), 40, 8);
      v109 = _mm_loadu_si128(&v124);
      v110 = _mm_loadu_si128(&v125);
      v111 = v126.m128i_i64[0];
      *(_QWORD *)(a2 + 416) = v108;
      *(__m128i *)v108 = v109;
      *(_QWORD *)(v108 + 32) = v111;
      *(__m128i *)(v108 + 16) = v110;
    }
  }
LABEL_44:
  sub_1C1AF40(a1, (__int64)"OCGKnobs", (__int64 *)(a2 + 312));
  sub_1C1AF40(a1, (__int64)"OCGKnobsFile", (__int64 *)(a2 + 320));
  sub_1C1AF40(a1, (__int64)"NVVMKnobsString", (__int64 *)(a2 + 296));
  sub_1C1AF40(a1, (__int64)"OmegaKnobs", (__int64 *)(a2 + 304));
  sub_1C1AF40(a1, (__int64)"FinalizerKnobs", (__int64 *)(a2 + 328));
  v124.m128i_i64[0] = *(_QWORD *)(a2 + 336);
  v124.m128i_i64[1] = a2 + 336;
  v50 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v51 = 0;
  if ( v50 )
    v51 = v124.m128i_i64[0] == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ProgramHash",
         0,
         v51,
         &v120,
         &v123) )
  {
    sub_1C141E0(a1, v124.m128i_i64);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v123);
  }
  else if ( v120 )
  {
    v124.m128i_i64[0] = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_QWORD *)v124.m128i_i64[1] = v124.m128i_i64[0];
  sub_1C1F7B0(a1, (__int64)"AlignEntries", a2 + 200);
  sub_1C1DFB0(a1, (__int64)"RegTargets", a2 + 168);
  sub_1C1DFB0(a1, (__int64)"PerfStatsRegTargets", a2 + 184);
  sub_1C14890(a1, (__int64)"PSIGSCBOffset", (unsigned int *)(a2 + 248), &v121, 0);
  sub_1C14890(a1, (__int64)"PSIThreadMaskBaseOffset", (unsigned int *)(a2 + 252), &v121, 0);
  sub_1C14890(a1, (__int64)"MSTSSharedMemBaseGSCBByteOffset", (unsigned int *)(a2 + 256), &v121, 0);
  sub_1C14890(a1, (__int64)"MaxMSTSSharedMemSizePerCTA", (unsigned int *)(a2 + 260), &v121, 0);
  v52 = v121;
  v124.m128i_i32[0] = *(_DWORD *)(a2 + 264);
  v124.m128i_i64[1] = a2 + 264;
  v53 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v54 = 0;
  if ( v53 )
    v54 = v124.m128i_i32[0] == v52;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "DisableCBSpeculateMask",
         0,
         v54,
         &v120,
         &v123) )
  {
    sub_1C14060(a1, v124.m128i_i32);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v123);
  }
  else if ( v120 )
  {
    v124.m128i_i32[0] = v52;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_DWORD *)v124.m128i_i64[1] = v124.m128i_i32[0];
  sub_1C14890(a1, (__int64)"CTASizeX", (unsigned int *)(a2 + 96), &v121, 0);
  sub_1C14890(a1, (__int64)"CTASizeY", (unsigned int *)(a2 + 100), &v121, 0);
  sub_1C14890(a1, (__int64)"CTASizeZ", (unsigned int *)(a2 + 104), &v121, 0);
  sub_1C14890(a1, (__int64)"SMemScratchBase", (unsigned int *)(a2 + 112), &v121, 0);
  sub_1C14890(a1, (__int64)"SharedMemorySize", (unsigned int *)(a2 + 112), (_DWORD *)(a2 + 112), 0);
  sub_1C14890(a1, (__int64)"MaxFlatSMemScratchPerThread", (unsigned int *)(a2 + 116), &v121, 0);
  sub_1C1DFB0(a1, (__int64)"FlatSMemScratchPerThread", a2 + 120);
  sub_1C1DFB0(a1, (__int64)"PerfStatsFlatSMemScratchPerThread", a2 + 136);
  sub_1C14890(a1, (__int64)"SMemScratchWarpStride", (unsigned int *)(a2 + 152), &v121, 0);
  v55 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v56 = 0;
  if ( v55 )
    v56 = *(_DWORD *)(a2 + 156) == -1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SMemPerSM",
         0,
         v56,
         &v123,
         &v124) )
  {
    sub_1C14590(a1, (int *)(a2 + 156));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v124.m128i_i64[0]);
  }
  else if ( (_BYTE)v123 )
  {
    *(_DWORD *)(a2 + 156) = -1;
  }
  v57 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v58 = 0;
  if ( v57 )
    v58 = *(_DWORD *)(a2 + 160) == -1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "LMemHIReservation",
         0,
         v58,
         &v123,
         &v124) )
  {
    sub_1C14590(a1, (int *)(a2 + 160));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v124.m128i_i64[0]);
  }
  else if ( (_BYTE)v123 )
  {
    *(_DWORD *)(a2 + 160) = -1;
  }
  v59 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v60 = 0;
  if ( v59 )
    v60 = *(_DWORD *)(a2 + 268) == -1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "NumReservedUReg",
         0,
         v60,
         &v123,
         &v124) )
  {
    sub_1C14590(a1, (int *)(a2 + 268));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v124.m128i_i64[0]);
  }
  else if ( (_BYTE)v123 )
  {
    *(_DWORD *)(a2 + 268) = -1;
  }
  sub_1C14890(a1, (__int64)"NumScratchURegs", (unsigned int *)(a2 + 272), &v121, 0);
  sub_1C14890(a1, (__int64)"MaxActiveWarpsPerSM", (unsigned int *)(a2 + 280), &v121, 0);
  sub_1C14890(a1, (__int64)"NumNopsAtStart", (unsigned int *)(a2 + 284), &v121, 0);
  sub_1C1FE50(a1, (__int64)"OptimizerConstBankConstants", a2 + 216);
}
