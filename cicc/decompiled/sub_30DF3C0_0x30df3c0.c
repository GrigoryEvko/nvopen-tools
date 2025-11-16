// Function: sub_30DF3C0
// Address: 0x30df3c0
//
__int64 __fastcall sub_30DF3C0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // r15
  int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // rax
  int v15; // ecx
  __int64 v16; // r13
  __m128i *v17; // rdx
  __m128i si128; // xmm0
  const char *v19; // rax
  size_t v20; // rdx
  void *v21; // rdi
  unsigned __int8 *v22; // rsi
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  const char *v25; // rax
  size_t v26; // rdx
  _WORD *v27; // rdi
  unsigned __int8 *v28; // rsi
  unsigned __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rdi
  _BYTE *v58; // rax
  __int64 v59; // rsi
  __int64 v60; // r12
  __int64 v61; // r15
  unsigned __int64 v62; // rdi
  __int64 *v63; // r12
  unsigned __int64 v64; // r12
  unsigned __int64 v65; // rdi
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  void *v70; // rdx
  __int64 v71; // rax
  _WORD *v72; // rdx
  size_t v73; // [rsp+8h] [rbp-508h]
  size_t v74; // [rsp+8h] [rbp-508h]
  __int64 v76; // [rsp+38h] [rbp-4D8h]
  __int64 v77; // [rsp+48h] [rbp-4C8h]
  __int64 v78; // [rsp+58h] [rbp-4B8h]
  __int64 v79; // [rsp+60h] [rbp-4B0h]
  __int64 v81; // [rsp+88h] [rbp-488h] BYREF
  __int64 v82[2]; // [rsp+90h] [rbp-480h] BYREF
  __int64 *v83; // [rsp+A0h] [rbp-470h]
  _QWORD v84[2]; // [rsp+B0h] [rbp-460h] BYREF
  __int64 (__fastcall *v85)(_QWORD *, _QWORD *, int); // [rsp+C0h] [rbp-450h]
  __int64 (__fastcall *v86)(__int64 *, __int64); // [rsp+C8h] [rbp-448h]
  int v87[24]; // [rsp+D0h] [rbp-440h] BYREF
  __int64 v88; // [rsp+130h] [rbp-3E0h] BYREF
  unsigned __int64 v89; // [rsp+138h] [rbp-3D8h]
  char v90; // [rsp+148h] [rbp-3C8h]
  char v91; // [rsp+158h] [rbp-3B8h]
  int v92; // [rsp+160h] [rbp-3B0h]
  __int64 v93; // [rsp+168h] [rbp-3A8h]
  __int64 v94; // [rsp+170h] [rbp-3A0h]
  __int64 v95; // [rsp+178h] [rbp-398h]
  unsigned int v96; // [rsp+180h] [rbp-390h]
  _QWORD v97[9]; // [rsp+190h] [rbp-380h] BYREF
  __int64 v98; // [rsp+1D8h] [rbp-338h]
  __int64 v99; // [rsp+1E0h] [rbp-330h]
  unsigned __int8 *v100; // [rsp+1F0h] [rbp-320h]
  unsigned __int8 v101; // [rsp+1FCh] [rbp-314h]
  unsigned int v102; // [rsp+210h] [rbp-300h]
  __int64 v103; // [rsp+220h] [rbp-2F0h]
  unsigned int v104; // [rsp+230h] [rbp-2E0h]
  __int64 v105; // [rsp+240h] [rbp-2D0h]
  unsigned int v106; // [rsp+250h] [rbp-2C0h]
  __int64 v107; // [rsp+260h] [rbp-2B0h]
  unsigned int v108; // [rsp+270h] [rbp-2A0h]
  __int64 v109; // [rsp+280h] [rbp-290h]
  unsigned int v110; // [rsp+290h] [rbp-280h]
  unsigned __int64 v111; // [rsp+2A0h] [rbp-270h]
  char v112; // [rsp+2B4h] [rbp-25Ch]
  __int64 v113; // [rsp+340h] [rbp-1D0h]
  unsigned int v114; // [rsp+350h] [rbp-1C0h]
  unsigned __int64 v115; // [rsp+368h] [rbp-1A8h]
  char v116; // [rsp+37Ch] [rbp-194h]
  unsigned int v117; // [rsp+400h] [rbp-110h]
  unsigned int v118; // [rsp+404h] [rbp-10Ch]
  unsigned int v119; // [rsp+408h] [rbp-108h]
  unsigned int v120; // [rsp+40Ch] [rbp-104h]
  unsigned int v121; // [rsp+410h] [rbp-100h]
  unsigned int v122; // [rsp+414h] [rbp-FCh]
  char v123; // [rsp+418h] [rbp-F8h]
  int v124; // [rsp+41Ch] [rbp-F4h]
  int v125; // [rsp+420h] [rbp-F0h]
  int v126; // [rsp+424h] [rbp-ECh]
  __int64 v127; // [rsp+428h] [rbp-E8h]
  __int64 v128; // [rsp+438h] [rbp-D8h]
  unsigned int v129; // [rsp+448h] [rbp-C8h]
  int v130; // [rsp+450h] [rbp-C0h]
  int v131; // [rsp+45Ch] [rbp-B4h]
  unsigned __int64 v132; // [rsp+470h] [rbp-A0h]
  unsigned int v133; // [rsp+478h] [rbp-98h]
  unsigned __int64 v134; // [rsp+480h] [rbp-90h]
  unsigned int v135; // [rsp+488h] [rbp-88h]
  char v136; // [rsp+490h] [rbp-80h]
  unsigned int v137; // [rsp+49Ch] [rbp-74h]
  unsigned int v138; // [rsp+4A0h] [rbp-70h]
  __int64 v139; // [rsp+4B0h] [rbp-60h]
  unsigned int v140; // [rsp+4C0h] [rbp-50h]
  __int64 (__fastcall **v141)(); // [rsp+4C8h] [rbp-48h] BYREF

  LOBYTE(v97[0]) = 1;
  LOBYTE(qword_5030CC8) = 1;
  if ( !qword_5030CF8 )
    sub_4263D6(a1, a2, a3);
  qword_5030D00(&unk_5030CE8, v97);
  v84[0] = a4;
  v6 = *(_QWORD *)(a3 + 40);
  v86 = sub_30D1110;
  v85 = sub_30D1190;
  v88 = v6;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  sub_D84780(&v88);
  sub_DF9330(&v81, v6 + 312);
  sub_30D6B30((__int64)v87);
  v7 = *(_QWORD *)(a3 + 80);
  v76 = a3 + 72;
  if ( v7 != a3 + 72 )
  {
    while ( 1 )
    {
      if ( !v7 )
        BUG();
      v8 = *(_QWORD *)(v7 + 32);
      v9 = v7 + 24;
      if ( v7 + 24 != v8 )
        break;
LABEL_58:
      v7 = *(_QWORD *)(v7 + 8);
      if ( v76 == v7 )
        goto LABEL_59;
    }
    v79 = v7;
    while ( 1 )
    {
      if ( !v8 )
        BUG();
      if ( (unsigned __int8)(*(_BYTE *)(v8 - 24) - 34) > 0x33u )
        goto LABEL_6;
      v10 = 0x8000000000041LL;
      if ( !_bittest64(&v10, (unsigned int)*(unsigned __int8 *)(v8 - 24) - 34) )
        goto LABEL_6;
      v11 = *(_QWORD *)(v8 - 56);
      if ( !v11 || *(_BYTE *)v11 || *(_QWORD *)(v11 + 24) != *(_QWORD *)(v8 + 56) || sub_B2FC80(*(_QWORD *)(v8 - 56)) )
        goto LABEL_6;
      sub_1049690(v82, v11);
      sub_30D4900(
        (__int64)v97,
        v11,
        v8 - 24,
        v87,
        (__int64)&v81,
        (__int64)&v88,
        (__int64)sub_26B9F70,
        (__int64)v84,
        0,
        v78,
        0,
        v77,
        (__int64)v82,
        1,
        0);
      sub_30D2590((__int64)v97, (__int64)v100, v98);
      v130 += v126 + v125;
      v12 = sub_30D4FE0((__int64 *)v97[1], v100, v99);
      v13 = v98;
      v14 = v131 + (__int64)-v12;
      if ( v14 >= 0x80000000LL )
        v14 = 0x7FFFFFFF;
      if ( v14 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        LODWORD(v14) = 0x80000000;
      v131 = v14;
      v15 = v14;
      if ( ((*(_WORD *)(v98 + 2) >> 4) & 0x3FF) == 9 )
      {
        v15 = v14 + 2000;
        v131 = v14 + 2000;
      }
      if ( v130 > v15 || v123 )
      {
        if ( *(_BYTE *)(v127 + 66) )
          goto LABEL_23;
        if ( !sub_B2DCC0(v98) )
          break;
      }
LABEL_25:
      v16 = *a2;
      v17 = *(__m128i **)(*a2 + 32);
      if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v17 <= 0x17u )
      {
        v16 = sub_CB6200(*a2, "      Analyzing call of ", 0x18u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_44CDD40);
        v17[1].m128i_i64[0] = 0x20666F206C6C6163LL;
        *v17 = si128;
        *(_QWORD *)(v16 + 32) += 24LL;
      }
      v19 = sub_BD5D20(v11);
      v21 = *(void **)(v16 + 32);
      v22 = (unsigned __int8 *)v19;
      v23 = *(_QWORD *)(v16 + 24) - (_QWORD)v21;
      if ( v23 < v20 )
      {
        v68 = sub_CB6200(v16, v22, v20);
        v21 = *(void **)(v68 + 32);
        v16 = v68;
        v23 = *(_QWORD *)(v68 + 24) - (_QWORD)v21;
      }
      else if ( v20 )
      {
        v73 = v20;
        memcpy(v21, v22, v20);
        v69 = *(_QWORD *)(v16 + 24);
        v70 = (void *)(*(_QWORD *)(v16 + 32) + v73);
        *(_QWORD *)(v16 + 32) = v70;
        v21 = v70;
        v23 = v69 - (_QWORD)v70;
      }
      if ( v23 <= 0xB )
      {
        v16 = sub_CB6200(v16, "... (caller:", 0xCu);
      }
      else
      {
        qmemcpy(v21, "... (caller:", 12);
        *(_QWORD *)(v16 + 32) += 12LL;
      }
      v24 = sub_B491C0(v8 - 24);
      v25 = sub_BD5D20(v24);
      v27 = *(_WORD **)(v16 + 32);
      v28 = (unsigned __int8 *)v25;
      v29 = *(_QWORD *)(v16 + 24) - (_QWORD)v27;
      if ( v29 < v26 )
      {
        v67 = sub_CB6200(v16, v28, v26);
        v27 = *(_WORD **)(v67 + 32);
        v16 = v67;
        v29 = *(_QWORD *)(v67 + 24) - (_QWORD)v27;
      }
      else if ( v26 )
      {
        v74 = v26;
        memcpy(v27, v28, v26);
        v71 = *(_QWORD *)(v16 + 24);
        v72 = (_WORD *)(*(_QWORD *)(v16 + 32) + v74);
        *(_QWORD *)(v16 + 32) = v72;
        v27 = v72;
        v29 = v71 - (_QWORD)v72;
      }
      if ( v29 <= 1 )
      {
        sub_CB6200(v16, (unsigned __int8 *)")\n", 2u);
      }
      else
      {
        *v27 = 2601;
        *(_QWORD *)(v16 + 32) += 2LL;
      }
      v30 = *a2;
      if ( (_BYTE)qword_5030CC8 )
        sub_A68C30(v98, *a2, (__int64)&v141, 0, 0);
      v31 = sub_904010(v30, "      NumConstantArgs: ");
      v32 = sub_CB59D0(v31, v117);
      sub_904010(v32, "\n");
      v33 = sub_904010(v30, "      NumConstantOffsetPtrArgs: ");
      v34 = sub_CB59D0(v33, v118);
      sub_904010(v34, "\n");
      v35 = sub_904010(v30, "      NumAllocaArgs: ");
      v36 = sub_CB59D0(v35, v119);
      sub_904010(v36, "\n");
      v37 = sub_904010(v30, "      NumConstantPtrCmps: ");
      v38 = sub_CB59D0(v37, v120);
      sub_904010(v38, "\n");
      v39 = sub_904010(v30, "      NumConstantPtrDiffs: ");
      v40 = sub_CB59D0(v39, v121);
      sub_904010(v40, "\n");
      v41 = sub_904010(v30, "      NumInstructionsSimplified: ");
      v42 = sub_CB59D0(v41, v122);
      sub_904010(v42, "\n");
      v43 = sub_904010(v30, "      NumInstructions: ");
      v44 = sub_CB59D0(v43, v102);
      sub_904010(v44, "\n");
      v45 = sub_904010(v30, "      SROACostSavings: ");
      v46 = sub_CB59D0(v45, v137);
      sub_904010(v46, "\n");
      v47 = sub_904010(v30, "      SROACostSavingsLost: ");
      v48 = sub_CB59D0(v47, v138);
      sub_904010(v48, "\n");
      v49 = sub_904010(v30, "      LoadEliminationCost: ");
      v50 = sub_CB59F0(v49, v124);
      sub_904010(v50, "\n");
      v51 = sub_904010(v30, "      ContainsNoDuplicateCall: ");
      v52 = sub_CB59F0(v51, v101);
      sub_904010(v52, "\n");
      v53 = sub_904010(v30, "      Cost: ");
      v54 = sub_CB59F0(v53, v131);
      sub_904010(v54, "\n");
      v55 = sub_904010(v30, "      Threshold: ");
      v56 = sub_CB59F0(v55, v130);
      sub_904010(v56, "\n");
      v57 = *a2;
      v58 = *(_BYTE **)(*a2 + 32);
      if ( *(_BYTE **)(*a2 + 24) == v58 )
      {
        sub_CB6200(v57, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v58 = 10;
        ++*(_QWORD *)(v57 + 32);
      }
      v97[0] = off_49D8928;
      v141 = off_4A325F8;
      nullsub_35();
      sub_C7D6A0(v139, 16LL * v140, 8);
      if ( v136 )
      {
        v136 = 0;
        if ( v135 > 0x40 && v134 )
          j_j___libc_free_0_0(v134);
        if ( v133 > 0x40 && v132 )
          j_j___libc_free_0_0(v132);
      }
      sub_C7D6A0(v128, 24LL * v129, 8);
      v97[0] = off_49D8850;
      if ( !v116 )
        _libc_free(v115);
      sub_C7D6A0(v113, 16LL * v114, 8);
      if ( !v112 )
        _libc_free(v111);
      v59 = v110;
      if ( v110 )
      {
        v60 = v109;
        v61 = v109 + 32LL * v110;
        do
        {
          if ( *(_QWORD *)v60 != -8192 && *(_QWORD *)v60 != -4096 && *(_DWORD *)(v60 + 24) > 0x40u )
          {
            v62 = *(_QWORD *)(v60 + 16);
            if ( v62 )
              j_j___libc_free_0_0(v62);
          }
          v60 += 32;
        }
        while ( v61 != v60 );
        v59 = v110;
      }
      sub_C7D6A0(v109, 32 * v59, 8);
      sub_C7D6A0(v107, 8LL * v108, 8);
      sub_C7D6A0(v105, 16LL * v106, 8);
      sub_C7D6A0(v103, 16LL * v104, 8);
      v63 = v83;
      if ( v83 )
      {
        sub_FDC110(v83);
        j_j___libc_free_0((unsigned __int64)v63);
        v8 = *(_QWORD *)(v8 + 8);
        if ( v9 == v8 )
        {
LABEL_57:
          v7 = v79;
          goto LABEL_58;
        }
      }
      else
      {
LABEL_6:
        v8 = *(_QWORD *)(v8 + 8);
        if ( v9 == v8 )
          goto LABEL_57;
      }
    }
    v13 = v98;
LABEL_23:
    if ( v13 + 72 != (*(_QWORD *)(v13 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
      sub_30DC7E0(v97);
    goto LABEL_25;
  }
LABEL_59:
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  sub_DFE7B0(&v81);
  sub_C7D6A0(v94, 16LL * v96, 8);
  v64 = v89;
  if ( v89 )
  {
    v65 = *(_QWORD *)(v89 + 8);
    if ( v65 )
      j_j___libc_free_0(v65);
    j_j___libc_free_0(v64);
  }
  if ( v85 )
    v85(v84, v84, 3);
  return a1;
}
