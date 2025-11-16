// Function: sub_E85910
// Address: 0xe85910
//
__int64 __fastcall sub_E85910(__int64 a1, _DWORD *a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  int v5; // eax
  unsigned __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15; // rdi
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 v18; // rax
  __int64 v19; // rdi
  unsigned __int64 v20; // rax
  __int64 v21; // rdi
  unsigned __int64 v22; // rax
  __int64 v23; // rdi
  unsigned __int64 v24; // rax
  __int64 v25; // rdi
  unsigned __int64 v26; // rax
  __int64 v27; // rdi
  unsigned __int64 v28; // rax
  __int64 v29; // rdi
  int v30; // r13d
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rdi
  unsigned __int64 v35; // rax
  __int64 v36; // rdi
  unsigned __int64 v37; // rax
  __int64 v38; // rdi
  unsigned __int64 v39; // rax
  __int64 v40; // rdi
  unsigned __int64 v41; // rax
  __int64 v42; // rdi
  unsigned __int64 v43; // rax
  __int64 v44; // rdi
  unsigned int v45; // ecx
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rax
  __int64 v48; // rdi
  unsigned __int64 v49; // rax
  __int64 v50; // rdi
  unsigned __int64 v51; // rax
  __int64 v52; // rdi
  unsigned __int64 v53; // rax
  __int64 v54; // rdi
  unsigned __int64 v55; // rax
  __int64 v56; // rdi
  unsigned __int64 v57; // rax
  __int64 v58; // rdi
  unsigned __int64 v59; // rax
  __int64 v60; // rdi
  unsigned __int64 v61; // rax
  __int64 v62; // rdi
  unsigned __int64 v63; // rax
  __int64 v64; // rdi
  unsigned __int64 v65; // rax
  __int64 v66; // rdi
  unsigned __int64 v67; // rax
  __int64 v68; // rdi
  unsigned __int64 v69; // rax
  __int64 v70; // rdi
  unsigned __int64 v71; // rax
  __int64 v72; // rdi
  unsigned __int64 v73; // rax
  __int64 v74; // rdi
  unsigned __int64 v75; // rax
  __int64 v76; // rdi
  unsigned __int64 v77; // rax
  __int64 v78; // rdi
  unsigned __int64 v79; // rax
  __int64 v80; // rdi
  unsigned __int64 v81; // rax
  __int64 v82; // rdi
  unsigned __int64 v83; // rax
  __int64 v84; // rdi
  unsigned __int64 v85; // rax
  __int64 v86; // rdi
  unsigned __int64 v87; // rax
  __int64 v88; // rdi
  unsigned __int64 v89; // rax
  __int64 v90; // rdi
  unsigned __int64 v91; // rax
  __int64 v92; // rdi
  unsigned __int64 v93; // rax
  __int64 v94; // rdi
  unsigned __int64 v95; // rax
  __int64 v96; // rdi
  unsigned __int64 v97; // rax
  __int64 v98; // rdi
  unsigned __int64 v99; // rax
  __int64 v100; // rdi
  unsigned __int64 v101; // rax
  __int64 v102; // rdi
  unsigned __int64 v103; // rax
  __int64 v104; // rdi
  unsigned __int64 v105; // rax
  __int64 v106; // rdi
  unsigned __int64 v107; // rax
  __int64 v108; // r12
  __int64 v109; // r13
  size_t v110; // rdx
  unsigned __int64 v111; // rax
  __int64 v112; // r13
  size_t v113; // rdx
  __int64 v114; // r12
  unsigned __int64 v115; // rax
  __int64 v116; // r13
  size_t v117; // rdx
  __int64 v118; // r12
  unsigned __int64 v119; // rax
  __int64 v120; // r13
  size_t v121; // rdx
  __int64 v122; // r12
  unsigned __int64 v123; // rax
  __int64 v124; // r13
  size_t v125; // rdx
  __int64 v126; // r12
  unsigned __int64 v127; // rax
  __int64 v128; // r13
  size_t v129; // rdx
  __int64 v130; // r12
  unsigned __int64 v131; // rax
  __int64 v132; // r13
  size_t v133; // rdx
  __int64 v134; // r12
  unsigned __int64 v135; // rax
  __int64 v136; // r13
  size_t v137; // rdx
  __int64 v138; // r12
  unsigned __int64 v139; // rax
  __int64 v140; // r13
  size_t v141; // rdx
  __int64 v142; // r12
  unsigned __int64 v143; // rax
  __int64 v144; // r13
  size_t v145; // rdx
  __int64 v146; // r12
  unsigned __int64 v147; // rax
  __int64 v148; // r13
  size_t v149; // rdx
  __int64 v150; // r12
  __int64 result; // rax
  int v152; // eax
  unsigned __int64 v153; // rax
  __int64 v154; // rdi
  unsigned __int64 v155; // rax
  __int64 v156; // rdi
  unsigned __int64 v157; // rax
  char v158; // al

  *(_BYTE *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 464) = sub_E6D970(
                            *(_QWORD *)(a1 + 920),
                            (__int64)"__TEXT",
                            6,
                            "__eh_frame",
                            (void *)0xA,
                            1744830475,
                            0,
                            4,
                            0);
  v3 = (unsigned int)a2[11];
  if ( (unsigned int)v3 <= 0x1F )
  {
    v4 = 3623879202LL;
    if ( _bittest64(&v4, v3) )
    {
      if ( ((a2[8] - 3) & 0xFFFFFFFD) == 0 || a2[12] == 31 )
        *(_BYTE *)(a1 + 9) = 1;
    }
  }
  v5 = sub_E65AF0(*(_QWORD *)(a1 + 920));
  switch ( v5 )
  {
    case 1:
      *(_BYTE *)(a1 + 10) = 1;
      break;
    case 2:
      v158 = 1;
      if ( a2[9] != 26 )
        v158 = *(_BYTE *)(a1 + 9);
      *(_BYTE *)(a1 + 10) = v158;
      break;
    case 0:
      *(_BYTE *)(a1 + 10) = 0;
      break;
  }
  *(_DWORD *)(a1 + 12) = 16;
  v6 = sub_E6D970(*(_QWORD *)(a1 + 920), (__int64)"__TEXT", 6, "__text", (void *)6, 0x80000000, 0, 2, 0);
  v7 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 24) = v6;
  v8 = sub_E6D970(v7, (__int64)"__DATA", 6, "__data", (void *)6, 0, 0, 19, 0);
  *(_QWORD *)(a1 + 40) = 0;
  v9 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 32) = v8;
  v10 = sub_E6D970(v9, (__int64)"__DATA", 6, "__thread_data", (void *)0xD, 17, 0, 19, 0);
  v11 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 424) = v10;
  v12 = sub_E6D970(v11, (__int64)"__DATA", 6, "__thread_bss", (void *)0xC, 18, 0, 12, 0);
  v13 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 432) = v12;
  v14 = sub_E6D970(v13, (__int64)"__DATA", 6, "__thread_vars", (void *)0xD, 19, 0, 19, 0);
  v15 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 544) = v14;
  v16 = sub_E6D970(v15, (__int64)"__DATA", 6, "__thread_init", (void *)0xD, 21, 0, 19, 0);
  v17 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 552) = v16;
  v18 = sub_E6D970(v17, (__int64)"__TEXT", 6, "__cstring", (void *)9, 2, 0, 5, 0);
  v19 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 560) = v18;
  v20 = sub_E6D970(v19, (__int64)"__TEXT", 6, "__ustring", (void *)9, 0, 0, 6, 0);
  v21 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 568) = v20;
  v22 = sub_E6D970(v21, (__int64)"__TEXT", 6, "__literal4", (void *)0xA, 3, 0, 8, 0);
  v23 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 632) = v22;
  v24 = sub_E6D970(v23, (__int64)"__TEXT", 6, "__literal8", (void *)0xA, 4, 0, 9, 0);
  v25 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 640) = v24;
  v26 = sub_E6D970(v25, (__int64)"__TEXT", 6, "__literal16", (void *)0xB, 14, 0, 10, 0);
  v27 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 648) = v26;
  v28 = sub_E6D970(v27, (__int64)"__TEXT", 6, "__const", (void *)7, 0, 0, 4, 0);
  v29 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 48) = v28;
  v30 = a2[8] - 22;
  v31 = sub_E6D970(v29, (__int64)"__DATA", 6, "__const", (void *)7, 0, 0, 20, 0);
  *(_QWORD *)(a1 + 592) = v31;
  if ( (v30 & 0xFFFFFFFD) != 0 )
  {
    v32 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 608) = v31;
    *(_QWORD *)(a1 + 576) = v32;
    *(_QWORD *)(a1 + 584) = *(_QWORD *)(a1 + 48);
    *(_QWORD *)(a1 + 600) = *(_QWORD *)(a1 + 32);
  }
  else
  {
    v153 = sub_E6D970(*(_QWORD *)(a1 + 920), (__int64)"__TEXT", 6, "__textcoal_nt", (void *)0xD, -2147483637, 0, 2, 0);
    v154 = *(_QWORD *)(a1 + 920);
    *(_QWORD *)(a1 + 576) = v153;
    v155 = sub_E6D970(v154, (__int64)"__TEXT", 6, "__const_coal", (void *)0xC, 11, 0, 4, 0);
    v156 = *(_QWORD *)(a1 + 920);
    *(_QWORD *)(a1 + 584) = v155;
    v157 = sub_E6D970(v156, (__int64)"__DATA", 6, "__datacoal_nt", (void *)0xD, 11, 0, 19, 0);
    *(_QWORD *)(a1 + 600) = v157;
    *(_QWORD *)(a1 + 608) = v157;
  }
  v33 = sub_E6D970(*(_QWORD *)(a1 + 920), (__int64)"__DATA", 6, "__common", (void *)8, 1, 0, 15, 0);
  v34 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 616) = v33;
  v35 = sub_E6D970(v34, (__int64)"__DATA", 6, "__bss", (void *)5, 1, 0, 15, 0);
  v36 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 624) = v35;
  v37 = sub_E6D970(v36, (__int64)"__DATA", 6, "__la_symbol_ptr", (void *)0xF, 7, 0, 0, 0);
  v38 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 656) = v37;
  v39 = sub_E6D970(v38, (__int64)"__DATA", 6, "__nl_symbol_ptr", (void *)0xF, 6, 0, 0, 0);
  v40 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 664) = v39;
  v41 = sub_E6D970(v40, (__int64)"__DATA", 6, "__thread_ptr", (void *)0xC, 20, 0, 0, 0);
  v42 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 672) = v41;
  v43 = sub_E6D970(v42, (__int64)"__DATA", 6, "__llvm_addrsig", (void *)0xE, 0, 0, 19, 0);
  v44 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 680) = v43;
  v46 = sub_E6D970(v44, (__int64)"__TEXT", 6, "__gcc_except_tab", (void *)0x10, 0, 0, 20, 0);
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 56) = v46;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  v45 = a2[11];
  LOBYTE(v46) = (v45 & 0xFFFFFFF7) == 1;
  if ( v45 <= 0x1F )
    LODWORD(v46) = (0xD8000020uLL >> v45) & 1 | v46;
  if ( !(_BYTE)v46 )
    goto LABEL_23;
  if ( ((a2[8] - 3) & 0xFFFFFFFD) != 0 && a2[9] != 26 )
  {
    if ( (v45 & 0xFFFFFFF7) != 1 )
    {
LABEL_19:
      if ( (v45 != 27 && v45 != 5 || (unsigned int)(a2[8] - 38) > 1) && a2[12] != 31 && v45 != 31 )
        goto LABEL_23;
      goto LABEL_49;
    }
    if ( sub_CC8200((__int64)a2, 0xAu, 6, 0) )
    {
      v45 = a2[11];
      goto LABEL_19;
    }
  }
LABEL_49:
  *(_QWORD *)(a1 + 64) = sub_E6D970(
                           *(_QWORD *)(a1 + 920),
                           (__int64)"__LD",
                           4,
                           "__compact_unwind",
                           (void *)0x10,
                           0x2000000,
                           0,
                           4,
                           0);
  v152 = a2[8];
  if ( (unsigned int)(v152 - 38) <= 1 )
    goto LABEL_50;
  if ( ((v152 - 3) & 0xFFFFFFFD) == 0 )
  {
    *(_DWORD *)(a1 + 16) = 50331648;
    goto LABEL_23;
  }
  if ( v152 == 1 || v152 == 36 )
LABEL_50:
    *(_DWORD *)(a1 + 16) = 0x4000000;
LABEL_23:
  v47 = sub_E6D970(
          *(_QWORD *)(a1 + 920),
          (__int64)"__DWARF",
          7,
          "__debug_names",
          (void *)0xD,
          0x2000000,
          0,
          0,
          "debug_names_begin");
  v48 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 192) = v47;
  v49 = sub_E6D970(v48, (__int64)"__DWARF", 7, "__apple_names", (void *)0xD, 0x2000000, 0, 0, "names_begin");
  v50 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 200) = v49;
  v51 = sub_E6D970(v50, (__int64)"__DWARF", 7, "__apple_objc", (void *)0xC, 0x2000000, 0, 0, "objc_begin");
  v52 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 208) = v51;
  v53 = sub_E6D970(v52, (__int64)"__DWARF", 7, "__apple_namespac", (void *)0x10, 0x2000000, 0, 0, "namespac_begin");
  v54 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 216) = v53;
  v55 = sub_E6D970(v54, (__int64)"__DWARF", 7, "__apple_types", (void *)0xD, 0x2000000, 0, 0, "types_begin");
  v56 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 224) = v55;
  v57 = sub_E6D970(v56, (__int64)"__DWARF", 7, "__swift_ast", (void *)0xB, 0x2000000, 0, 0, 0);
  v58 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 384) = v57;
  v59 = sub_E6D970(v58, (__int64)"__DWARF", 7, "__debug_abbrev", (void *)0xE, 0x2000000, 0, 0, "section_abbrev");
  v60 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 80) = v59;
  v61 = sub_E6D970(v60, (__int64)"__DWARF", 7, "__debug_info", (void *)0xC, 0x2000000, 0, 0, "section_info");
  v62 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 88) = v61;
  v63 = sub_E6D970(v62, (__int64)"__DWARF", 7, "__debug_line", (void *)0xC, 0x2000000, 0, 0, "section_line");
  v64 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 96) = v63;
  v65 = sub_E6D970(v64, (__int64)"__DWARF", 7, "__debug_line_str", (void *)0x10, 0x2000000, 0, 0, "section_line_str");
  v66 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 104) = v65;
  v67 = sub_E6D970(v66, (__int64)"__DWARF", 7, "__debug_frame", (void *)0xD, 0x2000000, 0, 0, "section_frame");
  v68 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 112) = v67;
  v69 = sub_E6D970(v68, (__int64)"__DWARF", 7, "__debug_pubnames", (void *)0x10, 0x2000000, 0, 0, 0);
  v70 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 184) = v69;
  v71 = sub_E6D970(v70, (__int64)"__DWARF", 7, "__debug_pubtypes", (void *)0x10, 0x2000000, 0, 0, 0);
  v72 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 120) = v71;
  v73 = sub_E6D970(v72, (__int64)"__DWARF", 7, "__debug_gnu_pubn", (void *)0x10, 0x2000000, 0, 0, 0);
  v74 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 368) = v73;
  v75 = sub_E6D970(v74, (__int64)"__DWARF", 7, "__debug_gnu_pubt", (void *)0x10, 0x2000000, 0, 0, 0);
  v76 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 376) = v75;
  v77 = sub_E6D970(v76, (__int64)"__DWARF", 7, "__debug_str", (void *)0xB, 0x2000000, 0, 0, "info_string");
  v78 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 136) = v77;
  v79 = sub_E6D970(v78, (__int64)"__DWARF", 7, "__debug_str_offs", (void *)0x10, 0x2000000, 0, 0, "section_str_off");
  v80 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 304) = v79;
  v81 = sub_E6D970(v80, (__int64)"__DWARF", 7, "__debug_addr", (void *)0xC, 0x2000000, 0, 0, "section_info");
  v82 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 312) = v81;
  v83 = sub_E6D970(v82, (__int64)"__DWARF", 7, "__debug_loc", (void *)0xB, 0x2000000, 0, 0, "section_debug_loc");
  v84 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 144) = v83;
  v85 = sub_E6D970(v84, (__int64)"__DWARF", 7, "__debug_loclists", (void *)0x10, 0x2000000, 0, 0, "section_debug_loc");
  v86 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 328) = v85;
  v87 = sub_E6D970(v86, (__int64)"__DWARF", 7, "__debug_aranges", (void *)0xF, 0x2000000, 0, 0, 0);
  v88 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 152) = v87;
  v89 = sub_E6D970(v88, (__int64)"__DWARF", 7, "__debug_ranges", (void *)0xE, 0x2000000, 0, 0, "debug_range");
  v90 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 160) = v89;
  v91 = sub_E6D970(v90, (__int64)"__DWARF", 7, "__debug_rnglists", (void *)0x10, 0x2000000, 0, 0, "debug_range");
  v92 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 320) = v91;
  v93 = sub_E6D970(v92, (__int64)"__DWARF", 7, "__debug_macinfo", (void *)0xF, 0x2000000, 0, 0, "debug_macinfo");
  v94 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 168) = v93;
  v95 = sub_E6D970(v94, (__int64)"__DWARF", 7, "__debug_macro", (void *)0xD, 0x2000000, 0, 0, "debug_macro");
  v96 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 176) = v95;
  v97 = sub_E6D970(v96, (__int64)"__DWARF", 7, "__debug_inlined", (void *)0xF, 0x2000000, 0, 0, 0);
  v98 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 128) = v97;
  v99 = sub_E6D970(v98, (__int64)"__DWARF", 7, "__debug_cu_index", (void *)0x10, 0x2000000, 0, 0, 0);
  v100 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 352) = v99;
  v101 = sub_E6D970(v100, (__int64)"__DWARF", 7, "__debug_tu_index", (void *)0x10, 0x2000000, 0, 0, 0);
  v102 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 360) = v101;
  v103 = sub_E6D970(v102, (__int64)"__LLVM_STACKMAPS", 16, "__llvm_stackmaps", (void *)0x10, 0, 0, 0, 0);
  v104 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 440) = v103;
  v105 = sub_E6D970(v104, (__int64)"__LLVM_FAULTMAPS", 16, "__llvm_faultmaps", (void *)0x10, 0, 0, 0, 0);
  v106 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 448) = v105;
  v107 = sub_E6D970(v106, (__int64)"__LLVM", 6, "__remarks", (void *)9, 0x2000000, 0, 0, 0);
  v108 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 456) = v107;
  if ( *(_QWORD *)(v108 + 16) )
  {
    v109 = *(_QWORD *)(v108 + 8);
    v110 = 0;
    if ( v109 )
      v110 = strlen(*(const char **)(v108 + 8));
    v111 = sub_E6D970(v108, v109, v110, "__swift5_fieldmd", (void *)0x10, 0, 0, 0, 0);
    v112 = *(_QWORD *)(a1 + 920);
    v113 = 0;
    *(_QWORD *)(a1 + 816) = v111;
    v114 = *(_QWORD *)(v112 + 8);
    if ( v114 )
      v113 = strlen(*(const char **)(v112 + 8));
    v115 = sub_E6D970(v112, v114, v113, "__swift5_assocty", (void *)0x10, 0, 0, 0, 0);
    v116 = *(_QWORD *)(a1 + 920);
    v117 = 0;
    *(_QWORD *)(a1 + 824) = v115;
    v118 = *(_QWORD *)(v116 + 8);
    if ( v118 )
      v117 = strlen(*(const char **)(v116 + 8));
    v119 = sub_E6D970(v116, v118, v117, "__swift5_builtin", (void *)0x10, 0, 0, 0, 0);
    v120 = *(_QWORD *)(a1 + 920);
    v121 = 0;
    *(_QWORD *)(a1 + 832) = v119;
    v122 = *(_QWORD *)(v120 + 8);
    if ( v122 )
      v121 = strlen(*(const char **)(v120 + 8));
    v123 = sub_E6D970(v120, v122, v121, "__swift5_capture", (void *)0x10, 0, 0, 0, 0);
    v124 = *(_QWORD *)(a1 + 920);
    v125 = 0;
    *(_QWORD *)(a1 + 840) = v123;
    v126 = *(_QWORD *)(v124 + 8);
    if ( v126 )
      v125 = strlen(*(const char **)(v124 + 8));
    v127 = sub_E6D970(v124, v126, v125, "__swift5_typeref", (void *)0x10, 0, 0, 0, 0);
    v128 = *(_QWORD *)(a1 + 920);
    v129 = 0;
    *(_QWORD *)(a1 + 848) = v127;
    v130 = *(_QWORD *)(v128 + 8);
    if ( v130 )
      v129 = strlen(*(const char **)(v128 + 8));
    v131 = sub_E6D970(v128, v130, v129, "__swift5_reflstr", (void *)0x10, 0, 0, 0, 0);
    v132 = *(_QWORD *)(a1 + 920);
    v133 = 0;
    *(_QWORD *)(a1 + 856) = v131;
    v134 = *(_QWORD *)(v132 + 8);
    if ( v134 )
      v133 = strlen(*(const char **)(v132 + 8));
    v135 = sub_E6D970(v132, v134, v133, "__swift5_proto", (void *)0xE, 0, 0, 0, 0);
    v136 = *(_QWORD *)(a1 + 920);
    v137 = 0;
    *(_QWORD *)(a1 + 864) = v135;
    v138 = *(_QWORD *)(v136 + 8);
    if ( v138 )
      v137 = strlen(*(const char **)(v136 + 8));
    v139 = sub_E6D970(v136, v138, v137, "__swift5_protos", (void *)0xF, 0, 0, 0, 0);
    v140 = *(_QWORD *)(a1 + 920);
    v141 = 0;
    *(_QWORD *)(a1 + 872) = v139;
    v142 = *(_QWORD *)(v140 + 8);
    if ( v142 )
      v141 = strlen(*(const char **)(v140 + 8));
    v143 = sub_E6D970(v140, v142, v141, "__swift5_acfuncs", (void *)0x10, 0, 0, 0, 0);
    v144 = *(_QWORD *)(a1 + 920);
    v145 = 0;
    *(_QWORD *)(a1 + 880) = v143;
    v146 = *(_QWORD *)(v144 + 8);
    if ( v146 )
      v145 = strlen(*(const char **)(v144 + 8));
    v147 = sub_E6D970(v144, v146, v145, "__swift5_mpenum", (void *)0xF, 0, 0, 0, 0);
    v148 = *(_QWORD *)(a1 + 920);
    v149 = 0;
    *(_QWORD *)(a1 + 888) = v147;
    v150 = *(_QWORD *)(v148 + 8);
    if ( v150 )
      v149 = strlen(*(const char **)(v148 + 8));
    *(_QWORD *)(a1 + 896) = sub_E6D970(v148, v150, v149, "__swift_ast", (void *)0xB, 0, 0, 0, 0);
  }
  result = *(_QWORD *)(a1 + 544);
  *(_QWORD *)(a1 + 416) = result;
  return result;
}
