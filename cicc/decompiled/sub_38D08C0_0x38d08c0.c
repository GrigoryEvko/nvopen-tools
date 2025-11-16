// Function: sub_38D08C0
// Address: 0x38d08c0
//
__int64 __fastcall sub_38D08C0(__int64 a1, _DWORD *a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdi
  unsigned int v30; // r13d
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // rax
  __int64 v57; // rdi
  __int64 v58; // rax
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // rdi
  __int64 v62; // rax
  __int64 v63; // rdi
  __int64 v64; // rax
  __int64 v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rdi
  __int64 v68; // rax
  __int64 v69; // rdi
  __int64 v70; // rax
  __int64 v71; // rdi
  __int64 v72; // rax
  __int64 v73; // rdi
  __int64 v74; // rax
  __int64 v75; // rdi
  __int64 v76; // rax
  __int64 v77; // rdi
  __int64 v78; // rax
  __int64 v79; // rdi
  __int64 v80; // rax
  __int64 v81; // rdi
  __int64 v82; // rax
  __int64 v83; // rdi
  __int64 v84; // rax
  __int64 v85; // rdi
  __int64 v86; // rax
  __int64 v87; // rdi
  __int64 v88; // rax
  __int64 v89; // rdi
  __int64 v90; // rax
  __int64 v91; // rdi
  __int64 v92; // rax
  __int64 v93; // rdi
  __int64 v94; // rax
  __int64 v95; // rdi
  __int64 v96; // rax
  __int64 v97; // rdi
  __int64 result; // rax
  int v99; // eax
  __int64 v100; // rax
  __int64 v101; // rdi
  __int64 v102; // rax
  __int64 v103; // rdi
  __int64 v104; // rax
  bool v105; // al
  bool v106; // al
  unsigned int v107; // [rsp+Ch] [rbp-2Ch] BYREF
  unsigned int v108; // [rsp+10h] [rbp-28h] BYREF
  _DWORD v109[9]; // [rsp+14h] [rbp-24h] BYREF

  *(_BYTE *)(a1 + 1) = 0;
  *(_QWORD *)(a1 + 408) = sub_38BFA90(*(_QWORD *)(a1 + 688), "__TEXT", 6u, "__eh_frame", 0xAu, 1744830475, 0, 3, 0);
  v3 = (unsigned int)a2[11];
  if ( (unsigned int)v3 <= 0x1E )
  {
    v4 = 1610614920;
    if ( _bittest64(&v4, v3) )
    {
      if ( a2[8] == 3 )
        *(_BYTE *)(a1 + 2) = 1;
    }
  }
  if ( a2[9] == 13 )
    *(_BYTE *)(a1 + 3) = 1;
  *(_QWORD *)(a1 + 4) = 0x100000009BLL;
  *(_QWORD *)(a1 + 12) = 0x9B00000010LL;
  v5 = a2[11];
  if ( v5 == 3 )
  {
    sub_16E2390((__int64)a2, &v107, &v108, v109);
    if ( v107 <= 8 )
LABEL_29:
      *(_BYTE *)a1 = 0;
  }
  else
  {
    if ( v5 != 11 )
      goto LABEL_9;
    sub_16E2390((__int64)a2, &v107, &v108, v109);
    if ( v107 != 10 )
    {
      v105 = v107 <= 9;
      goto LABEL_28;
    }
    if ( v108 != 5 )
    {
      v105 = v108 <= 4;
LABEL_28:
      if ( !v105 )
        goto LABEL_9;
      goto LABEL_29;
    }
  }
LABEL_9:
  v6 = sub_38BFA90(*(_QWORD *)(a1 + 688), "__TEXT", 6u, "__text", 6u, 0x80000000, 0, 1, 0);
  v7 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 24) = v6;
  v8 = sub_38BFA90(v7, "__DATA", 6u, "__data", 6u, 0, 0, 17, 0);
  *(_QWORD *)(a1 + 40) = 0;
  v9 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 32) = v8;
  v10 = sub_38BFA90(v9, "__DATA", 6u, "__thread_data", 0xDu, 17, 0, 17, 0);
  v11 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 376) = v10;
  v12 = sub_38BFA90(v11, "__DATA", 6u, "__thread_bss", 0xCu, 18, 0, 11, 0);
  v13 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 384) = v12;
  v14 = sub_38BFA90(v13, "__DATA", 6u, "__thread_vars", 0xDu, 19, 0, 17, 0);
  v15 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 496) = v14;
  v16 = sub_38BFA90(v15, "__DATA", 6u, "__thread_init", 0xDu, 21, 0, 17, 0);
  v17 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 504) = v16;
  v18 = sub_38BFA90(v17, "__TEXT", 6u, "__cstring", 9u, 2, 0, 4, 0);
  v19 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 512) = v18;
  v20 = sub_38BFA90(v19, "__TEXT", 6u, "__ustring", 9u, 0, 0, 5, 0);
  v21 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 520) = v20;
  v22 = sub_38BFA90(v21, "__TEXT", 6u, "__literal4", 0xAu, 3, 0, 7, 0);
  v23 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 584) = v22;
  v24 = sub_38BFA90(v23, "__TEXT", 6u, "__literal8", 0xAu, 4, 0, 8, 0);
  v25 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 592) = v24;
  v26 = sub_38BFA90(v25, "__TEXT", 6u, "__literal16", 0xBu, 14, 0, 9, 0);
  v27 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 600) = v26;
  v28 = sub_38BFA90(v27, "__TEXT", 6u, "__const", 7u, 0, 0, 3, 0);
  v29 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 48) = v28;
  v30 = a2[8] - 16;
  v31 = sub_38BFA90(v29, "__DATA", 6u, "__const", 7u, 0, 0, 18, 0);
  *(_QWORD *)(a1 + 544) = v31;
  if ( v30 <= 1 )
  {
    v100 = sub_38BFA90(*(_QWORD *)(a1 + 688), "__TEXT", 6u, "__textcoal_nt", 0xDu, -2147483637, 0, 1, 0);
    v101 = *(_QWORD *)(a1 + 688);
    *(_QWORD *)(a1 + 528) = v100;
    v102 = sub_38BFA90(v101, "__TEXT", 6u, "__const_coal", 0xCu, 11, 0, 3, 0);
    v103 = *(_QWORD *)(a1 + 688);
    *(_QWORD *)(a1 + 536) = v102;
    v104 = sub_38BFA90(v103, "__DATA", 6u, "__datacoal_nt", 0xDu, 11, 0, 17, 0);
    *(_QWORD *)(a1 + 552) = v104;
    *(_QWORD *)(a1 + 560) = v104;
  }
  else
  {
    v32 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 560) = v31;
    *(_QWORD *)(a1 + 528) = v32;
    *(_QWORD *)(a1 + 536) = *(_QWORD *)(a1 + 48);
    *(_QWORD *)(a1 + 552) = *(_QWORD *)(a1 + 32);
  }
  v33 = sub_38BFA90(*(_QWORD *)(a1 + 688), "__DATA", 6u, "__common", 8u, 1, 0, 13, 0);
  v34 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 568) = v33;
  v35 = sub_38BFA90(v34, "__DATA", 6u, "__bss", 5u, 1, 0, 13, 0);
  v36 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 576) = v35;
  v37 = sub_38BFA90(v36, "__DATA", 6u, "__la_symbol_ptr", 0xFu, 7, 0, 0, 0);
  v38 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 608) = v37;
  v39 = sub_38BFA90(v38, "__DATA", 6u, "__nl_symbol_ptr", 0xFu, 6, 0, 0, 0);
  v40 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 616) = v39;
  v41 = sub_38BFA90(v40, "__DATA", 6u, "__thread_ptr", 0xCu, 20, 0, 0, 0);
  v42 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 624) = v41;
  v43 = sub_38BFA90(v42, "__TEXT", 6u, "__gcc_except_tab", 0x10u, 0, 0, 18, 0);
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 56) = v43;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  v44 = (unsigned int)a2[11];
  if ( (unsigned int)v44 > 0x1E )
    goto LABEL_19;
  v45 = 1610614920;
  if ( !_bittest64(&v45, v44) )
    goto LABEL_19;
  if ( a2[8] != 3 && a2[9] != 13 )
  {
    if ( (_DWORD)v44 == 3 )
    {
      sub_16E2390((__int64)a2, &v107, &v108, v109);
      if ( v107 <= 9 )
        goto LABEL_40;
    }
    else
    {
      if ( (_DWORD)v44 != 11 )
        goto LABEL_17;
      sub_16E2390((__int64)a2, &v107, &v108, v109);
      if ( v107 != 10 )
      {
        v106 = v107 <= 9;
LABEL_37:
        if ( !v106 )
          goto LABEL_21;
LABEL_40:
        LODWORD(v44) = a2[11];
LABEL_17:
        if ( (_DWORD)v44 != 7 && (_DWORD)v44 != 29 || (unsigned int)(a2[8] - 31) > 1 )
          goto LABEL_19;
        goto LABEL_21;
      }
      if ( v108 != 6 )
      {
        v106 = v108 <= 5;
        goto LABEL_37;
      }
    }
  }
LABEL_21:
  *(_QWORD *)(a1 + 64) = sub_38BFA90(*(_QWORD *)(a1 + 688), "__LD", 4u, "__compact_unwind", 0x10u, 0x2000000, 0, 3, 0);
  v99 = a2[8];
  if ( (unsigned int)(v99 - 31) <= 1 )
    goto LABEL_22;
  if ( v99 == 3 )
  {
    *(_DWORD *)(a1 + 20) = 50331648;
    goto LABEL_19;
  }
  if ( v99 == 29 || v99 == 1 )
LABEL_22:
    *(_DWORD *)(a1 + 20) = 0x4000000;
LABEL_19:
  v46 = sub_38BFA90(*(_QWORD *)(a1 + 688), "__DWARF", 7u, "__debug_names", 0xDu, 0x2000000, 0, 0, "debug_names_begin");
  v47 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 176) = v46;
  v48 = sub_38BFA90(v47, "__DWARF", 7u, "__apple_names", 0xDu, 0x2000000, 0, 0, "names_begin");
  v49 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 184) = v48;
  v50 = sub_38BFA90(v49, "__DWARF", 7u, "__apple_objc", 0xCu, 0x2000000, 0, 0, "objc_begin");
  v51 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 192) = v50;
  v52 = sub_38BFA90(v51, "__DWARF", 7u, "__apple_namespac", 0x10u, 0x2000000, 0, 0, "namespac_begin");
  v53 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 200) = v52;
  v54 = sub_38BFA90(v53, "__DWARF", 7u, "__apple_types", 0xDu, 0x2000000, 0, 0, "types_begin");
  v55 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 208) = v54;
  v56 = sub_38BFA90(v55, "__DWARF", 7u, "__swift_ast", 0xBu, 0x2000000, 0, 0, 0);
  v57 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 336) = v56;
  v58 = sub_38BFA90(v57, "__DWARF", 7u, "__debug_abbrev", 0xEu, 0x2000000, 0, 0, "section_abbrev");
  v59 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 72) = v58;
  v60 = sub_38BFA90(v59, "__DWARF", 7u, "__debug_info", 0xCu, 0x2000000, 0, 0, "section_info");
  v61 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 80) = v60;
  v62 = sub_38BFA90(v61, "__DWARF", 7u, "__debug_line", 0xCu, 0x2000000, 0, 0, "section_line");
  v63 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 88) = v62;
  v64 = sub_38BFA90(v63, "__DWARF", 7u, "__debug_line_str", 0x10u, 0x2000000, 0, 0, "section_line_str");
  v65 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 96) = v64;
  v66 = sub_38BFA90(v65, "__DWARF", 7u, "__debug_frame", 0xDu, 0x2000000, 0, 0, 0);
  v67 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 104) = v66;
  v68 = sub_38BFA90(v67, "__DWARF", 7u, "__debug_pubnames", 0x10u, 0x2000000, 0, 0, 0);
  v69 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 168) = v68;
  v70 = sub_38BFA90(v69, "__DWARF", 7u, "__debug_pubtypes", 0x10u, 0x2000000, 0, 0, 0);
  v71 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 112) = v70;
  v72 = sub_38BFA90(v71, "__DWARF", 7u, "__debug_gnu_pubn", 0x10u, 0x2000000, 0, 0, 0);
  v73 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 320) = v72;
  v74 = sub_38BFA90(v73, "__DWARF", 7u, "__debug_gnu_pubt", 0x10u, 0x2000000, 0, 0, 0);
  v75 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 328) = v74;
  v76 = sub_38BFA90(v75, "__DWARF", 7u, "__debug_str", 0xBu, 0x2000000, 0, 0, "info_string");
  v77 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 128) = v76;
  v78 = sub_38BFA90(v77, "__DWARF", 7u, "__debug_str_offs", 0x10u, 0x2000000, 0, 0, "section_str_off");
  v79 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 272) = v78;
  v80 = sub_38BFA90(v79, "__DWARF", 7u, "__debug_loc", 0xBu, 0x2000000, 0, 0, "section_debug_loc");
  v81 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 136) = v80;
  v82 = sub_38BFA90(v81, "__DWARF", 7u, "__debug_aranges", 0xFu, 0x2000000, 0, 0, 0);
  v83 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 144) = v82;
  v84 = sub_38BFA90(v83, "__DWARF", 7u, "__debug_ranges", 0xEu, 0x2000000, 0, 0, "debug_range");
  v85 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 152) = v84;
  v86 = sub_38BFA90(v85, "__DWARF", 7u, "__debug_rnglists", 0x10u, 0x2000000, 0, 0, "debug_range");
  v87 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 288) = v86;
  v88 = sub_38BFA90(v87, "__DWARF", 7u, "__debug_macinfo", 0xFu, 0x2000000, 0, 0, "debug_macinfo");
  v89 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 160) = v88;
  v90 = sub_38BFA90(v89, "__DWARF", 7u, "__debug_inlined", 0xFu, 0x2000000, 0, 0, 0);
  v91 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 120) = v90;
  v92 = sub_38BFA90(v91, "__DWARF", 7u, "__debug_cu_index", 0x10u, 0x2000000, 0, 0, 0);
  v93 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 304) = v92;
  v94 = sub_38BFA90(v93, "__DWARF", 7u, "__debug_tu_index", 0x10u, 0x2000000, 0, 0, 0);
  v95 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 312) = v94;
  v96 = sub_38BFA90(v95, "__LLVM_STACKMAPS", 0x10u, "__llvm_stackmaps", 0x10u, 0, 0, 0, 0);
  v97 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 392) = v96;
  *(_QWORD *)(a1 + 400) = sub_38BFA90(v97, "__LLVM_FAULTMAPS", 0x10u, "__llvm_faultmaps", 0x10u, 0, 0, 0, 0);
  result = *(_QWORD *)(a1 + 496);
  *(_QWORD *)(a1 + 368) = result;
  return result;
}
