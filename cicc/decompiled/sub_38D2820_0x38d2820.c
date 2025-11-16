// Function: sub_38D2820
// Address: 0x38d2820
//
__int64 __fastcall sub_38D2820(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  int v5; // r13d
  __int64 v6; // rax
  int v7; // ecx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdi
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
  __int64 v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rax
  __int64 v56; // rdi
  __int64 v57; // rax
  __int64 v58; // rdi
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // rdi
  __int64 result; // rax

  v3 = sub_38C23B0(*(_QWORD *)(a1 + 688), ".eh_frame", 9, -1073741760, 0x11u, 0);
  v4 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 408) = v3;
  v5 = *(_DWORD *)(a2 + 32);
  *(_BYTE *)a1 = 1;
  v6 = sub_38C23B0(v4, ".bss", 4, -1073741696, 0xDu, 0);
  v7 = 1610743840;
  v8 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 40) = v6;
  if ( v5 != 29 )
    v7 = 1610612768;
  v9 = sub_38C23B0(v8, ".text", 5, v7, 1u, 0);
  v10 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 24) = v9;
  v11 = sub_38C23B0(v10, ".data", 5, -1073741760, 0x11u, 0);
  v12 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 32) = v11;
  *(_QWORD *)(a1 + 48) = sub_38C23B0(v12, ".rdata", 6, 1073741888, 3u, 0);
  if ( *(_DWORD *)(a2 + 32) == 32 )
    *(_QWORD *)(a1 + 56) = 0;
  else
    *(_QWORD *)(a1 + 56) = sub_38C23B0(*(_QWORD *)(a1 + 688), ".gcc_except_table", 17, 1073741888, 3u, 0);
  *(_QWORD *)(a1 + 344) = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug$S", 8, 1107296320, 0, 0);
  *(_QWORD *)(a1 + 352) = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug$T", 8, 1107296320, 0, 0);
  v13 = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug$H", 8, 1107296320, 0, 0);
  v14 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 360) = v13;
  v15 = sub_38C23B0(v14, ".debug_abbrev", 13, 1107296320, 0, "section_abbrev");
  v16 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 72) = v15;
  v17 = sub_38C23B0(v16, ".debug_info", 11, 1107296320, 0, "section_info");
  v18 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 80) = v17;
  v19 = sub_38C23B0(v18, ".debug_line", 11, 1107296320, 0, "section_line");
  v20 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 88) = v19;
  *(_QWORD *)(a1 + 96) = sub_38C23B0(v20, ".debug_line_str", 15, 1107296320, 0, "section_line_str");
  *(_QWORD *)(a1 + 104) = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug_frame", 12, 1107296320, 0, 0);
  *(_QWORD *)(a1 + 168) = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug_pubnames", 15, 1107296320, 0, 0);
  *(_QWORD *)(a1 + 112) = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug_pubtypes", 15, 1107296320, 0, 0);
  *(_QWORD *)(a1 + 320) = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug_gnu_pubnames", 19, 1107296320, 0, 0);
  v21 = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug_gnu_pubtypes", 19, 1107296320, 0, 0);
  v22 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 328) = v21;
  v23 = sub_38C23B0(v22, ".debug_str", 10, 1107296320, 0, "info_string");
  v24 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 128) = v23;
  v25 = sub_38C23B0(v24, ".debug_str_offsets", 18, 1107296320, 0, "section_str_off");
  v26 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 272) = v25;
  *(_QWORD *)(a1 + 136) = sub_38C23B0(v26, ".debug_loc", 10, 1107296320, 0, "section_debug_loc");
  v27 = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug_aranges", 14, 1107296320, 0, 0);
  v28 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 144) = v27;
  v29 = sub_38C23B0(v28, ".debug_ranges", 13, 1107296320, 0, "debug_range");
  v30 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 152) = v29;
  v31 = sub_38C23B0(v30, ".debug_macinfo", 14, 1107296320, 0, "debug_macinfo");
  v32 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 160) = v31;
  v33 = sub_38C23B0(v32, ".debug_info.dwo", 15, 1107296320, 0, "section_info_dwo");
  v34 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 216) = v33;
  v35 = sub_38C23B0(v34, ".debug_types.dwo", 16, 1107296320, 0, "section_types_dwo");
  v36 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 224) = v35;
  v37 = sub_38C23B0(v36, ".debug_abbrev.dwo", 17, 1107296320, 0, "section_abbrev_dwo");
  v38 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 232) = v37;
  *(_QWORD *)(a1 + 240) = sub_38C23B0(v38, ".debug_str.dwo", 14, 1107296320, 0, "skel_string");
  v39 = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug_line.dwo", 15, 1107296320, 0, 0);
  v40 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 248) = v39;
  v41 = sub_38C23B0(v40, ".debug_loc.dwo", 14, 1107296320, 0, "skel_loc");
  v42 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 256) = v41;
  v43 = sub_38C23B0(v42, ".debug_str_offsets.dwo", 22, 1107296320, 0, "section_str_off_dwo");
  v44 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 264) = v43;
  *(_QWORD *)(a1 + 280) = sub_38C23B0(v44, ".debug_addr", 11, 1107296320, 0, "addr_sec");
  *(_QWORD *)(a1 + 304) = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug_cu_index", 15, 1107296320, 0, 0);
  v45 = sub_38C23B0(*(_QWORD *)(a1 + 688), ".debug_tu_index", 15, 1107296320, 0, 0);
  v46 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 312) = v45;
  v47 = sub_38C23B0(v46, ".debug_names", 12, 1107296320, 0, "debug_names_begin");
  v48 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 176) = v47;
  v49 = sub_38C23B0(v48, ".apple_names", 12, 1107296320, 0, "names_begin");
  v50 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 184) = v49;
  v51 = sub_38C23B0(v50, ".apple_namespaces", 17, 1107296320, 0, "namespac_begin");
  v52 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 200) = v51;
  v53 = sub_38C23B0(v52, ".apple_types", 12, 1107296320, 0, "types_begin");
  v54 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 208) = v53;
  *(_QWORD *)(a1 + 192) = sub_38C23B0(v54, ".apple_objc", 11, 1107296320, 0, "objc_begin");
  v55 = sub_38C23B0(*(_QWORD *)(a1 + 688), ".drectve", 8, 2560, 0, 0);
  v56 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 632) = v55;
  v57 = sub_38C23B0(v56, ".pdata", 6, 1073741888, 0x11u, 0);
  v58 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 640) = v57;
  *(_QWORD *)(a1 + 648) = sub_38C23B0(v58, ".xdata", 6, 1073741888, 0x11u, 0);
  *(_QWORD *)(a1 + 656) = sub_38C23B0(*(_QWORD *)(a1 + 688), ".sxdata", 7, 512, 0, 0);
  v59 = sub_38C23B0(*(_QWORD *)(a1 + 688), ".gfids$y", 8, 1073741888, 0, 0);
  v60 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 664) = v59;
  v61 = sub_38C23B0(v60, ".tls$", 5, -1073741760, 0x11u, 0);
  v62 = *(_QWORD *)(a1 + 688);
  *(_QWORD *)(a1 + 376) = v61;
  result = sub_38C23B0(v62, ".llvm_stackmaps", 15, 1073741888, 3u, 0);
  *(_QWORD *)(a1 + 392) = result;
  return result;
}
