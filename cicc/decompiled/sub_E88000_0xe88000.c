// Function: sub_E88000
// Address: 0xe88000
//
unsigned __int64 __fastcall sub_E88000(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  _QWORD *v4; // rdi
  int v5; // r13d
  unsigned __int64 v6; // rax
  unsigned int v7; // ecx
  _QWORD *v8; // rdi
  unsigned __int64 v9; // rax
  _QWORD *v10; // rdi
  unsigned __int64 v11; // rax
  _QWORD *v12; // rdi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  _QWORD *v15; // rdi
  unsigned __int64 v16; // rax
  _QWORD *v17; // rdi
  unsigned __int64 v18; // rax
  _QWORD *v19; // rdi
  unsigned __int64 v20; // rax
  _QWORD *v21; // rdi
  unsigned __int64 v22; // rax
  _QWORD *v23; // rdi
  unsigned __int64 v24; // rax
  _QWORD *v25; // rdi
  unsigned __int64 v26; // rax
  _QWORD *v27; // rdi
  unsigned __int64 v28; // rax
  _QWORD *v29; // rdi
  unsigned __int64 v30; // rax
  _QWORD *v31; // rdi
  unsigned __int64 v32; // rax
  _QWORD *v33; // rdi
  unsigned __int64 v34; // rax
  _QWORD *v35; // rdi
  unsigned __int64 v36; // rax
  _QWORD *v37; // rdi
  unsigned __int64 v38; // rax
  _QWORD *v39; // rdi
  unsigned __int64 v40; // rax
  _QWORD *v41; // rdi
  unsigned __int64 v42; // rax
  _QWORD *v43; // rdi
  unsigned __int64 v44; // rax
  _QWORD *v45; // rdi
  unsigned __int64 v46; // rax
  _QWORD *v47; // rdi
  unsigned __int64 v48; // rax
  _QWORD *v49; // rdi
  unsigned __int64 v50; // rax
  _QWORD *v51; // rdi
  unsigned __int64 v52; // rax
  _QWORD *v53; // rdi
  unsigned __int64 v54; // rax
  _QWORD *v55; // rdi
  unsigned __int64 v56; // rax
  _QWORD *v57; // rdi
  unsigned __int64 v58; // rax
  _QWORD *v59; // rdi
  unsigned __int64 v60; // rax
  _QWORD *v61; // rdi
  unsigned __int64 v62; // rax
  _QWORD *v63; // rdi
  unsigned __int64 v64; // rax
  _QWORD *v65; // rdi
  unsigned __int64 v66; // rax
  _QWORD *v67; // rdi
  unsigned __int64 v68; // rax
  _QWORD *v69; // rdi
  unsigned __int64 v70; // rax
  _QWORD *v71; // rdi
  unsigned __int64 v72; // rax
  _QWORD *v73; // rdi
  unsigned __int64 v74; // rax
  _QWORD *v75; // rdi
  unsigned __int64 v76; // rax
  _QWORD *v77; // rdi
  unsigned __int64 v78; // rax
  _QWORD *v79; // rdi
  unsigned __int64 v80; // rax
  _QWORD *v81; // rdi
  unsigned __int64 v82; // rax
  _QWORD *v83; // rdi
  unsigned __int64 v84; // rax
  _QWORD *v85; // rdi
  unsigned __int64 v86; // rax
  _QWORD *v87; // rdi
  unsigned __int64 v88; // rax
  _QWORD *v89; // rdi
  unsigned __int64 v90; // rax
  _QWORD *v91; // rdi
  unsigned __int64 v92; // rax
  _QWORD *v93; // rdi
  unsigned __int64 v94; // rax
  _QWORD *v95; // rdi
  unsigned __int64 v96; // rax
  _QWORD *v97; // rdi
  unsigned __int64 v98; // rax
  _QWORD *v99; // rdi
  unsigned __int64 v100; // rax
  _QWORD *v101; // rdi
  unsigned __int64 v102; // rax
  _QWORD *v103; // rdi
  unsigned __int64 v104; // rax
  _QWORD *v105; // rdi
  unsigned __int64 v106; // rax
  _QWORD *v107; // rdi
  unsigned __int64 result; // rax
  __int64 v109; // rdx

  v3 = sub_E6E280(*(_QWORD **)(a1 + 920), ".eh_frame", 9u, 0x40000040u);
  v4 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 464) = v3;
  v5 = *(_DWORD *)(a2 + 32);
  v6 = sub_E6E280(v4, ".bss", 4u, 0xC0000080);
  v7 = 1610743840;
  v8 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 40) = v6;
  if ( v5 != 36 )
    v7 = 1610612768;
  v9 = sub_E6E280(v8, ".text", 5u, v7);
  v10 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 24) = v9;
  v11 = sub_E6E280(v10, ".data", 5u, 0xC0000040);
  v12 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 32) = v11;
  *(_QWORD *)(a1 + 48) = sub_E6E280(v12, ".rdata", 6u, 0x40000040u);
  v13 = *(unsigned int *)(a2 + 32);
  if ( (unsigned int)v13 <= 0x27 && (v109 = 0x900000000ALL, _bittest64(&v109, v13)) )
  {
    *(_QWORD *)(a1 + 56) = 0;
    if ( *(_DWORD *)(a2 + 32) == 3 )
      goto LABEL_8;
  }
  else
  {
    *(_QWORD *)(a1 + 56) = sub_E6E280(*(_QWORD **)(a1 + 920), ".gcc_except_table", 0x11u, 0x40000040u);
    if ( *(_DWORD *)(a2 + 32) == 3 )
LABEL_8:
      *(_QWORD *)(a1 + 72) = sub_E6E280(*(_QWORD **)(a1 + 920), ".impcall", 8u, 0x200u);
  }
  v14 = sub_E6E280(*(_QWORD **)(a1 + 920), ".debug$S", 8u, 0x42000040u);
  v15 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 392) = v14;
  v16 = sub_E6E280(v15, ".debug$T", 8u, 0x42000040u);
  v17 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 400) = v16;
  v18 = sub_E6E280(v17, ".debug$H", 8u, 0x42000040u);
  v19 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 408) = v18;
  v20 = sub_E6E280(v19, ".debug_abbrev", 0xDu, 0x42000040u);
  v21 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 80) = v20;
  v22 = sub_E6E280(v21, ".debug_info", 0xBu, 0x42000040u);
  v23 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 88) = v22;
  v24 = sub_E6E280(v23, ".debug_line", 0xBu, 0x42000040u);
  v25 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 96) = v24;
  v26 = sub_E6E280(v25, ".debug_line_str", 0xFu, 0x42000040u);
  v27 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 104) = v26;
  v28 = sub_E6E280(v27, ".debug_frame", 0xCu, 0x42000040u);
  v29 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 112) = v28;
  v30 = sub_E6E280(v29, ".debug_pubnames", 0xFu, 0x42000040u);
  v31 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 184) = v30;
  v32 = sub_E6E280(v31, ".debug_pubtypes", 0xFu, 0x42000040u);
  v33 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 120) = v32;
  v34 = sub_E6E280(v33, ".debug_gnu_pubnames", 0x13u, 0x42000040u);
  v35 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 368) = v34;
  v36 = sub_E6E280(v35, ".debug_gnu_pubtypes", 0x13u, 0x42000040u);
  v37 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 376) = v36;
  v38 = sub_E6E280(v37, ".debug_str", 0xAu, 0x42000040u);
  v39 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 136) = v38;
  v40 = sub_E6E280(v39, ".debug_str_offsets", 0x12u, 0x42000040u);
  v41 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 304) = v40;
  v42 = sub_E6E280(v41, ".debug_loc", 0xAu, 0x42000040u);
  v43 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 144) = v42;
  v44 = sub_E6E280(v43, ".debug_loclists", 0xFu, 0x42000040u);
  v45 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 328) = v44;
  v46 = sub_E6E280(v45, ".debug_aranges", 0xEu, 0x42000040u);
  v47 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 152) = v46;
  v48 = sub_E6E280(v47, ".debug_ranges", 0xDu, 0x42000040u);
  v49 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 160) = v48;
  v50 = sub_E6E280(v49, ".debug_rnglists", 0xFu, 0x42000040u);
  v51 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 320) = v50;
  v52 = sub_E6E280(v51, ".debug_macinfo", 0xEu, 0x42000040u);
  v53 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 168) = v52;
  v54 = sub_E6E280(v53, ".debug_macro", 0xCu, 0x42000040u);
  v55 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 176) = v54;
  v56 = sub_E6E280(v55, ".debug_macinfo.dwo", 0x12u, 0x42000040u);
  v57 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 288) = v56;
  v58 = sub_E6E280(v57, ".debug_macro.dwo", 0x10u, 0x42000040u);
  v59 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 296) = v58;
  v60 = sub_E6E280(v59, ".debug_info.dwo", 0xFu, 0x42000040u);
  v61 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 232) = v60;
  v62 = sub_E6E280(v61, ".debug_types.dwo", 0x10u, 0x42000040u);
  v63 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 240) = v62;
  v64 = sub_E6E280(v63, ".debug_abbrev.dwo", 0x11u, 0x42000040u);
  v65 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 248) = v64;
  v66 = sub_E6E280(v65, ".debug_str.dwo", 0xEu, 0x42000040u);
  v67 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 256) = v66;
  v68 = sub_E6E280(v67, ".debug_line.dwo", 0xFu, 0x42000040u);
  v69 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 264) = v68;
  v70 = sub_E6E280(v69, ".debug_loc.dwo", 0xEu, 0x42000040u);
  v71 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 272) = v70;
  v72 = sub_E6E280(v71, ".debug_str_offsets.dwo", 0x16u, 0x42000040u);
  v73 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 280) = v72;
  v74 = sub_E6E280(v73, ".debug_addr", 0xBu, 0x42000040u);
  v75 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 312) = v74;
  v76 = sub_E6E280(v75, ".debug_cu_index", 0xFu, 0x42000040u);
  v77 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 352) = v76;
  v78 = sub_E6E280(v77, ".debug_tu_index", 0xFu, 0x42000040u);
  v79 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 360) = v78;
  v80 = sub_E6E280(v79, ".debug_names", 0xCu, 0x42000040u);
  v81 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 192) = v80;
  v82 = sub_E6E280(v81, ".apple_names", 0xCu, 0x42000040u);
  v83 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 200) = v82;
  v84 = sub_E6E280(v83, ".apple_namespaces", 0x11u, 0x42000040u);
  v85 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 216) = v84;
  v86 = sub_E6E280(v85, ".apple_types", 0xCu, 0x42000040u);
  v87 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 224) = v86;
  v88 = sub_E6E280(v87, ".apple_objc", 0xBu, 0x42000040u);
  v89 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 208) = v88;
  v90 = sub_E6E280(v89, ".drectve", 8u, 0xA00u);
  v91 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 688) = v90;
  v92 = sub_E6E280(v91, ".pdata", 6u, 0x40000040u);
  v93 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 696) = v92;
  v94 = sub_E6E280(v93, ".xdata", 6u, 0x40000040u);
  v95 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 704) = v94;
  v96 = sub_E6E280(v95, ".sxdata", 7u, 0x200u);
  v97 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 712) = v96;
  v98 = sub_E6E280(v97, ".gehcont$y", 0xAu, 0x40000040u);
  v99 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 720) = v98;
  v100 = sub_E6E280(v99, ".gfids$y", 8u, 0x40000040u);
  v101 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 728) = v100;
  v102 = sub_E6E280(v101, ".giats$y", 8u, 0x40000040u);
  v103 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 736) = v102;
  v104 = sub_E6E280(v103, ".gljmp$y", 8u, 0x40000040u);
  v105 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 744) = v104;
  v106 = sub_E6E280(v105, ".tls$", 5u, 0xC0000040);
  v107 = *(_QWORD **)(a1 + 920);
  *(_QWORD *)(a1 + 424) = v106;
  result = sub_E6E280(v107, ".llvm_stackmaps", 0xFu, 0x40000040u);
  *(_QWORD *)(a1 + 440) = result;
  return result;
}
