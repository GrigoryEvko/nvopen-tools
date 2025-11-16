// Function: sub_E86C20
// Address: 0xe86c20
//
unsigned __int64 __fastcall sub_E86C20(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rdi
  int v5; // r11d
  int v6; // r10d
  unsigned __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int64 v13; // rax
  __int64 v14; // rdi
  unsigned __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int64 v17; // rax
  __int64 v18; // rdi
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  unsigned __int64 v21; // rax
  __int64 v22; // rdi
  unsigned __int64 v23; // rax
  __int64 v24; // rdi
  unsigned __int64 v25; // rax
  __int64 v26; // rdi
  unsigned __int64 v27; // rax
  __int64 v28; // rdi
  unsigned __int64 v29; // rax
  __int64 v30; // rdi
  int v31; // r14d
  unsigned __int64 v32; // rax
  __int64 v33; // rdi
  unsigned __int64 v34; // rax
  __int64 v35; // rdi
  unsigned __int64 v36; // rax
  __int64 v37; // rdi
  unsigned __int64 v38; // rax
  __int64 v39; // rdi
  unsigned __int64 v40; // rax
  __int64 v41; // rdi
  unsigned __int64 v42; // rax
  __int64 v43; // rdi
  unsigned __int64 v44; // rax
  __int64 v45; // rdi
  unsigned __int64 v46; // rax
  __int64 v47; // rdi
  unsigned __int64 v48; // rax
  __int64 v49; // rdi
  unsigned __int64 v50; // rax
  __int64 v51; // rdi
  unsigned __int64 v52; // rax
  __int64 v53; // rdi
  unsigned __int64 v54; // rax
  __int64 v55; // rdi
  unsigned __int64 v56; // rax
  __int64 v57; // rdi
  unsigned __int64 v58; // rax
  __int64 v59; // rdi
  unsigned __int64 v60; // rax
  __int64 v61; // rdi
  unsigned __int64 v62; // rax
  __int64 v63; // rdi
  unsigned __int64 v64; // rax
  __int64 v65; // rdi
  unsigned __int64 v66; // rax
  __int64 v67; // rdi
  unsigned __int64 v68; // rax
  __int64 v69; // rdi
  unsigned __int64 v70; // rax
  __int64 v71; // rdi
  unsigned __int64 v72; // rax
  __int64 v73; // rdi
  unsigned __int64 v74; // rax
  __int64 v75; // rdi
  unsigned __int64 v76; // rax
  __int64 v77; // rdi
  unsigned __int64 v78; // rax
  __int64 v79; // rdi
  unsigned __int64 v80; // rax
  __int64 v81; // rdi
  unsigned __int64 v82; // rax
  __int64 v83; // rdi
  unsigned __int64 v84; // rax
  __int64 v85; // rdi
  unsigned __int64 v86; // rax
  __int64 v87; // rdi
  unsigned __int64 v88; // rax
  __int64 v89; // rdi
  unsigned __int64 v90; // rax
  __int64 v91; // rdi
  unsigned __int64 v92; // rax
  __int64 v93; // rdi
  unsigned __int64 v94; // rax
  __int64 v95; // rdi
  unsigned __int64 v96; // rax
  __int64 v97; // rdi
  unsigned __int64 v98; // rax
  __int64 v99; // rdi
  unsigned __int64 v100; // rax
  __int64 v101; // rdi
  unsigned __int64 v102; // rax
  __int64 v103; // rdi
  unsigned __int64 v104; // rax
  __int64 v105; // rdi
  unsigned __int64 v106; // rax
  __int64 v107; // rdi
  unsigned __int64 v108; // rax
  __int64 v109; // rdi
  __int64 v110; // rdi
  __int64 v111; // rdi
  unsigned __int64 v112; // rax
  __int64 v113; // rdi
  unsigned __int64 v114; // rax
  __int64 v115; // rdi
  unsigned __int64 v116; // rax
  __int64 v117; // rdi
  unsigned __int64 result; // rax
  int v119; // [rsp+8h] [rbp-98h]
  unsigned int v120; // [rsp+Ch] [rbp-94h]
  size_t v121[4]; // [rsp+10h] [rbp-90h] BYREF
  char v122; // [rsp+30h] [rbp-70h]
  char v123; // [rsp+31h] [rbp-6Fh]
  _BYTE v124[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v125; // [rsp+60h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 920);
  switch ( *(_DWORD *)(a2 + 32) )
  {
    case 3:
    case 4:
    case 0x18:
    case 0x19:
    case 0x27:
      *(_DWORD *)(a1 + 12) = 27 - ((a3 == 0) - 1);
      break;
    case 8:
    case 9:
      *(_DWORD *)(a1 + 12) = 12;
      break;
    case 0xC:
      *(_DWORD *)(a1 + 12) = 16 * (*(_BYTE *)(a1 + 912) != 0);
      break;
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13:
      if ( *(_BYTE *)(a1 + 912) )
        goto LABEL_2;
      *(_DWORD *)(a1 + 12) = (*(_DWORD *)(*(_QWORD *)(v4 + 152) + 8LL) != 4) + 11;
      break;
    case 0x29:
      *(_DWORD *)(a1 + 12) = 11;
      break;
    default:
LABEL_2:
      *(_DWORD *)(a1 + 12) = 27;
      break;
  }
  if ( *(_DWORD *)(a2 + 32) == 39 )
  {
    v5 = 1879048193;
    v6 = 2;
  }
  else
  {
    v5 = 1;
    v6 = (*(_DWORD *)(a2 + 44) == 12) + 2;
  }
  v125 = 257;
  v119 = v5;
  v120 = v6;
  v123 = 1;
  v121[0] = (size_t)".bss";
  v122 = 3;
  v7 = sub_E71CB0(v4, v121, 8, 3u, 0, (__int64)v124, 0, -1, 0);
  v8 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 40) = v7;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".text";
  v122 = 3;
  v9 = sub_E71CB0(v8, v121, 1, 6u, 0, (__int64)v124, 0, -1, 0);
  v10 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 24) = v9;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".data";
  v122 = 3;
  v11 = sub_E71CB0(v10, v121, 1, 3u, 0, (__int64)v124, 0, -1, 0);
  v12 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 32) = v11;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".rodata";
  v122 = 3;
  v13 = sub_E71CB0(v12, v121, 1, 2u, 0, (__int64)v124, 0, -1, 0);
  v14 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 48) = v13;
  v121[0] = (size_t)".tdata";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v15 = sub_E71CB0(v14, v121, 1, 0x403u, 0, (__int64)v124, 0, -1, 0);
  v16 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 424) = v15;
  v121[0] = (size_t)".tbss";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v17 = sub_E71CB0(v16, v121, 8, 0x403u, 0, (__int64)v124, 0, -1, 0);
  v18 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 432) = v17;
  v121[0] = (size_t)".data.rel.ro";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v19 = sub_E71CB0(v18, v121, 1, 3u, 0, (__int64)v124, 0, -1, 0);
  v20 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 504) = v19;
  v121[0] = (size_t)".rodata.cst4";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v21 = sub_E71CB0(v20, v121, 1, 0x12u, 4, (__int64)v124, 0, -1, 0);
  v22 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 512) = v21;
  v121[0] = (size_t)".rodata.cst8";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v23 = sub_E71CB0(v22, v121, 1, 0x12u, 8, (__int64)v124, 0, -1, 0);
  v24 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 520) = v23;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".rodata.cst16";
  v122 = 3;
  v25 = sub_E71CB0(v24, v121, 1, 0x12u, 16, (__int64)v124, 0, -1, 0);
  v26 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 528) = v25;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".rodata.cst32";
  v122 = 3;
  v27 = sub_E71CB0(v26, v121, 1, 0x12u, 32, (__int64)v124, 0, -1, 0);
  v28 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 536) = v27;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".gcc_except_table";
  v122 = 3;
  v29 = sub_E71CB0(v28, v121, 1, 2u, 0, (__int64)v124, 0, -1, 0);
  v123 = 1;
  *(_QWORD *)(a1 + 56) = v29;
  v30 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  LODWORD(v29) = *(_DWORD *)(a2 + 32);
  v125 = 257;
  v122 = 3;
  v31 = (unsigned int)(v29 - 16) < 4 ? 1879048222 : 1;
  v121[0] = (size_t)".debug_abbrev";
  v32 = sub_E71CB0(v30, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v33 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 80) = v32;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_info";
  v122 = 3;
  v34 = sub_E71CB0(v33, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v35 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 88) = v34;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_line";
  v122 = 3;
  v36 = sub_E71CB0(v35, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v37 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 96) = v36;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_line_str";
  v122 = 3;
  v38 = sub_E71CB0(v37, v121, v31, 0x30u, 1, (__int64)v124, 0, -1, 0);
  v39 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 104) = v38;
  v121[0] = (size_t)".debug_frame";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v40 = sub_E71CB0(v39, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v41 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 112) = v40;
  v121[0] = (size_t)".debug_pubnames";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v42 = sub_E71CB0(v41, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v43 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 184) = v42;
  v121[0] = (size_t)".debug_pubtypes";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v44 = sub_E71CB0(v43, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v45 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 120) = v44;
  v121[0] = (size_t)".debug_gnu_pubnames";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v46 = sub_E71CB0(v45, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v47 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 368) = v46;
  v121[0] = (size_t)".debug_gnu_pubtypes";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v48 = sub_E71CB0(v47, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v49 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 376) = v48;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_str";
  v122 = 3;
  v50 = sub_E71CB0(v49, v121, v31, 0x30u, 1, (__int64)v124, 0, -1, 0);
  v51 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 136) = v50;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_loc";
  v122 = 3;
  v52 = sub_E71CB0(v51, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v53 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 144) = v52;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_aranges";
  v122 = 3;
  v54 = sub_E71CB0(v53, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v55 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 152) = v54;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_ranges";
  v122 = 3;
  v56 = sub_E71CB0(v55, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v57 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 160) = v56;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_macinfo";
  v122 = 3;
  v58 = sub_E71CB0(v57, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v59 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 168) = v58;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_macro";
  v122 = 3;
  v60 = sub_E71CB0(v59, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v61 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 176) = v60;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_names";
  v122 = 3;
  v62 = sub_E71CB0(v61, v121, 1, 0, 0, (__int64)v124, 0, -1, 0);
  v63 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 192) = v62;
  v121[0] = (size_t)".apple_names";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v64 = sub_E71CB0(v63, v121, 1, 0, 0, (__int64)v124, 0, -1, 0);
  v65 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 200) = v64;
  v121[0] = (size_t)".apple_objc";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v66 = sub_E71CB0(v65, v121, 1, 0, 0, (__int64)v124, 0, -1, 0);
  v67 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 208) = v66;
  v121[0] = (size_t)".apple_namespaces";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v68 = sub_E71CB0(v67, v121, 1, 0, 0, (__int64)v124, 0, -1, 0);
  v69 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 216) = v68;
  v121[0] = (size_t)".apple_types";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v70 = sub_E71CB0(v69, v121, 1, 0, 0, (__int64)v124, 0, -1, 0);
  v71 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 224) = v70;
  v121[0] = (size_t)".debug_str_offsets";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v72 = sub_E71CB0(v71, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v73 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 304) = v72;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_addr";
  v122 = 3;
  v74 = sub_E71CB0(v73, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v75 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 312) = v74;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_rnglists";
  v122 = 3;
  v76 = sub_E71CB0(v75, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v77 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 320) = v76;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_loclists";
  v122 = 3;
  v78 = sub_E71CB0(v77, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v79 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 328) = v78;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_info.dwo";
  v122 = 3;
  v80 = sub_E71CB0(v79, v121, v31, 0x80000000, 0, (__int64)v124, 0, -1, 0);
  v81 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 232) = v80;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_types.dwo";
  v122 = 3;
  v82 = sub_E71CB0(v81, v121, v31, 0x80000000, 0, (__int64)v124, 0, -1, 0);
  v83 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 240) = v82;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_abbrev.dwo";
  v122 = 3;
  v84 = sub_E71CB0(v83, v121, v31, 0x80000000, 0, (__int64)v124, 0, -1, 0);
  v85 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 248) = v84;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_str.dwo";
  v122 = 3;
  v86 = sub_E71CB0(v85, v121, v31, 0x80000030, 1, (__int64)v124, 0, -1, 0);
  v87 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 256) = v86;
  v121[0] = (size_t)".debug_line.dwo";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v88 = sub_E71CB0(v87, v121, v31, 0x80000000, 0, (__int64)v124, 0, -1, 0);
  v89 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 264) = v88;
  v121[0] = (size_t)".debug_loc.dwo";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v90 = sub_E71CB0(v89, v121, v31, 0x80000000, 0, (__int64)v124, 0, -1, 0);
  v91 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 272) = v90;
  v121[0] = (size_t)".debug_str_offsets.dwo";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v92 = sub_E71CB0(v91, v121, v31, 0x80000000, 0, (__int64)v124, 0, -1, 0);
  v93 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 280) = v92;
  v121[0] = (size_t)".debug_rnglists.dwo";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v94 = sub_E71CB0(v93, v121, v31, 0x80000000, 0, (__int64)v124, 0, -1, 0);
  v95 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 336) = v94;
  v121[0] = (size_t)".debug_macinfo.dwo";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v96 = sub_E71CB0(v95, v121, v31, 0x80000000, 0, (__int64)v124, 0, -1, 0);
  v97 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 288) = v96;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_macro.dwo";
  v122 = 3;
  v98 = sub_E71CB0(v97, v121, v31, 0x80000000, 0, (__int64)v124, 0, -1, 0);
  v99 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 296) = v98;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_loclists.dwo";
  v122 = 3;
  v100 = sub_E71CB0(v99, v121, v31, 0x80000000, 0, (__int64)v124, 0, -1, 0);
  v101 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 344) = v100;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_cu_index";
  v122 = 3;
  v102 = sub_E71CB0(v101, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v103 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 352) = v102;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".debug_tu_index";
  v122 = 3;
  v104 = sub_E71CB0(v103, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v105 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 360) = v104;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".llvm_stackmaps";
  v122 = 3;
  v106 = sub_E71CB0(v105, v121, 1, 2u, 0, (__int64)v124, 0, -1, 0);
  v107 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 440) = v106;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".llvm_faultmaps";
  v122 = 3;
  v108 = sub_E71CB0(v107, v121, 1, 2u, 0, (__int64)v124, 0, -1, 0);
  v109 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 448) = v108;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".eh_frame";
  v122 = 3;
  *(_QWORD *)(a1 + 464) = sub_E71CB0(v109, v121, v119, v120, 0, (__int64)v124, 0, -1, 0);
  if ( (unsigned int)(*(_DWORD *)(a2 + 32) - 42) <= 1 )
  {
    v110 = *(_QWORD *)(a1 + 920);
    v125 = 257;
    v123 = 1;
    v121[0] = (size_t)".comment";
    v122 = 3;
    *(_QWORD *)(a1 + 904) = sub_E71CB0(v110, v121, 1, 0x30u, 1, (__int64)v124, 0, -1, 0);
  }
  v111 = *(_QWORD *)(a1 + 920);
  v121[0] = (size_t)".stack_sizes";
  v123 = 1;
  v122 = 3;
  v125 = 257;
  v112 = sub_E71CB0(v111, v121, 1, 0, 0, (__int64)v124, 0, -1, 0);
  v113 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 472) = v112;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".pseudo_probe";
  v122 = 3;
  v114 = sub_E71CB0(v113, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v115 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 480) = v114;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".pseudo_probe_desc";
  v122 = 3;
  v116 = sub_E71CB0(v115, v121, v31, 0, 0, (__int64)v124, 0, -1, 0);
  v117 = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 488) = v116;
  v125 = 257;
  v123 = 1;
  v121[0] = (size_t)".llvm_stats";
  v122 = 3;
  result = sub_E71CB0(v117, v121, 1, 0, 0, (__int64)v124, 0, -1, 0);
  *(_QWORD *)(a1 + 496) = result;
  return result;
}
