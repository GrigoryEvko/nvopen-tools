// Function: sub_E88840
// Address: 0xe88840
//
__int64 __fastcall sub_E88840(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rax
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
  __int64 v27; // rdi
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
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rdi
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
  __int64 v63; // rax
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rdi
  __int64 result; // rax
  void *v70[4]; // [rsp+0h] [rbp-90h] BYREF
  char v71; // [rsp+20h] [rbp-70h]
  char v72; // [rsp+21h] [rbp-6Fh]
  _BYTE v73[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v74; // [rsp+50h] [rbp-40h]

  v2 = a1[115];
  v70[0] = ".text";
  v72 = 1;
  v71 = 3;
  v74 = 257;
  a1[3] = sub_E6D8A0(v2, v70, 2, 0, (__int64)v73, -1);
  v3 = a1[115];
  v74 = 257;
  v72 = 1;
  v70[0] = ".data";
  v71 = 3;
  a1[4] = sub_E6D8A0(v3, v70, 19, 0, (__int64)v73, -1);
  v4 = a1[115];
  v74 = 257;
  v72 = 1;
  v70[0] = ".debug_line";
  v71 = 3;
  a1[12] = sub_E6D8A0(v4, v70, 0, 0, (__int64)v73, -1);
  v5 = a1[115];
  v74 = 257;
  v72 = 1;
  v70[0] = ".debug_line_str";
  v71 = 3;
  a1[13] = sub_E6D8A0(v5, v70, 0, 1, (__int64)v73, -1);
  v6 = a1[115];
  v74 = 257;
  v72 = 1;
  v70[0] = ".debug_str";
  v71 = 3;
  v7 = sub_E6D8A0(v6, v70, 0, 1, (__int64)v73, -1);
  v8 = a1[115];
  v72 = 1;
  a1[17] = v7;
  v74 = 257;
  v70[0] = ".debug_loc";
  v71 = 3;
  v9 = sub_E6D8A0(v8, v70, 0, 0, (__int64)v73, -1);
  v10 = a1[115];
  v72 = 1;
  a1[18] = v9;
  v74 = 257;
  v70[0] = ".debug_abbrev";
  v71 = 3;
  v11 = sub_E6D8A0(v10, v70, 0, 0, (__int64)v73, -1);
  v12 = a1[115];
  a1[10] = v11;
  v74 = 257;
  v72 = 1;
  v70[0] = ".debug_aranges";
  v71 = 3;
  v13 = sub_E6D8A0(v12, v70, 0, 0, (__int64)v73, -1);
  v14 = a1[115];
  v72 = 1;
  a1[19] = v13;
  v74 = 257;
  v70[0] = ".debug_ranges";
  v71 = 3;
  v15 = sub_E6D8A0(v14, v70, 0, 0, (__int64)v73, -1);
  v16 = a1[115];
  v72 = 1;
  a1[20] = v15;
  v74 = 257;
  v70[0] = ".debug_macinfo";
  v71 = 3;
  v17 = sub_E6D8A0(v16, v70, 0, 0, (__int64)v73, -1);
  v18 = a1[115];
  v72 = 1;
  a1[21] = v17;
  v70[0] = ".debug_macro";
  v71 = 3;
  v74 = 257;
  v19 = sub_E6D8A0(v18, v70, 0, 0, (__int64)v73, -1);
  v20 = a1[115];
  v72 = 1;
  a1[22] = v19;
  v70[0] = ".debug_cu_index";
  v71 = 3;
  v74 = 257;
  v21 = sub_E6D8A0(v20, v70, 0, 0, (__int64)v73, -1);
  v22 = a1[115];
  v72 = 1;
  a1[44] = v21;
  v70[0] = ".debug_tu_index";
  v71 = 3;
  v74 = 257;
  v23 = sub_E6D8A0(v22, v70, 0, 0, (__int64)v73, -1);
  v24 = a1[115];
  v72 = 1;
  a1[45] = v23;
  v70[0] = ".debug_info";
  v71 = 3;
  v74 = 257;
  v25 = sub_E6D8A0(v24, v70, 0, 0, (__int64)v73, -1);
  v26 = a1[115];
  a1[11] = v25;
  v72 = 1;
  v70[0] = ".debug_frame";
  v71 = 3;
  v74 = 257;
  a1[14] = sub_E6D8A0(v26, v70, 0, 0, (__int64)v73, -1);
  v27 = a1[115];
  v74 = 257;
  v72 = 1;
  v70[0] = ".debug_pubnames";
  v71 = 3;
  a1[23] = sub_E6D8A0(v27, v70, 0, 0, (__int64)v73, -1);
  v28 = a1[115];
  v74 = 257;
  v72 = 1;
  v70[0] = ".debug_pubtypes";
  v71 = 3;
  v29 = sub_E6D8A0(v28, v70, 0, 0, (__int64)v73, -1);
  v30 = a1[115];
  a1[15] = v29;
  v74 = 257;
  v72 = 1;
  v70[0] = ".debug_gnu_pubnames";
  v71 = 3;
  v31 = sub_E6D8A0(v30, v70, 0, 0, (__int64)v73, -1);
  v32 = a1[115];
  a1[46] = v31;
  v74 = 257;
  v72 = 1;
  v70[0] = ".debug_gnu_pubtypes";
  v71 = 3;
  v33 = sub_E6D8A0(v32, v70, 0, 0, (__int64)v73, -1);
  v34 = a1[115];
  v72 = 1;
  a1[47] = v33;
  v74 = 257;
  v70[0] = ".debug_names";
  v71 = 3;
  v35 = sub_E6D8A0(v34, v70, 0, 0, (__int64)v73, -1);
  v36 = a1[115];
  v72 = 1;
  a1[24] = v35;
  v74 = 257;
  v70[0] = ".debug_str_offsets";
  v71 = 3;
  v37 = sub_E6D8A0(v36, v70, 0, 0, (__int64)v73, -1);
  v38 = a1[115];
  v72 = 1;
  a1[38] = v37;
  v74 = 257;
  v70[0] = ".debug_addr";
  v71 = 3;
  a1[39] = sub_E6D8A0(v38, v70, 0, 0, (__int64)v73, -1);
  v39 = a1[115];
  v70[0] = ".debug_rnglists";
  v72 = 1;
  v71 = 3;
  v74 = 257;
  v40 = sub_E6D8A0(v39, v70, 0, 0, (__int64)v73, -1);
  v41 = a1[115];
  v72 = 1;
  a1[40] = v40;
  v70[0] = ".debug_loclists";
  v71 = 3;
  v74 = 257;
  v42 = sub_E6D8A0(v41, v70, 0, 0, (__int64)v73, -1);
  v43 = a1[115];
  v72 = 1;
  a1[41] = v42;
  v70[0] = ".debug_info.dwo";
  v71 = 3;
  v74 = 257;
  v44 = sub_E6D8A0(v43, v70, 0, 0, (__int64)v73, -1);
  v45 = a1[115];
  v72 = 1;
  a1[29] = v44;
  v70[0] = ".debug_types.dwo";
  v71 = 3;
  v74 = 257;
  v46 = sub_E6D8A0(v45, v70, 0, 0, (__int64)v73, -1);
  v47 = a1[115];
  v72 = 1;
  a1[30] = v46;
  v70[0] = ".debug_abbrev.dwo";
  v71 = 3;
  v74 = 257;
  v48 = sub_E6D8A0(v47, v70, 0, 0, (__int64)v73, -1);
  v49 = a1[115];
  v74 = 257;
  a1[31] = v48;
  v72 = 1;
  v70[0] = ".debug_str.dwo";
  v71 = 3;
  a1[32] = sub_E6D8A0(v49, v70, 0, 1, (__int64)v73, -1);
  v50 = a1[115];
  v74 = 257;
  v72 = 1;
  v70[0] = ".debug_line.dwo";
  v71 = 3;
  v51 = sub_E6D8A0(v50, v70, 0, 0, (__int64)v73, -1);
  v52 = a1[115];
  a1[33] = v51;
  v74 = 257;
  v72 = 1;
  v70[0] = ".debug_loc.dwo";
  v71 = 3;
  v53 = sub_E6D8A0(v52, v70, 0, 0, (__int64)v73, -1);
  v54 = a1[115];
  a1[34] = v53;
  v74 = 257;
  v72 = 1;
  v70[0] = ".debug_str_offsets.dwo";
  v71 = 3;
  v55 = sub_E6D8A0(v54, v70, 0, 0, (__int64)v73, -1);
  v56 = a1[115];
  v72 = 1;
  a1[35] = v55;
  v74 = 257;
  v70[0] = ".debug_rnglists.dwo";
  v71 = 3;
  v57 = sub_E6D8A0(v56, v70, 0, 0, (__int64)v73, -1);
  v58 = a1[115];
  v72 = 1;
  a1[42] = v57;
  v74 = 257;
  v70[0] = ".debug_macinfo.dwo";
  v71 = 3;
  v59 = sub_E6D8A0(v58, v70, 0, 0, (__int64)v73, -1);
  v60 = a1[115];
  v72 = 1;
  a1[36] = v59;
  v74 = 257;
  v70[0] = ".debug_macro.dwo";
  v71 = 3;
  v61 = sub_E6D8A0(v60, v70, 0, 0, (__int64)v73, -1);
  v62 = a1[115];
  v72 = 1;
  a1[37] = v61;
  v70[0] = ".debug_loclists.dwo";
  v71 = 3;
  v74 = 257;
  v63 = sub_E6D8A0(v62, v70, 0, 0, (__int64)v73, -1);
  v64 = a1[115];
  v72 = 1;
  a1[43] = v63;
  v70[0] = ".debug_cu_index";
  v71 = 3;
  v74 = 257;
  v65 = sub_E6D8A0(v64, v70, 0, 0, (__int64)v73, -1);
  v66 = a1[115];
  v72 = 1;
  a1[44] = v65;
  v70[0] = ".debug_tu_index";
  v71 = 3;
  v74 = 257;
  v67 = sub_E6D8A0(v66, v70, 0, 0, (__int64)v73, -1);
  v68 = a1[115];
  a1[45] = v67;
  v70[0] = ".rodata.gcc_except_table";
  v72 = 1;
  v71 = 3;
  v74 = 257;
  result = sub_E6D8A0(v68, v70, 20, 0, (__int64)v73, -1);
  a1[7] = result;
  return result;
}
