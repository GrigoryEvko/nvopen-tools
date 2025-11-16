// Function: sub_38D17C0
// Address: 0x38d17c0
//
__int64 __fastcall sub_38D17C0(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rdi
  int v5; // eax
  int v6; // r10d
  int v7; // r15d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
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
  __int64 v25; // rdi
  __int64 v26; // rdi
  int v27; // eax
  int v28; // r14d
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rdi
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
  __int64 v63; // rax
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rax
  __int64 v72; // rdi
  __int64 v73; // rax
  __int64 v74; // rdi
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
  __int64 v90; // rdi
  __int64 v91; // rax
  __int64 v92; // rdi
  __int64 v93; // rax
  __int64 v94; // rdi
  __int64 v95; // rax
  __int64 v96; // rdi
  int v98; // eax
  int v99; // edx
  int v100; // eax
  bool v101; // zf
  int v102; // eax
  __int64 v103; // [rsp-10h] [rbp-90h]
  int v104; // [rsp+Ch] [rbp-74h]
  _QWORD v105[2]; // [rsp+10h] [rbp-70h] BYREF
  char v106; // [rsp+20h] [rbp-60h]
  char v107; // [rsp+21h] [rbp-5Fh]
  _BYTE v108[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v109; // [rsp+40h] [rbp-40h]

  switch ( *(_DWORD *)(a2 + 32) )
  {
    case 7:
    case 8:
    case 0xC:
    case 0xD:
      *(_DWORD *)(a1 + 12) = 12;
      break;
    case 0xA:
    case 0xB:
      *(_DWORD *)(a1 + 12) = 11;
      break;
    case 0x11:
    case 0x12:
    case 0x20:
      *(_DWORD *)(a1 + 12) = 27 - ((a3 == 0) - 1);
      break;
    default:
      *(_DWORD *)(a1 + 12) = 27;
      break;
  }
  v4 = *(_QWORD *)(a1 + 688);
  switch ( *(_DWORD *)(a2 + 32) )
  {
    case 1:
    case 2:
    case 0x1D:
    case 0x1E:
      if ( *(_DWORD *)(*(_QWORD *)(v4 + 16) + 348LL) == 3 )
        goto LABEL_7;
      goto LABEL_15;
    case 3:
    case 4:
      if ( !*(_BYTE *)(a1 + 684) )
        goto LABEL_5;
      *(_DWORD *)(a1 + 16) = 156;
      *(_QWORD *)(a1 + 4) = 0x1C0000009CLL;
      v5 = *(_DWORD *)(a2 + 32);
      goto LABEL_6;
    case 9:
      v101 = *(_BYTE *)(a1 + 684) == 0;
      *(_QWORD *)(a1 + 4) = 0;
      *(_QWORD *)(a1 + 12) = 0;
      if ( v101 )
        goto LABEL_13;
      *(_QWORD *)(a1 + 4) = 0x1000000090LL;
      *(_QWORD *)(a1 + 12) = 0x9000000010LL;
      v5 = *(_DWORD *)(a2 + 32);
      goto LABEL_6;
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
      *(_DWORD *)(a1 + 4) = 128;
      *(_DWORD *)(a1 + 16) = 155;
      if ( *(_DWORD *)(a2 + 44) != 5 )
        goto LABEL_13;
      *(_QWORD *)(a1 + 4) = 0x1B0000009BLL;
      v5 = *(_DWORD *)(a2 + 32);
      goto LABEL_6;
    case 0x10:
    case 0x1F:
LABEL_15:
      v98 = *(_BYTE *)(a1 + 684) != 0 ? 0x9B : 0;
      v99 = -(*(_BYTE *)(a1 + 684) == 0);
      *(_DWORD *)(a1 + 4) = v98;
      *(_DWORD *)(a1 + 16) = v98;
      *(_DWORD *)(a1 + 8) = ~(_BYTE)v99 & 0x1B;
      v5 = *(_DWORD *)(a2 + 32);
      goto LABEL_6;
    case 0x11:
    case 0x12:
      *(_DWORD *)(a1 + 16) = 148;
      *(_QWORD *)(a1 + 4) = 0x1400000094LL;
      v5 = *(_DWORD *)(a2 + 32);
      goto LABEL_6;
    case 0x17:
    case 0x19:
    case 0x1A:
      if ( !*(_BYTE *)(a1 + 684) )
        goto LABEL_5;
      *(_DWORD *)(a1 + 16) = 155;
      *(_QWORD *)(a1 + 4) = 0x1B0000009BLL;
LABEL_13:
      v5 = *(_DWORD *)(a2 + 32);
      goto LABEL_6;
    case 0x18:
      v101 = *(_BYTE *)(a1 + 684) == 0;
      *(_DWORD *)(a1 + 8) = 27;
      if ( v101 )
      {
        *(_DWORD *)(a1 + 4) = 0;
        *(_DWORD *)(a1 + 16) = 0;
      }
      else
      {
        *(_DWORD *)(a1 + 4) = 155;
        *(_DWORD *)(a1 + 16) = 155;
      }
      v5 = *(_DWORD *)(a2 + 32);
      goto LABEL_6;
    case 0x20:
      if ( *(_BYTE *)(a1 + 684) )
      {
        v100 = 156 - (a3 == 0);
        *(_DWORD *)(a1 + 4) = v100;
        *(_DWORD *)(a1 + 16) = v100;
        *(_DWORD *)(a1 + 8) = 28 - (a3 == 0);
      }
      else
      {
        v102 = a3 == 0 ? 3 : 0;
        *(_DWORD *)(a1 + 4) = v102;
        *(_DWORD *)(a1 + 8) = v102;
        *(_DWORD *)(a1 + 16) = v102;
      }
      v5 = *(_DWORD *)(a2 + 32);
      goto LABEL_6;
    case 0x2E:
LABEL_5:
      *(_QWORD *)(a1 + 4) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      v5 = *(_DWORD *)(a2 + 32);
LABEL_6:
      v6 = 1879048193;
      v7 = 2;
      if ( v5 != 32 )
        goto LABEL_7;
      goto LABEL_8;
    default:
LABEL_7:
      v6 = 1;
      v7 = (*(_DWORD *)(a2 + 44) == 14) + 2;
LABEL_8:
      v105[0] = ".bss";
      v104 = v6;
      v107 = 1;
      v106 = 3;
      v109 = 257;
      *(_QWORD *)(a1 + 40) = sub_38C3B80(v4, (__int64)v105, 8, 3, 0, (__int64)v108, -1, 0);
      v8 = *(_QWORD *)(a1 + 688);
      v109 = 257;
      v107 = 1;
      v105[0] = ".text";
      v106 = 3;
      v9 = sub_38C3B80(v8, (__int64)v105, 1, 6, 0, (__int64)v108, -1, 0);
      v10 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 24) = v9;
      v109 = 257;
      v107 = 1;
      v105[0] = ".data";
      v106 = 3;
      *(_QWORD *)(a1 + 32) = sub_38C3B80(v10, (__int64)v105, 1, 3, 0, (__int64)v108, -1, 0);
      v11 = *(_QWORD *)(a1 + 688);
      v109 = 257;
      v107 = 1;
      v105[0] = ".rodata";
      v106 = 3;
      v12 = sub_38C3B80(v11, (__int64)v105, 1, 2, 0, (__int64)v108, -1, 0);
      v13 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 48) = v12;
      v109 = 257;
      v107 = 1;
      v105[0] = ".tdata";
      v106 = 3;
      *(_QWORD *)(a1 + 376) = sub_38C3B80(v13, (__int64)v105, 1, 1027, 0, (__int64)v108, -1, 0);
      v14 = *(_QWORD *)(a1 + 688);
      v109 = 257;
      v107 = 1;
      v105[0] = ".tbss";
      v106 = 3;
      v15 = sub_38C3B80(v14, (__int64)v105, 8, 1027, 0, (__int64)v108, -1, 0);
      v16 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 384) = v15;
      v109 = 257;
      v107 = 1;
      v105[0] = ".data.rel.ro";
      v106 = 3;
      v17 = sub_38C3B80(v16, (__int64)v105, 1, 3, 0, (__int64)v108, -1, 0);
      v18 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 456) = v17;
      v109 = 257;
      v107 = 1;
      v105[0] = ".rodata.cst4";
      v106 = 3;
      v19 = sub_38C3B80(v18, (__int64)v105, 1, 18, 4, (__int64)v108, -1, 0);
      v20 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 464) = v19;
      v109 = 257;
      v107 = 1;
      v105[0] = ".rodata.cst8";
      v106 = 3;
      v21 = sub_38C3B80(v20, (__int64)v105, 1, 18, 8, (__int64)v108, -1, 0);
      v22 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 472) = v21;
      v109 = 257;
      v107 = 1;
      v105[0] = ".rodata.cst16";
      v106 = 3;
      v23 = sub_38C3B80(v22, (__int64)v105, 1, 18, 16, (__int64)v108, -1, 0);
      v24 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 480) = v23;
      v109 = 257;
      v107 = 1;
      v105[0] = ".rodata.cst32";
      v106 = 3;
      *(_QWORD *)(a1 + 488) = sub_38C3B80(v24, (__int64)v105, 1, 18, 32, (__int64)v108, -1, 0);
      v25 = *(_QWORD *)(a1 + 688);
      v105[0] = ".gcc_except_table";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      *(_QWORD *)(a1 + 56) = sub_38C3B80(v25, (__int64)v105, 1, 2, 0, (__int64)v108, -1, 0);
      v26 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 344) = 0;
      *(_QWORD *)(a1 + 352) = 0;
      v27 = *(_DWORD *)(a2 + 32);
      v107 = 1;
      v105[0] = ".debug_abbrev";
      v106 = 3;
      v28 = (unsigned int)(v27 - 10) < 4 ? 1879048222 : 1;
      v109 = 257;
      v29 = sub_38C3B80(v26, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v30 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 72) = v29;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_info";
      v106 = 3;
      v31 = sub_38C3B80(v30, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v32 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 80) = v31;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_line";
      v106 = 3;
      *(_QWORD *)(a1 + 88) = sub_38C3B80(v32, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v33 = *(_QWORD *)(a1 + 688);
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_line_str";
      v106 = 3;
      v34 = sub_38C3B80(v33, (__int64)v105, v28, 48, 1, (__int64)v108, -1, 0);
      v35 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 96) = v34;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_frame";
      v106 = 3;
      v36 = sub_38C3B80(v35, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v37 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 104) = v36;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_pubnames";
      v106 = 3;
      v38 = sub_38C3B80(v37, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v39 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 168) = v38;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_pubtypes";
      v106 = 3;
      v40 = sub_38C3B80(v39, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v41 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 112) = v40;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_gnu_pubnames";
      v106 = 3;
      v42 = sub_38C3B80(v41, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v43 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 320) = v42;
      v105[0] = ".debug_gnu_pubtypes";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      *(_QWORD *)(a1 + 328) = sub_38C3B80(v43, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v44 = *(_QWORD *)(a1 + 688);
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_str";
      v106 = 3;
      v45 = sub_38C3B80(v44, (__int64)v105, v28, 48, 1, (__int64)v108, -1, 0);
      v46 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 128) = v45;
      v105[0] = ".debug_loc";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      v47 = sub_38C3B80(v46, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v48 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 136) = v47;
      v105[0] = ".debug_aranges";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      v49 = sub_38C3B80(v48, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v50 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 144) = v49;
      v105[0] = ".debug_ranges";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      v51 = sub_38C3B80(v50, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v52 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 152) = v51;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_macinfo";
      v106 = 3;
      v53 = sub_38C3B80(v52, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v54 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 160) = v53;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_names";
      v106 = 3;
      v55 = sub_38C3B80(v54, (__int64)v105, 1, 0, 0, (__int64)v108, -1, 0);
      v56 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 176) = v55;
      v109 = 257;
      v107 = 1;
      v105[0] = ".apple_names";
      v106 = 3;
      v57 = sub_38C3B80(v56, (__int64)v105, 1, 0, 0, (__int64)v108, -1, 0);
      v58 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 184) = v57;
      v109 = 257;
      v107 = 1;
      v105[0] = ".apple_objc";
      v106 = 3;
      v59 = sub_38C3B80(v58, (__int64)v105, 1, 0, 0, (__int64)v108, -1, 0);
      v60 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 192) = v59;
      v109 = 257;
      v107 = 1;
      v105[0] = ".apple_namespaces";
      v106 = 3;
      v61 = sub_38C3B80(v60, (__int64)v105, 1, 0, 0, (__int64)v108, -1, 0);
      v62 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 200) = v61;
      v109 = 257;
      v107 = 1;
      v105[0] = ".apple_types";
      v106 = 3;
      v63 = sub_38C3B80(v62, (__int64)v105, 1, 0, 0, (__int64)v108, -1, 0);
      v64 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 208) = v63;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_str_offsets";
      v106 = 3;
      v65 = sub_38C3B80(v64, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v66 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 272) = v65;
      v105[0] = ".debug_addr";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      v67 = sub_38C3B80(v66, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v68 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 280) = v67;
      v105[0] = ".debug_rnglists";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      v69 = sub_38C3B80(v68, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v70 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 288) = v69;
      v105[0] = ".debug_info.dwo";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      v71 = sub_38C3B80(v70, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v72 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 216) = v71;
      v105[0] = ".debug_types.dwo";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      v73 = sub_38C3B80(v72, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v74 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 224) = v73;
      v105[0] = ".debug_abbrev.dwo";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      *(_QWORD *)(a1 + 232) = sub_38C3B80(v74, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v75 = *(_QWORD *)(a1 + 688);
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_str.dwo";
      v106 = 3;
      v76 = sub_38C3B80(v75, (__int64)v105, v28, 48, 1, (__int64)v108, -1, 0);
      v77 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 240) = v76;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_line.dwo";
      v106 = 3;
      v78 = sub_38C3B80(v77, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v79 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 248) = v78;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_loc.dwo";
      v106 = 3;
      v80 = sub_38C3B80(v79, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v81 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 256) = v80;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_str_offsets.dwo";
      v106 = 3;
      v82 = sub_38C3B80(v81, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v83 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 264) = v82;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_rnglists.dwo";
      v106 = 3;
      v84 = sub_38C3B80(v83, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v85 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 296) = v84;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_cu_index";
      v106 = 3;
      v86 = sub_38C3B80(v85, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v87 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 304) = v86;
      v109 = 257;
      v107 = 1;
      v105[0] = ".debug_tu_index";
      v106 = 3;
      v88 = sub_38C3B80(v87, (__int64)v105, v28, 0, 0, (__int64)v108, -1, 0);
      v89 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 312) = v88;
      v109 = 257;
      v107 = 1;
      v105[0] = ".llvm_stackmaps";
      v106 = 3;
      *(_QWORD *)(a1 + 392) = sub_38C3B80(v89, (__int64)v105, 1, 2, 0, (__int64)v108, -1, 0);
      v90 = *(_QWORD *)(a1 + 688);
      v105[0] = ".llvm_faultmaps";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      v91 = sub_38C3B80(v90, (__int64)v105, 1, 2, 0, (__int64)v108, -1, 0);
      v92 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 400) = v91;
      v105[0] = ".eh_frame";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      v93 = sub_38C3B80(v92, (__int64)v105, v104, v7, 0, (__int64)v108, -1, 0);
      v94 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 408) = v93;
      v109 = 257;
      v107 = 1;
      v105[0] = ".comment";
      v106 = 3;
      v95 = sub_38C3B80(v94, (__int64)v105, 1, 48, 1, (__int64)v108, -1, 0);
      v96 = *(_QWORD *)(a1 + 688);
      *(_QWORD *)(a1 + 672) = v95;
      v105[0] = ".stack_sizes";
      v107 = 1;
      v106 = 3;
      v109 = 257;
      *(_QWORD *)(a1 + 416) = sub_38C3B80(v96, (__int64)v105, 1, 0, 0, (__int64)v108, -1, 0);
      return v103;
  }
}
