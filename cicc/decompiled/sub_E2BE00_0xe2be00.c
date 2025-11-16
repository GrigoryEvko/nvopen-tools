// Function: sub_E2BE00
// Address: 0xe2be00
//
void __fastcall sub_E2BE00(__int64 a1, __int64 *a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  char *v6; // rdi
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  char v9; // si
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  char *v12; // rdi
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  char *v17; // rdi
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  char *v20; // rdi
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  char *v23; // rdi
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  char *v28; // rdi
  unsigned __int64 v29; // rdx
  __int64 v30; // rax
  char *v31; // rdi
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  char *v34; // rdi
  unsigned __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  char *v39; // rdi
  unsigned __int64 v40; // rdx
  __int64 v41; // rax
  char *v42; // rdi
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  char *v45; // rdi
  unsigned __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  char *v50; // rdi
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  char *v53; // rdi
  __int64 v54; // rax
  unsigned __int64 v55; // rdx
  char *v56; // rdi
  unsigned __int64 v57; // rdx
  __int64 v58; // rax
  char *v59; // rdi
  __int64 v60; // rax
  unsigned __int64 v61; // rdx
  char *v62; // rdi
  unsigned __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  char *v67; // rdi
  unsigned __int64 v68; // rdx
  __int64 v69; // rax
  char *v70; // rdi
  __int64 v71; // rax
  unsigned __int64 v72; // rdx
  char *v73; // rdi
  unsigned __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // rax
  unsigned __int64 v77; // rdx
  char *v78; // rdi
  unsigned __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v81; // rax
  unsigned __int64 v82; // rdx
  char *v83; // rdi
  unsigned __int64 v84; // rdx
  __int64 v85; // rax
  __int64 v86; // rax
  unsigned __int64 v87; // rdx
  char *v88; // rdi
  unsigned __int64 v89; // rdx
  __int64 v90; // rax
  __int64 v91; // rax
  unsigned __int64 v92; // rdx
  char *v93; // rdi
  unsigned __int64 v94; // rdx
  __int64 v95; // rax
  __int64 v96; // rax
  unsigned __int64 v97; // rdx
  char *v98; // rdi
  unsigned __int64 v99; // rdx
  __int64 v100; // rax
  __int64 v101; // rax
  unsigned __int64 v102; // rdx
  char *v103; // rdi
  unsigned __int64 v104; // rdx
  __int64 v105; // rax
  __int64 v106; // rax
  unsigned __int64 v107; // rdx
  char *v108; // rdi
  unsigned __int64 v109; // rdx
  __int64 v110; // rax
  __int64 v111; // rax
  unsigned __int64 v112; // rdx
  char *v113; // rdi
  unsigned __int64 v114; // rdx
  __int64 v115; // rax
  char *v116; // rdi
  __int64 v117; // rax
  unsigned __int64 v118; // rdx
  char *v119; // rdi
  unsigned __int64 v120; // rdx
  __int64 v121; // rax
  __int64 v122; // rax
  unsigned __int64 v123; // rdx
  char *v124; // rdi
  unsigned __int64 v125; // rdx
  __int64 v126; // rax

  switch ( *(_DWORD *)(a1 + 16) )
  {
    case 0:
      v106 = a2[1];
      v107 = a2[2];
      v108 = (char *)*a2;
      if ( v106 + 4 <= v107 )
        goto LABEL_124;
      v109 = 2 * v107;
      if ( v106 + 996 > v109 )
        a2[2] = v106 + 996;
      else
        a2[2] = v109;
      v110 = realloc(v108);
      *a2 = v110;
      v108 = (char *)v110;
      if ( !v110 )
        goto LABEL_166;
      v106 = a2[1];
LABEL_124:
      *(_DWORD *)&v108[v106] = 1684631414;
      a2[1] += 4;
      break;
    case 1:
      v117 = a2[1];
      v118 = a2[2];
      v119 = (char *)*a2;
      if ( v117 + 4 <= v118 )
        goto LABEL_136;
      v120 = 2 * v118;
      if ( v117 + 996 > v120 )
        a2[2] = v117 + 996;
      else
        a2[2] = v120;
      v121 = realloc(v119);
      *a2 = v121;
      v119 = (char *)v121;
      if ( !v121 )
        goto LABEL_166;
      v117 = a2[1];
LABEL_136:
      *(_DWORD *)&v119[v117] = 1819242338;
      a2[1] += 4;
      break;
    case 2:
      v96 = a2[1];
      v97 = a2[2];
      v98 = (char *)*a2;
      if ( v96 + 4 <= v97 )
        goto LABEL_112;
      v99 = 2 * v97;
      if ( v96 + 996 > v99 )
        a2[2] = v96 + 996;
      else
        a2[2] = v99;
      v100 = realloc(v98);
      *a2 = v100;
      v98 = (char *)v100;
      if ( !v100 )
        goto LABEL_166;
      v96 = a2[1];
LABEL_112:
      *(_DWORD *)&v98[v96] = 1918986339;
      a2[1] += 4;
      break;
    case 3:
      v122 = a2[1];
      v123 = a2[2];
      v124 = (char *)*a2;
      if ( v122 + 11 <= v123 )
        goto LABEL_142;
      v125 = 2 * v123;
      if ( v122 + 1003 > v125 )
        a2[2] = v122 + 1003;
      else
        a2[2] = v125;
      v126 = realloc(v124);
      *a2 = v126;
      v124 = (char *)v126;
      if ( !v126 )
        goto LABEL_166;
      v122 = a2[1];
LABEL_142:
      qmemcpy(&v124[v122], "signed char", 11);
      a2[1] += 11;
      break;
    case 4:
      v101 = a2[1];
      v102 = a2[2];
      v103 = (char *)*a2;
      if ( v101 + 13 <= v102 )
        goto LABEL_118;
      v104 = 2 * v102;
      if ( v101 + 1005 > v104 )
        a2[2] = v101 + 1005;
      else
        a2[2] = v104;
      v105 = realloc(v103);
      *a2 = v105;
      v103 = (char *)v105;
      if ( !v105 )
        goto LABEL_166;
      v101 = a2[1];
LABEL_118:
      qmemcpy(&v103[v101], "unsigned char", 13);
      a2[1] += 13;
      break;
    case 5:
      v111 = a2[1];
      v112 = a2[2];
      v113 = (char *)*a2;
      if ( v111 + 7 <= v112 )
        goto LABEL_130;
      v114 = 2 * v112;
      if ( v111 + 999 > v114 )
        a2[2] = v111 + 999;
      else
        a2[2] = v114;
      v115 = realloc(v113);
      *a2 = v115;
      v113 = (char *)v115;
      if ( !v115 )
        goto LABEL_166;
      v111 = a2[1];
LABEL_130:
      v116 = &v113[v111];
      *(_DWORD *)v116 = 1918986339;
      *((_WORD *)v116 + 2) = 24376;
      v116[6] = 116;
      a2[1] += 7;
      break;
    case 6:
      v91 = a2[1];
      v92 = a2[2];
      v93 = (char *)*a2;
      if ( v91 + 8 <= v92 )
        goto LABEL_106;
      v94 = 2 * v92;
      if ( v91 + 1000 > v94 )
        a2[2] = v91 + 1000;
      else
        a2[2] = v94;
      v95 = realloc(v93);
      *a2 = v95;
      v93 = (char *)v95;
      if ( !v95 )
        goto LABEL_166;
      v91 = a2[1];
LABEL_106:
      *(_QWORD *)&v93[v91] = 0x745F363172616863LL;
      a2[1] += 8;
      break;
    case 7:
      v86 = a2[1];
      v87 = a2[2];
      v88 = (char *)*a2;
      if ( v86 + 8 <= v87 )
        goto LABEL_100;
      v89 = 2 * v87;
      if ( v86 + 1000 > v89 )
        a2[2] = v86 + 1000;
      else
        a2[2] = v89;
      v90 = realloc(v88);
      *a2 = v90;
      v88 = (char *)v90;
      if ( !v90 )
        goto LABEL_166;
      v86 = a2[1];
LABEL_100:
      *(_QWORD *)&v88[v86] = 0x745F323372616863LL;
      a2[1] += 8;
      break;
    case 8:
      v65 = a2[1];
      v66 = a2[2];
      v67 = (char *)*a2;
      if ( v65 + 5 <= v66 )
        goto LABEL_76;
      v68 = 2 * v66;
      if ( v65 + 997 > v68 )
        a2[2] = v65 + 997;
      else
        a2[2] = v68;
      v69 = realloc(v67);
      *a2 = v69;
      v67 = (char *)v69;
      if ( !v69 )
        goto LABEL_166;
      v65 = a2[1];
LABEL_76:
      v70 = &v67[v65];
      *(_DWORD *)v70 = 1919903859;
      v70[4] = 116;
      a2[1] += 5;
      break;
    case 9:
      v76 = a2[1];
      v77 = a2[2];
      v78 = (char *)*a2;
      if ( v76 + 14 <= v77 )
        goto LABEL_88;
      v79 = 2 * v77;
      if ( v76 + 1006 > v79 )
        a2[2] = v76 + 1006;
      else
        a2[2] = v79;
      v80 = realloc(v78);
      *a2 = v80;
      v78 = (char *)v80;
      if ( !v80 )
        goto LABEL_166;
      v76 = a2[1];
LABEL_88:
      qmemcpy(&v78[v76], "unsigned short", 14);
      a2[1] += 14;
      break;
    case 0xA:
      v54 = a2[1];
      v55 = a2[2];
      v56 = (char *)*a2;
      if ( v54 + 3 <= v55 )
        goto LABEL_64;
      v57 = 2 * v55;
      if ( v54 + 995 > v57 )
        a2[2] = v54 + 995;
      else
        a2[2] = v57;
      v58 = realloc(v56);
      *a2 = v58;
      v56 = (char *)v58;
      if ( !v58 )
        goto LABEL_166;
      v54 = a2[1];
LABEL_64:
      v59 = &v56[v54];
      *(_WORD *)v59 = 28265;
      v59[2] = 116;
      a2[1] += 3;
      break;
    case 0xB:
      v81 = a2[1];
      v82 = a2[2];
      v83 = (char *)*a2;
      if ( v81 + 12 <= v82 )
        goto LABEL_94;
      v84 = 2 * v82;
      if ( v81 + 1004 > v84 )
        a2[2] = v81 + 1004;
      else
        a2[2] = v84;
      v85 = realloc(v83);
      *a2 = v85;
      v83 = (char *)v85;
      if ( !v85 )
        goto LABEL_166;
      v81 = a2[1];
LABEL_94:
      qmemcpy(&v83[v81], "unsigned int", 12);
      a2[1] += 12;
      break;
    case 0xC:
      v60 = a2[1];
      v61 = a2[2];
      v62 = (char *)*a2;
      if ( v60 + 4 <= v61 )
        goto LABEL_70;
      v63 = 2 * v61;
      if ( v60 + 996 > v63 )
        a2[2] = v60 + 996;
      else
        a2[2] = v63;
      v64 = realloc(v62);
      *a2 = v64;
      v62 = (char *)v64;
      if ( !v64 )
        goto LABEL_166;
      v60 = a2[1];
LABEL_70:
      *(_DWORD *)&v62[v60] = 1735290732;
      a2[1] += 4;
      break;
    case 0xD:
      v71 = a2[1];
      v72 = a2[2];
      v73 = (char *)*a2;
      if ( v71 + 13 <= v72 )
        goto LABEL_82;
      v74 = 2 * v72;
      if ( v71 + 1005 > v74 )
        a2[2] = v71 + 1005;
      else
        a2[2] = v74;
      v75 = realloc(v73);
      *a2 = v75;
      v73 = (char *)v75;
      if ( !v75 )
        goto LABEL_166;
      v71 = a2[1];
LABEL_82:
      qmemcpy(&v73[v71], "unsigned long", 13);
      a2[1] += 13;
      break;
    case 0xE:
      v48 = a2[1];
      v49 = a2[2];
      v50 = (char *)*a2;
      if ( v48 + 7 <= v49 )
        goto LABEL_58;
      v51 = 2 * v49;
      if ( v48 + 999 > v51 )
        a2[2] = v48 + 999;
      else
        a2[2] = v51;
      v52 = realloc(v50);
      *a2 = v52;
      v50 = (char *)v52;
      if ( !v52 )
        goto LABEL_166;
      v48 = a2[1];
LABEL_58:
      v53 = &v50[v48];
      *(_DWORD *)v53 = 1852399455;
      *((_WORD *)v53 + 2) = 13940;
      v53[6] = 52;
      a2[1] += 7;
      break;
    case 0xF:
      v43 = a2[1];
      v44 = a2[2];
      v45 = (char *)*a2;
      if ( v43 + 16 <= v44 )
        goto LABEL_52;
      v46 = 2 * v44;
      if ( v43 + 1008 > v46 )
        a2[2] = v43 + 1008;
      else
        a2[2] = v46;
      v47 = realloc(v45);
      *a2 = v47;
      v45 = (char *)v47;
      if ( !v47 )
        goto LABEL_166;
      v43 = a2[1];
LABEL_52:
      *(__m128i *)&v45[v43] = _mm_load_si128((const __m128i *)&xmmword_3F7CA60);
      a2[1] += 16;
      break;
    case 0x10:
      v26 = a2[1];
      v27 = a2[2];
      v28 = (char *)*a2;
      if ( v26 + 7 <= v27 )
        goto LABEL_34;
      v29 = 2 * v27;
      if ( v26 + 999 > v29 )
        a2[2] = v26 + 999;
      else
        a2[2] = v29;
      v30 = realloc(v28);
      *a2 = v30;
      v28 = (char *)v30;
      if ( !v30 )
        goto LABEL_166;
      v26 = a2[1];
LABEL_34:
      v31 = &v28[v26];
      *(_DWORD *)v31 = 1634231159;
      *((_WORD *)v31 + 2) = 24434;
      v31[6] = 116;
      a2[1] += 7;
      break;
    case 0x11:
      v37 = a2[1];
      v38 = a2[2];
      v39 = (char *)*a2;
      if ( v37 + 5 <= v38 )
        goto LABEL_46;
      v40 = 2 * v38;
      if ( v37 + 997 > v40 )
        a2[2] = v37 + 997;
      else
        a2[2] = v40;
      v41 = realloc(v39);
      *a2 = v41;
      v39 = (char *)v41;
      if ( !v41 )
        goto LABEL_166;
      v37 = a2[1];
LABEL_46:
      v42 = &v39[v37];
      *(_DWORD *)v42 = 1634692198;
      v42[4] = 116;
      a2[1] += 5;
      break;
    case 0x12:
      v15 = a2[1];
      v16 = a2[2];
      v17 = (char *)*a2;
      if ( v15 + 6 <= v16 )
        goto LABEL_22;
      v18 = 2 * v16;
      if ( v15 + 998 > v18 )
        a2[2] = v15 + 998;
      else
        a2[2] = v18;
      v19 = realloc(v17);
      *a2 = v19;
      v17 = (char *)v19;
      if ( !v19 )
        goto LABEL_166;
      v15 = a2[1];
LABEL_22:
      v20 = &v17[v15];
      *(_DWORD *)v20 = 1651863396;
      *((_WORD *)v20 + 2) = 25964;
      a2[1] += 6;
      break;
    case 0x13:
      v32 = a2[1];
      v33 = a2[2];
      v34 = (char *)*a2;
      if ( v32 + 11 <= v33 )
        goto LABEL_40;
      v35 = 2 * v33;
      if ( v32 + 1003 > v35 )
        a2[2] = v32 + 1003;
      else
        a2[2] = v35;
      v36 = realloc(v34);
      *a2 = v36;
      v34 = (char *)v36;
      if ( !v36 )
        goto LABEL_166;
      v32 = a2[1];
LABEL_40:
      qmemcpy(&v34[v32], "long double", 11);
      a2[1] += 11;
      break;
    case 0x14:
      v21 = a2[1];
      v22 = a2[2];
      v23 = (char *)*a2;
      if ( v21 + 14 <= v22 )
        goto LABEL_28;
      v24 = 2 * v22;
      if ( v21 + 1006 > v24 )
        a2[2] = v21 + 1006;
      else
        a2[2] = v24;
      v25 = realloc(v23);
      *a2 = v25;
      v23 = (char *)v25;
      if ( !v25 )
        goto LABEL_166;
      v21 = a2[1];
LABEL_28:
      qmemcpy(&v23[v21], "std::nullptr_t", 14);
      a2[1] += 14;
      break;
    case 0x15:
      v10 = a2[1];
      v11 = a2[2];
      v12 = (char *)*a2;
      if ( v10 + 4 <= v11 )
        goto LABEL_16;
      v13 = 2 * v11;
      if ( v10 + 996 > v13 )
        a2[2] = v10 + 996;
      else
        a2[2] = v13;
      v14 = realloc(v12);
      *a2 = v14;
      v12 = (char *)v14;
      if ( !v14 )
        goto LABEL_166;
      v10 = a2[1];
LABEL_16:
      *(_DWORD *)&v12[v10] = 1869903201;
      a2[1] += 4;
      break;
    case 0x16:
      v4 = a2[1];
      v5 = a2[2];
      v6 = (char *)*a2;
      if ( v4 + 14 <= v5 )
        goto LABEL_7;
      v7 = 2 * v5;
      if ( v4 + 1006 > v7 )
        a2[2] = v4 + 1006;
      else
        a2[2] = v7;
      v8 = realloc(v6);
      *a2 = v8;
      v6 = (char *)v8;
      if ( !v8 )
LABEL_166:
        abort();
      v4 = a2[1];
LABEL_7:
      qmemcpy(&v6[v4], "decltype(auto)", 14);
      a2[1] += 14;
      break;
    default:
      break;
  }
  v9 = *(_BYTE *)(a1 + 12);
  if ( v9 )
    sub_E2A820(a2, v9, 1, 0);
}
