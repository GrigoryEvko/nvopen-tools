// Function: sub_375DA40
// Address: 0x375da40
//
__int64 __fastcall sub_375DA40(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 result; // rax
  unsigned int v6; // r13d
  int v7; // r15d
  int v8; // r12d
  char v9; // dl
  __int64 v10; // r10
  int v11; // esi
  unsigned int v12; // edi
  _DWORD *v13; // rax
  int v14; // r11d
  _DWORD *v15; // rax
  __int64 v16; // r8
  int v17; // eax
  unsigned int v18; // edx
  int *v19; // rsi
  int v20; // edi
  unsigned int v21; // eax
  __int64 v22; // r8
  int v23; // eax
  unsigned int v24; // edx
  int *v25; // rsi
  int v26; // edi
  unsigned int v27; // eax
  __int64 v28; // r8
  int v29; // eax
  unsigned int v30; // edx
  int *v31; // rsi
  int v32; // edi
  unsigned int v33; // eax
  __int64 v34; // r8
  int v35; // eax
  unsigned int v36; // edx
  int *v37; // rsi
  int v38; // edi
  unsigned int v39; // eax
  __int64 v40; // r8
  int v41; // eax
  unsigned int v42; // edx
  int *v43; // rsi
  int v44; // edi
  unsigned int v45; // eax
  __int64 v46; // r8
  int v47; // eax
  unsigned int v48; // edx
  int *v49; // rsi
  int v50; // edi
  unsigned int v51; // eax
  __int64 v52; // r8
  int v53; // eax
  unsigned int v54; // edx
  int *v55; // rsi
  int v56; // edi
  unsigned int v57; // eax
  __int64 v58; // r8
  int v59; // eax
  unsigned int v60; // edx
  int *v61; // rsi
  int v62; // edi
  unsigned int v63; // eax
  __int64 v64; // r8
  int v65; // eax
  unsigned int v66; // edx
  int *v67; // rsi
  int v68; // edi
  unsigned int v69; // eax
  __int64 v70; // r8
  unsigned int v71; // edx
  int *v72; // rsi
  int v73; // edi
  unsigned int v74; // eax
  __int64 v75; // r8
  int v76; // edx
  int v77; // r9d
  __int64 v78; // rsi
  unsigned int v79; // eax
  int v80; // edx
  unsigned int v81; // eax
  int v82; // eax
  int v83; // eax
  int v84; // eax
  int v85; // eax
  int v86; // eax
  int v87; // eax
  int v88; // eax
  int v89; // eax
  int v90; // eax
  unsigned int v91; // esi
  unsigned int v92; // eax
  _DWORD *v93; // r9
  int v94; // edi
  unsigned int v95; // r10d
  int v96; // esi
  int v97; // r9d
  int v98; // esi
  int v99; // r9d
  int v100; // esi
  int v101; // r9d
  int v102; // esi
  int v103; // r9d
  int v104; // esi
  int v105; // r9d
  int v106; // esi
  int v107; // r9d
  int v108; // esi
  int v109; // r9d
  int v110; // esi
  int v111; // r9d
  int v112; // esi
  int v113; // r9d
  int v114; // esi
  int v115; // r9d
  int v116; // ecx
  __int64 v117; // rsi
  int v118; // eax
  unsigned int v119; // edx
  int v120; // edi
  __int64 v121; // rsi
  int v122; // eax
  unsigned int v123; // edx
  int v124; // edi
  int v125; // ecx
  _DWORD *v126; // r11
  int v127; // eax
  int v128; // eax
  int v129; // ecx
  int v131; // [rsp+8h] [rbp-38h]
  int v132; // [rsp+Ch] [rbp-34h]

  result = *(unsigned int *)(a2 + 68);
  v132 = result;
  if ( !(_DWORD)result )
    return result;
  v6 = 0;
  v131 = (a2 >> 9) ^ (a2 >> 4);
  do
  {
    v7 = sub_375D5B0(a1, a3, v6);
    result = sub_375D5B0(a1, a2, v6);
    v8 = result;
    if ( (_DWORD)result == v7 )
      goto LABEL_48;
    v9 = *(_BYTE *)(a1 + 1536) & 1;
    if ( v9 )
    {
      v10 = a1 + 1544;
      v11 = 7;
    }
    else
    {
      v91 = *(_DWORD *)(a1 + 1552);
      v10 = *(_QWORD *)(a1 + 1544);
      if ( !v91 )
      {
        v92 = *(_DWORD *)(a1 + 1536);
        ++*(_QWORD *)(a1 + 1528);
        v93 = 0;
        v94 = (v92 >> 1) + 1;
LABEL_83:
        v95 = 3 * v91;
        goto LABEL_84;
      }
      v11 = v91 - 1;
    }
    v12 = v11 & (37 * result);
    v13 = (_DWORD *)(v10 + 8LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
    {
LABEL_7:
      v15 = v13 + 1;
      goto LABEL_8;
    }
    v116 = 1;
    v93 = 0;
    while ( v14 != -1 )
    {
      if ( v14 == -2 && !v93 )
        v93 = v13;
      v12 = v11 & (v116 + v12);
      v13 = (_DWORD *)(v10 + 8LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
        goto LABEL_7;
      ++v116;
    }
    v95 = 24;
    v91 = 8;
    if ( !v93 )
      v93 = v13;
    v92 = *(_DWORD *)(a1 + 1536);
    ++*(_QWORD *)(a1 + 1528);
    v94 = (v92 >> 1) + 1;
    if ( !v9 )
    {
      v91 = *(_DWORD *)(a1 + 1552);
      goto LABEL_83;
    }
LABEL_84:
    if ( 4 * v94 >= v95 )
    {
      sub_375BDE0(a1 + 1528, 2 * v91);
      if ( (*(_BYTE *)(a1 + 1536) & 1) != 0 )
      {
        v117 = a1 + 1544;
        v118 = 7;
      }
      else
      {
        v127 = *(_DWORD *)(a1 + 1552);
        v117 = *(_QWORD *)(a1 + 1544);
        if ( !v127 )
          goto LABEL_169;
        v118 = v127 - 1;
      }
      v119 = v118 & (37 * v8);
      v93 = (_DWORD *)(v117 + 8LL * v119);
      v120 = *v93;
      if ( v8 == *v93 )
        goto LABEL_140;
      v129 = 1;
      v126 = 0;
      while ( v120 != -1 )
      {
        if ( v120 == -2 && !v126 )
          v126 = v93;
        v119 = v118 & (v129 + v119);
        v93 = (_DWORD *)(v117 + 8LL * v119);
        v120 = *v93;
        if ( v8 == *v93 )
          goto LABEL_140;
        ++v129;
      }
LABEL_146:
      if ( v126 )
        v93 = v126;
LABEL_140:
      v92 = *(_DWORD *)(a1 + 1536);
      goto LABEL_86;
    }
    if ( v91 - *(_DWORD *)(a1 + 1540) - v94 <= v91 >> 3 )
    {
      sub_375BDE0(a1 + 1528, v91);
      if ( (*(_BYTE *)(a1 + 1536) & 1) != 0 )
      {
        v121 = a1 + 1544;
        v122 = 7;
      }
      else
      {
        v128 = *(_DWORD *)(a1 + 1552);
        v121 = *(_QWORD *)(a1 + 1544);
        if ( !v128 )
        {
LABEL_169:
          *(_DWORD *)(a1 + 1536) = (2 * (*(_DWORD *)(a1 + 1536) >> 1) + 2) | *(_DWORD *)(a1 + 1536) & 1;
          BUG();
        }
        v122 = v128 - 1;
      }
      v123 = v122 & (37 * v8);
      v93 = (_DWORD *)(v121 + 8LL * v123);
      v124 = *v93;
      if ( v8 == *v93 )
        goto LABEL_140;
      v125 = 1;
      v126 = 0;
      while ( v124 != -1 )
      {
        if ( !v126 && v124 == -2 )
          v126 = v93;
        v123 = v122 & (v125 + v123);
        v93 = (_DWORD *)(v121 + 8LL * v123);
        v124 = *v93;
        if ( v8 == *v93 )
          goto LABEL_140;
        ++v125;
      }
      goto LABEL_146;
    }
LABEL_86:
    *(_DWORD *)(a1 + 1536) = (2 * (v92 >> 1) + 2) | v92 & 1;
    if ( *v93 != -1 )
      --*(_DWORD *)(a1 + 1540);
    *v93 = v8;
    v15 = v93 + 1;
    v93[1] = 0;
LABEL_8:
    *v15 = v7;
    if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
    {
      v16 = a1 + 520;
      v17 = 7;
    }
    else
    {
      v90 = *(_DWORD *)(a1 + 528);
      v16 = *(_QWORD *)(a1 + 520);
      if ( !v90 )
        goto LABEL_12;
      v17 = v90 - 1;
    }
    v18 = v17 & (37 * v8);
    v19 = (int *)(v16 + 24LL * v18);
    v20 = *v19;
    if ( v8 == *v19 )
    {
LABEL_11:
      *v19 = -2;
      v21 = *(_DWORD *)(a1 + 512);
      ++*(_DWORD *)(a1 + 516);
      *(_DWORD *)(a1 + 512) = (2 * (v21 >> 1) - 2) | v21 & 1;
    }
    else
    {
      v96 = 1;
      while ( v20 != -1 )
      {
        v97 = v96 + 1;
        v18 = v17 & (v96 + v18);
        v19 = (int *)(v16 + 24LL * v18);
        v20 = *v19;
        if ( v8 == *v19 )
          goto LABEL_11;
        v96 = v97;
      }
    }
LABEL_12:
    if ( (*(_BYTE *)(a1 + 720) & 1) != 0 )
    {
      v22 = a1 + 728;
      v23 = 7;
    }
    else
    {
      v89 = *(_DWORD *)(a1 + 736);
      v22 = *(_QWORD *)(a1 + 728);
      if ( !v89 )
        goto LABEL_16;
      v23 = v89 - 1;
    }
    v24 = v23 & (37 * v8);
    v25 = (int *)(v22 + 8LL * v24);
    v26 = *v25;
    if ( v8 == *v25 )
    {
LABEL_15:
      *v25 = -2;
      v27 = *(_DWORD *)(a1 + 720);
      ++*(_DWORD *)(a1 + 724);
      *(_DWORD *)(a1 + 720) = (2 * (v27 >> 1) - 2) | v27 & 1;
    }
    else
    {
      v114 = 1;
      while ( v26 != -1 )
      {
        v115 = v114 + 1;
        v24 = v23 & (v114 + v24);
        v25 = (int *)(v22 + 8LL * v24);
        v26 = *v25;
        if ( v8 == *v25 )
          goto LABEL_15;
        v114 = v115;
      }
    }
LABEL_16:
    if ( (*(_BYTE *)(a1 + 800) & 1) != 0 )
    {
      v28 = a1 + 808;
      v29 = 7;
    }
    else
    {
      v88 = *(_DWORD *)(a1 + 816);
      v28 = *(_QWORD *)(a1 + 808);
      if ( !v88 )
        goto LABEL_20;
      v29 = v88 - 1;
    }
    v30 = v29 & (37 * v8);
    v31 = (int *)(v28 + 12LL * v30);
    v32 = *v31;
    if ( v8 == *v31 )
    {
LABEL_19:
      *v31 = -2;
      v33 = *(_DWORD *)(a1 + 800);
      ++*(_DWORD *)(a1 + 804);
      *(_DWORD *)(a1 + 800) = (2 * (v33 >> 1) - 2) | v33 & 1;
    }
    else
    {
      v112 = 1;
      while ( v32 != -1 )
      {
        v113 = v112 + 1;
        v30 = v29 & (v112 + v30);
        v31 = (int *)(v28 + 12LL * v30);
        v32 = *v31;
        if ( v8 == *v31 )
          goto LABEL_19;
        v112 = v113;
      }
    }
LABEL_20:
    if ( (*(_BYTE *)(a1 + 912) & 1) != 0 )
    {
      v34 = a1 + 920;
      v35 = 7;
    }
    else
    {
      v87 = *(_DWORD *)(a1 + 928);
      v34 = *(_QWORD *)(a1 + 920);
      if ( !v87 )
        goto LABEL_24;
      v35 = v87 - 1;
    }
    v36 = v35 & (37 * v8);
    v37 = (int *)(v34 + 8LL * v36);
    v38 = *v37;
    if ( v8 == *v37 )
    {
LABEL_23:
      *v37 = -2;
      v39 = *(_DWORD *)(a1 + 912);
      ++*(_DWORD *)(a1 + 916);
      *(_DWORD *)(a1 + 912) = (2 * (v39 >> 1) - 2) | v39 & 1;
    }
    else
    {
      v110 = 1;
      while ( v38 != -1 )
      {
        v111 = v110 + 1;
        v36 = v35 & (v110 + v36);
        v37 = (int *)(v34 + 8LL * v36);
        v38 = *v37;
        if ( v8 == *v37 )
          goto LABEL_23;
        v110 = v111;
      }
    }
LABEL_24:
    if ( (*(_BYTE *)(a1 + 992) & 1) != 0 )
    {
      v40 = a1 + 1000;
      v41 = 7;
    }
    else
    {
      v86 = *(_DWORD *)(a1 + 1008);
      v40 = *(_QWORD *)(a1 + 1000);
      if ( !v86 )
        goto LABEL_28;
      v41 = v86 - 1;
    }
    v42 = v41 & (37 * v8);
    v43 = (int *)(v40 + 8LL * v42);
    v44 = *v43;
    if ( v8 == *v43 )
    {
LABEL_27:
      *v43 = -2;
      v45 = *(_DWORD *)(a1 + 992);
      ++*(_DWORD *)(a1 + 996);
      *(_DWORD *)(a1 + 992) = (2 * (v45 >> 1) - 2) | v45 & 1;
    }
    else
    {
      v108 = 1;
      while ( v44 != -1 )
      {
        v109 = v108 + 1;
        v42 = v41 & (v108 + v42);
        v43 = (int *)(v40 + 8LL * v42);
        v44 = *v43;
        if ( v8 == *v43 )
          goto LABEL_27;
        v108 = v109;
      }
    }
LABEL_28:
    if ( (*(_BYTE *)(a1 + 1072) & 1) != 0 )
    {
      v46 = a1 + 1080;
      v47 = 7;
    }
    else
    {
      v85 = *(_DWORD *)(a1 + 1088);
      v46 = *(_QWORD *)(a1 + 1080);
      if ( !v85 )
        goto LABEL_32;
      v47 = v85 - 1;
    }
    v48 = v47 & (37 * v8);
    v49 = (int *)(v46 + 8LL * v48);
    v50 = *v49;
    if ( v8 == *v49 )
    {
LABEL_31:
      *v49 = -2;
      v51 = *(_DWORD *)(a1 + 1072);
      ++*(_DWORD *)(a1 + 1076);
      *(_DWORD *)(a1 + 1072) = (2 * (v51 >> 1) - 2) | v51 & 1;
    }
    else
    {
      v106 = 1;
      while ( v50 != -1 )
      {
        v107 = v106 + 1;
        v48 = v47 & (v106 + v48);
        v49 = (int *)(v46 + 8LL * v48);
        v50 = *v49;
        if ( v8 == *v49 )
          goto LABEL_31;
        v106 = v107;
      }
    }
LABEL_32:
    if ( (*(_BYTE *)(a1 + 1152) & 1) != 0 )
    {
      v52 = a1 + 1160;
      v53 = 7;
    }
    else
    {
      v84 = *(_DWORD *)(a1 + 1168);
      v52 = *(_QWORD *)(a1 + 1160);
      if ( !v84 )
        goto LABEL_36;
      v53 = v84 - 1;
    }
    v54 = v53 & (37 * v8);
    v55 = (int *)(v52 + 12LL * v54);
    v56 = *v55;
    if ( v8 == *v55 )
    {
LABEL_35:
      *v55 = -2;
      v57 = *(_DWORD *)(a1 + 1152);
      ++*(_DWORD *)(a1 + 1156);
      *(_DWORD *)(a1 + 1152) = (2 * (v57 >> 1) - 2) | v57 & 1;
    }
    else
    {
      v104 = 1;
      while ( v56 != -1 )
      {
        v105 = v104 + 1;
        v54 = v53 & (v104 + v54);
        v55 = (int *)(v52 + 12LL * v54);
        v56 = *v55;
        if ( v8 == *v55 )
          goto LABEL_35;
        v104 = v105;
      }
    }
LABEL_36:
    if ( (*(_BYTE *)(a1 + 1264) & 1) != 0 )
    {
      v58 = a1 + 1272;
      v59 = 7;
    }
    else
    {
      v83 = *(_DWORD *)(a1 + 1280);
      v58 = *(_QWORD *)(a1 + 1272);
      if ( !v83 )
        goto LABEL_40;
      v59 = v83 - 1;
    }
    v60 = v59 & (37 * v8);
    v61 = (int *)(v58 + 8LL * v60);
    v62 = *v61;
    if ( v8 == *v61 )
    {
LABEL_39:
      *v61 = -2;
      v63 = *(_DWORD *)(a1 + 1264);
      ++*(_DWORD *)(a1 + 1268);
      *(_DWORD *)(a1 + 1264) = (2 * (v63 >> 1) - 2) | v63 & 1;
    }
    else
    {
      v102 = 1;
      while ( v62 != -1 )
      {
        v103 = v102 + 1;
        v60 = v59 & (v102 + v60);
        v61 = (int *)(v58 + 8LL * v60);
        v62 = *v61;
        if ( v8 == *v61 )
          goto LABEL_39;
        v102 = v103;
      }
    }
LABEL_40:
    if ( (*(_BYTE *)(a1 + 1344) & 1) != 0 )
    {
      v64 = a1 + 1352;
      v65 = 7;
      goto LABEL_42;
    }
    v82 = *(_DWORD *)(a1 + 1360);
    v64 = *(_QWORD *)(a1 + 1352);
    if ( v82 )
    {
      v65 = v82 - 1;
LABEL_42:
      v66 = v65 & (37 * v8);
      v67 = (int *)(v64 + 12LL * v66);
      v68 = *v67;
      if ( v8 == *v67 )
      {
LABEL_43:
        *v67 = -2;
        v69 = *(_DWORD *)(a1 + 1344);
        ++*(_DWORD *)(a1 + 1348);
        *(_DWORD *)(a1 + 1344) = (2 * (v69 >> 1) - 2) | v69 & 1;
      }
      else
      {
        v100 = 1;
        while ( v68 != -1 )
        {
          v101 = v100 + 1;
          v66 = v65 & (v100 + v66);
          v67 = (int *)(v64 + 12LL * v66);
          v68 = *v67;
          if ( v8 == *v67 )
            goto LABEL_43;
          v100 = v101;
        }
      }
    }
    if ( (*(_BYTE *)(a1 + 1456) & 1) != 0 )
    {
      v70 = a1 + 1464;
      result = 7;
      goto LABEL_46;
    }
    result = *(unsigned int *)(a1 + 1472);
    v70 = *(_QWORD *)(a1 + 1464);
    if ( (_DWORD)result )
    {
      result = (unsigned int)(result - 1);
LABEL_46:
      v71 = result & (37 * v8);
      v72 = (int *)(v70 + 8LL * v71);
      v73 = *v72;
      if ( v8 == *v72 )
      {
LABEL_47:
        *v72 = -2;
        v74 = *(_DWORD *)(a1 + 1456);
        ++*(_DWORD *)(a1 + 1460);
        result = (2 * (v74 >> 1) - 2) | v74 & 1;
        *(_DWORD *)(a1 + 1456) = result;
      }
      else
      {
        v98 = 1;
        while ( v73 != -1 )
        {
          v99 = v98 + 1;
          v71 = result & (v98 + v71);
          v72 = (int *)(v70 + 8LL * v71);
          v73 = *v72;
          if ( v8 == *v72 )
            goto LABEL_47;
          v98 = v99;
        }
      }
    }
LABEL_48:
    if ( (*(_BYTE *)(a1 + 304) & 1) != 0 )
    {
      v75 = a1 + 312;
      v76 = 7;
    }
    else
    {
      v80 = *(_DWORD *)(a1 + 320);
      v75 = *(_QWORD *)(a1 + 312);
      if ( !v80 )
        goto LABEL_58;
      v76 = v80 - 1;
    }
    v77 = 1;
    for ( result = v76 & (v6 + v131); ; result = v76 & v79 )
    {
      v78 = v75 + 24LL * (unsigned int)result;
      if ( a2 == *(_QWORD *)v78 )
        break;
      if ( !*(_QWORD *)v78 && *(_DWORD *)(v78 + 8) == -1 )
        goto LABEL_58;
LABEL_53:
      v79 = v77 + result;
      ++v77;
    }
    if ( *(_DWORD *)(v78 + 8) != v6 )
      goto LABEL_53;
    *(_QWORD *)v78 = 0;
    *(_DWORD *)(v78 + 8) = -2;
    v81 = *(_DWORD *)(a1 + 304);
    ++*(_DWORD *)(a1 + 308);
    result = (2 * (v81 >> 1) - 2) | v81 & 1;
    *(_DWORD *)(a1 + 304) = result;
LABEL_58:
    ++v6;
  }
  while ( v132 != v6 );
  return result;
}
