// Function: sub_39F6BB0
// Address: 0x39f6bb0
//
void __fastcall sub_39F6BB0(char *a1, unsigned __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 *v4; // r15
  unsigned __int64 v8; // rcx
  char v9; // al
  char *v10; // rdx
  char v11; // si
  __int64 v12; // rax
  char v13; // si
  __int64 v14; // rax
  __int64 v15; // r9
  unsigned __int8 v16; // si
  int v17; // ecx
  char v18; // r8
  unsigned __int64 v19; // rdx
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rdi
  int v23; // ecx
  char v24; // si
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 *v29; // rax
  _QWORD *v30; // rax
  unsigned __int64 v31; // rdi
  int v32; // ecx
  char v33; // si
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  int v37; // ecx
  char v38; // si
  unsigned __int64 v39; // rax
  int v40; // ecx
  unsigned __int64 v41; // rdx
  char v42; // si
  unsigned __int64 v43; // rax
  __int64 v44; // r8
  int v45; // ecx
  char v46; // si
  unsigned __int64 v47; // rax
  __int64 v48; // rax
  int v49; // ecx
  unsigned __int64 v50; // rdx
  char v51; // si
  unsigned __int64 v52; // rax
  __int64 v53; // r8
  unsigned int v54; // ecx
  char v55; // si
  unsigned __int64 v56; // rax
  __int64 v57; // r8
  __int64 v58; // rax
  __int64 v59; // rdi
  int v60; // ecx
  char v61; // si
  unsigned __int64 v62; // rax
  unsigned int v63; // ecx
  __int64 v64; // rdi
  char v65; // si
  unsigned __int64 v66; // rax
  int v67; // ecx
  unsigned __int64 v68; // rdx
  char v69; // si
  unsigned __int64 v70; // rax
  __int64 v71; // r8
  int v72; // ecx
  char v73; // si
  unsigned __int64 v74; // rax
  __int64 v75; // r8
  __int64 v76; // rax
  int v77; // ecx
  unsigned __int64 v78; // rdx
  char v79; // si
  unsigned __int64 v80; // rax
  __int64 v81; // r8
  int v82; // ecx
  char v83; // si
  unsigned __int64 v84; // rax
  __int64 v85; // r8
  __int64 v86; // rax
  int v87; // ecx
  char v88; // si
  unsigned __int64 v89; // rax
  unsigned int v90; // ecx
  char v91; // si
  unsigned __int64 v92; // rax
  unsigned __int64 v93; // rdi
  int v94; // ecx
  char v95; // si
  unsigned __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // rdi
  int v99; // ecx
  char v100; // si
  unsigned __int64 v101; // rax
  __int64 v102; // rdi
  int v103; // ecx
  char v104; // si
  unsigned __int64 v105; // rax
  unsigned int v106; // ecx
  char v107; // si
  unsigned __int64 v108; // rax
  unsigned __int64 v109; // rsi
  int v110; // ecx
  char v111; // dl
  unsigned __int64 v112; // rax
  int v113; // ecx
  char v114; // dl
  unsigned __int64 v115; // rax
  unsigned __int64 v116; // rsi
  int v117; // ecx
  char v118; // dl
  unsigned __int64 v119; // rax
  int v120; // ecx
  char v121; // si
  unsigned __int64 v122; // rax
  int v123; // ecx
  char v124; // si
  unsigned __int64 v125; // rax
  __int64 v126; // rdi
  int v127; // ecx
  char v128; // si
  unsigned __int64 v129; // rax
  __int64 v130; // rdi
  int v131; // ecx
  char v132; // si
  unsigned __int64 v133; // rax
  int v134; // ecx
  __int64 v135; // rdi
  char v136; // si
  unsigned __int64 v137; // rax
  __int64 v138; // rdi
  int v139; // ecx
  char v140; // si
  unsigned __int64 v141; // rax
  __int64 v142; // [rsp-Eh] [rbp-198h] BYREF
  unsigned __int64 v143[8]; // [rsp+14Ah] [rbp-40h] BYREF

  *(_QWORD *)(a4 + 288) = 0;
  if ( (unsigned __int64)a1 < a2 )
  {
    v4 = 0;
    do
    {
      v8 = *(_QWORD *)(a4 + 328);
      if ( v8 >= a3[19] + (a3[24] >> 63) )
        break;
      v9 = *a1;
      v10 = a1 + 1;
      v11 = *a1 & 0xC0;
      switch ( v11 )
      {
        case 64:
          ++a1;
          *(_QWORD *)(a4 + 328) = v8 + *(_QWORD *)(a4 + 352) * (v9 & 0x3F);
          break;
        case -128:
          v13 = *a1++;
          v14 = v9 & 0x3F;
          v15 = 0;
          v16 = v13 & 0x3F;
          v17 = 0;
          do
          {
            v18 = *a1++;
            v19 = (unsigned __int64)(v18 & 0x7F) << v17;
            v17 += 7;
            v15 |= v19;
          }
          while ( v18 < 0 );
          v20 = *(_QWORD *)(a4 + 344) * v15;
          if ( v16 <= 0x11u )
          {
            v21 = a4 + 16 * v14;
            *(_DWORD *)(v21 + 8) = 1;
            *(_QWORD *)v21 = v20;
          }
          break;
        case -64:
          v12 = *a1 & 0x3F;
          if ( (*a1 & 0x3Fu) > 0x11 )
          {
LABEL_17:
            ++a1;
          }
          else
          {
            ++a1;
            *(_DWORD *)(a4 + 16 * v12 + 8) = 0;
          }
          break;
        default:
          switch ( v9 )
          {
            case 0:
            case 45:
              goto LABEL_17;
            case 1:
              a1 = sub_39F5E90(a3, *(_BYTE *)(a4 + 368), v10, v143);
              *(_QWORD *)(a4 + 328) = v143[0];
              continue;
            case 2:
              v26 = *(_QWORD *)(a4 + 352) * (unsigned __int8)a1[1];
              a1 += 2;
              *(_QWORD *)(a4 + 328) = v8 + v26;
              continue;
            case 3:
              v27 = *(_QWORD *)(a4 + 352) * *(unsigned __int16 *)(a1 + 1);
              a1 += 3;
              *(_QWORD *)(a4 + 328) = v8 + v27;
              continue;
            case 4:
              v28 = *(_QWORD *)(a4 + 352) * *(unsigned int *)(a1 + 1);
              a1 += 5;
              *(_QWORD *)(a4 + 328) = v8 + v28;
              continue;
            case 5:
              ++a1;
              v120 = 0;
              v50 = 0;
              do
              {
                v121 = *a1++;
                v122 = (unsigned __int64)(v121 & 0x7F) << v120;
                v120 += 7;
                v50 |= v122;
              }
              while ( v121 < 0 );
              v53 = 0;
              v123 = 0;
              do
              {
                v124 = *a1++;
                v125 = (unsigned __int64)(v124 & 0x7F) << v123;
                v123 += 7;
                v53 |= v125;
              }
              while ( v124 < 0 );
              goto LABEL_49;
            case 6:
              ++a1;
              v109 = 0;
              v113 = 0;
              do
              {
                v114 = *a1++;
                v115 = (unsigned __int64)(v114 & 0x7F) << v113;
                v113 += 7;
                v109 |= v115;
              }
              while ( v114 < 0 );
              goto LABEL_96;
            case 7:
              ++a1;
              v116 = 0;
              v117 = 0;
              do
              {
                v118 = *a1++;
                v119 = (unsigned __int64)(v118 & 0x7F) << v117;
                v117 += 7;
                v116 |= v119;
              }
              while ( v118 < 0 );
              if ( v116 <= 0x11 )
                *(_DWORD *)(a4 + 16 * v116 + 8) = 6;
              continue;
            case 8:
              ++a1;
              v109 = 0;
              v110 = 0;
              do
              {
                v111 = *a1++;
                v112 = (unsigned __int64)(v111 & 0x7F) << v110;
                v110 += 7;
                v109 |= v112;
              }
              while ( v111 < 0 );
LABEL_96:
              if ( v109 <= 0x11 )
                *(_DWORD *)(a4 + 16 * v109 + 8) = 0;
              continue;
            case 9:
              ++a1;
              v40 = 0;
              v41 = 0;
              do
              {
                v42 = *a1++;
                v43 = (unsigned __int64)(v42 & 0x7F) << v40;
                v40 += 7;
                v41 |= v43;
              }
              while ( v42 < 0 );
              v44 = 0;
              v45 = 0;
              do
              {
                v46 = *a1++;
                v47 = (unsigned __int64)(v46 & 0x7F) << v45;
                v45 += 7;
                v44 |= v47;
              }
              while ( v46 < 0 );
              if ( v41 <= 0x11 )
              {
                v48 = a4 + 16 * v41;
                *(_DWORD *)(v48 + 8) = 2;
                *(_QWORD *)v48 = v44;
              }
              continue;
            case 10:
              if ( v4 )
              {
                v29 = v4;
                v4 = (__int64 *)v4[36];
              }
              else
              {
                v29 = &v142;
              }
              qmemcpy(v29, (const void *)a4, 0x148u);
              *(_QWORD *)(a4 + 288) = v29;
              ++a1;
              continue;
            case 11:
              v30 = *(_QWORD **)(a4 + 288);
              qmemcpy((void *)a4, v30, 0x148u);
              v30[36] = v4;
              ++a1;
              v4 = v30;
              continue;
            case 12:
              v130 = 0;
              v131 = 0;
              do
              {
                v132 = *v10++;
                v133 = (unsigned __int64)(v132 & 0x7F) << v131;
                v131 += 7;
                v130 |= v133;
              }
              while ( v132 < 0 );
              *(_QWORD *)(a4 + 304) = v130;
              v134 = 0;
              v135 = 0;
              do
              {
                v136 = *v10++;
                v137 = (unsigned __int64)(v136 & 0x7F) << v134;
                v134 += 7;
                v135 |= v137;
              }
              while ( v136 < 0 );
              *(_QWORD *)(a4 + 296) = v135;
              a1 = v10;
              *(_DWORD *)(a4 + 320) = 1;
              continue;
            case 13:
              v138 = 0;
              v139 = 0;
              do
              {
                v140 = *v10++;
                v141 = (unsigned __int64)(v140 & 0x7F) << v139;
                v139 += 7;
                v138 |= v141;
              }
              while ( v140 < 0 );
              *(_QWORD *)(a4 + 304) = v138;
              a1 = v10;
              *(_DWORD *)(a4 + 320) = 1;
              continue;
            case 14:
              v126 = 0;
              v127 = 0;
              do
              {
                v128 = *v10++;
                v129 = (unsigned __int64)(v128 & 0x7F) << v127;
                v127 += 7;
                v126 |= v129;
              }
              while ( v128 < 0 );
              *(_QWORD *)(a4 + 296) = v126;
              a1 = v10;
              continue;
            case 15:
              *(_QWORD *)(a4 + 312) = v10;
              v22 = 0;
              v23 = 0;
              *(_DWORD *)(a4 + 320) = 2;
              do
              {
                v24 = *v10++;
                v25 = (unsigned __int64)(v24 & 0x7F) << v23;
                v23 += 7;
                v22 |= v25;
              }
              while ( v24 < 0 );
              a1 = &v10[v22];
              continue;
            case 16:
              v31 = 0;
              v32 = 0;
              do
              {
                v33 = *v10++;
                v34 = (unsigned __int64)(v33 & 0x7F) << v32;
                v32 += 7;
                v31 |= v34;
              }
              while ( v33 < 0 );
              if ( v31 <= 0x11 )
              {
                v35 = a4 + 16 * v31;
                *(_DWORD *)(v35 + 8) = 3;
                *(_QWORD *)v35 = v10;
              }
              v36 = 0;
              v37 = 0;
              do
              {
                v38 = *v10++;
                v39 = (unsigned __int64)(v38 & 0x7F) << v37;
                v37 += 7;
                v36 |= v39;
              }
              while ( v38 < 0 );
              a1 = &v10[v36];
              continue;
            case 17:
              ++a1;
              v49 = 0;
              v50 = 0;
              do
              {
                v51 = *a1++;
                v52 = (unsigned __int64)(v51 & 0x7F) << v49;
                v49 += 7;
                v50 |= v52;
              }
              while ( v51 < 0 );
              v53 = 0;
              v54 = 0;
              do
              {
                v55 = *a1++;
                v56 = (unsigned __int64)(v55 & 0x7F) << v54;
                v54 += 7;
                v53 |= v56;
              }
              while ( v55 < 0 );
              if ( v54 <= 0x3F && (v55 & 0x40) != 0 )
                v53 |= -1LL << v54;
LABEL_49:
              v57 = *(_QWORD *)(a4 + 344) * v53;
              if ( v50 <= 0x11 )
              {
                v58 = a4 + 16 * v50;
                *(_DWORD *)(v58 + 8) = 1;
                *(_QWORD *)v58 = v57;
              }
              continue;
            case 18:
              v59 = 0;
              v60 = 0;
              do
              {
                v61 = *v10++;
                v62 = (unsigned __int64)(v61 & 0x7F) << v60;
                v60 += 7;
                v59 |= v62;
              }
              while ( v61 < 0 );
              *(_QWORD *)(a4 + 304) = v59;
              v63 = 0;
              v64 = 0;
              do
              {
                v65 = *v10++;
                v66 = (unsigned __int64)(v65 & 0x7F) << v63;
                v63 += 7;
                v64 |= v66;
              }
              while ( v65 < 0 );
              if ( v63 <= 0x3F && (v65 & 0x40) != 0 )
                v64 |= -1LL << v63;
              *(_DWORD *)(a4 + 320) = 1;
              goto LABEL_59;
            case 19:
              v64 = 0;
              v106 = 0;
              do
              {
                v107 = *v10++;
                v108 = (unsigned __int64)(v107 & 0x7F) << v106;
                v106 += 7;
                v64 |= v108;
              }
              while ( v107 < 0 );
              if ( v106 <= 0x3F && (v107 & 0x40) != 0 )
                v64 |= -1LL << v106;
LABEL_59:
              *(_QWORD *)(a4 + 296) = *(_QWORD *)(a4 + 344) * v64;
              a1 = v10;
              continue;
            case 20:
              ++a1;
              v77 = 0;
              v78 = 0;
              do
              {
                v79 = *a1++;
                v80 = (unsigned __int64)(v79 & 0x7F) << v77;
                v77 += 7;
                v78 |= v80;
              }
              while ( v79 < 0 );
              v81 = 0;
              v82 = 0;
              do
              {
                v83 = *a1++;
                v84 = (unsigned __int64)(v83 & 0x7F) << v82;
                v82 += 7;
                v81 |= v84;
              }
              while ( v83 < 0 );
              goto LABEL_70;
            case 21:
              ++a1;
              v87 = 0;
              v78 = 0;
              do
              {
                v88 = *a1++;
                v89 = (unsigned __int64)(v88 & 0x7F) << v87;
                v87 += 7;
                v78 |= v89;
              }
              while ( v88 < 0 );
              v81 = 0;
              v90 = 0;
              do
              {
                v91 = *a1++;
                v92 = (unsigned __int64)(v91 & 0x7F) << v90;
                v90 += 7;
                v81 |= v92;
              }
              while ( v91 < 0 );
              if ( v90 <= 0x3F && (v91 & 0x40) != 0 )
                v81 |= -1LL << v90;
LABEL_70:
              v85 = *(_QWORD *)(a4 + 344) * v81;
              if ( v78 <= 0x11 )
              {
                v86 = a4 + 16 * v78;
                *(_DWORD *)(v86 + 8) = 4;
                *(_QWORD *)v86 = v85;
              }
              break;
            case 22:
              v93 = 0;
              v94 = 0;
              do
              {
                v95 = *v10++;
                v96 = (unsigned __int64)(v95 & 0x7F) << v94;
                v94 += 7;
                v93 |= v96;
              }
              while ( v95 < 0 );
              if ( v93 <= 0x11 )
              {
                v97 = a4 + 16 * v93;
                *(_DWORD *)(v97 + 8) = 5;
                *(_QWORD *)v97 = v10;
              }
              v98 = 0;
              v99 = 0;
              do
              {
                v100 = *v10++;
                v101 = (unsigned __int64)(v100 & 0x7F) << v99;
                v99 += 7;
                v98 |= v101;
              }
              while ( v100 < 0 );
              a1 = &v10[v98];
              break;
            case 46:
              v102 = 0;
              v103 = 0;
              do
              {
                v104 = *v10++;
                v105 = (unsigned __int64)(v104 & 0x7F) << v103;
                v103 += 7;
                v102 |= v105;
              }
              while ( v104 < 0 );
              a3[26] = v102;
              a1 = v10;
              break;
            case 47:
              ++a1;
              v67 = 0;
              v68 = 0;
              do
              {
                v69 = *a1++;
                v70 = (unsigned __int64)(v69 & 0x7F) << v67;
                v67 += 7;
                v68 |= v70;
              }
              while ( v69 < 0 );
              v71 = 0;
              v72 = 0;
              do
              {
                v73 = *a1++;
                v74 = (unsigned __int64)(v73 & 0x7F) << v72;
                v72 += 7;
                v71 |= v74;
              }
              while ( v73 < 0 );
              v75 = *(_QWORD *)(a4 + 344) * v71;
              if ( v68 <= 0x11 )
              {
                v76 = a4 + 16 * v68;
                *(_DWORD *)(v76 + 8) = 1;
                *(_QWORD *)v76 = -v75;
              }
              break;
            default:
              abort();
          }
          break;
      }
    }
    while ( (unsigned __int64)a1 < a2 );
  }
}
