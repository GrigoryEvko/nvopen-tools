// Function: sub_1B6E4B0
// Address: 0x1b6e4b0
//
__int64 __fastcall sub_1B6E4B0(__int64 a1, __int64 a2, _BYTE *a3, double a4, double a5, double a6)
{
  __int64 v6; // r15
  __int64 v9; // r12
  __int64 **v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r15
  char v17; // al
  __int64 v18; // r15
  __int64 v19; // rsi
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 **v22; // rbx
  char v23; // al
  __int64 v24; // rax
  __int64 v26; // rax
  unsigned int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // r10
  unsigned __int64 v30; // r9
  unsigned int v31; // esi
  int v32; // eax
  __int64 v33; // r15
  unsigned int v34; // eax
  __int64 v35; // r9
  __int64 v36; // rsi
  unsigned __int64 v37; // r10
  __int64 v38; // rax
  _QWORD *v39; // rax
  char v40; // al
  char v41; // al
  __int64 **v42; // rdx
  char v43; // al
  __int64 v44; // rcx
  __int64 v45; // rcx
  __int64 v46; // rsi
  __int64 v47; // r9
  unsigned __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  unsigned int v52; // esi
  int v53; // eax
  __int64 v54; // rax
  __int64 v55; // rax
  _QWORD *v56; // rax
  __int64 v57; // rax
  int v58; // eax
  __int64 v59; // rax
  _QWORD *v60; // rax
  int v61; // eax
  __int64 v62; // rax
  unsigned int v63; // esi
  int v64; // eax
  __int64 v65; // rax
  _QWORD *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  unsigned int v69; // esi
  int v70; // eax
  __int64 v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // [rsp-80h] [rbp-80h]
  __int64 v74; // [rsp-78h] [rbp-78h]
  unsigned __int64 v75; // [rsp-78h] [rbp-78h]
  __int64 v76; // [rsp-70h] [rbp-70h]
  unsigned __int64 v77; // [rsp-70h] [rbp-70h]
  __int64 v78; // [rsp-70h] [rbp-70h]
  unsigned __int64 v79; // [rsp-68h] [rbp-68h]
  __int64 v80; // [rsp-68h] [rbp-68h]
  __int64 v81; // [rsp-68h] [rbp-68h]
  __int64 v82; // [rsp-68h] [rbp-68h]
  __int64 v83; // [rsp-68h] [rbp-68h]
  __int64 v84; // [rsp-68h] [rbp-68h]
  __int64 v85; // [rsp-60h] [rbp-60h]
  __int64 v86; // [rsp-60h] [rbp-60h]
  __int64 v87; // [rsp-60h] [rbp-60h]
  __int64 v88; // [rsp-60h] [rbp-60h]
  unsigned __int64 v89; // [rsp-60h] [rbp-60h]
  __int64 v90; // [rsp-60h] [rbp-60h]
  unsigned __int64 v91; // [rsp-60h] [rbp-60h]
  __int64 v92; // [rsp-60h] [rbp-60h]
  unsigned __int64 v93; // [rsp-60h] [rbp-60h]
  __int64 v94; // [rsp-58h] [rbp-58h]
  __int64 v95; // [rsp-58h] [rbp-58h]
  __int64 v96; // [rsp-58h] [rbp-58h]
  __int64 v97; // [rsp-58h] [rbp-58h]
  __int64 v98; // [rsp-58h] [rbp-58h]
  unsigned __int64 v99; // [rsp-58h] [rbp-58h]
  __int64 v100; // [rsp-58h] [rbp-58h]
  unsigned __int64 v101; // [rsp-58h] [rbp-58h]
  unsigned __int64 v102; // [rsp-58h] [rbp-58h]
  __int64 v103; // [rsp-58h] [rbp-58h]
  unsigned __int64 v104; // [rsp-58h] [rbp-58h]
  __int64 v105; // [rsp-58h] [rbp-58h]
  unsigned __int64 v106; // [rsp-58h] [rbp-58h]
  __int64 v107; // [rsp-58h] [rbp-58h]
  int v108; // [rsp-50h] [rbp-50h]
  __int64 v109; // [rsp-50h] [rbp-50h]
  __int64 v110; // [rsp-50h] [rbp-50h]
  __int64 v111; // [rsp-50h] [rbp-50h]
  __int64 v112; // [rsp-50h] [rbp-50h]
  unsigned __int64 v113; // [rsp-50h] [rbp-50h]
  __int64 v114; // [rsp-50h] [rbp-50h]
  unsigned __int64 v115; // [rsp-50h] [rbp-50h]
  __int64 v116; // [rsp-50h] [rbp-50h]
  __int64 v117; // [rsp-50h] [rbp-50h]
  __int64 v118; // [rsp-50h] [rbp-50h]
  __int64 v119; // [rsp-50h] [rbp-50h]
  __int64 v120; // [rsp-50h] [rbp-50h]
  __int64 v121; // [rsp-50h] [rbp-50h]
  __int64 v122; // [rsp-50h] [rbp-50h]
  unsigned int v123; // [rsp-3Ch] [rbp-3Ch] BYREF

  if ( !a1 )
LABEL_6:
    BUG();
  v6 = 1;
  v9 = sub_14DBA30(a1, (__int64)a3, 0);
  if ( !v9 )
    v9 = a1;
  v10 = *(__int64 ***)v9;
  v11 = *(_QWORD *)v9;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v11 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v26 = *(_QWORD *)(v11 + 32);
        v11 = *(_QWORD *)(v11 + 24);
        v6 *= v26;
        continue;
      case 1:
        v12 = 16;
        break;
      case 2:
        v12 = 32;
        break;
      case 3:
      case 9:
        v12 = 64;
        break;
      case 4:
        v12 = 80;
        break;
      case 5:
      case 6:
        v12 = 128;
        break;
      case 7:
        v12 = 8 * (unsigned int)sub_15A9520((__int64)a3, 0);
        break;
      case 0xB:
        v12 = *(_DWORD *)(v11 + 8) >> 8;
        break;
      case 0xD:
        v12 = 8LL * *(_QWORD *)sub_15A9930((__int64)a3, v11);
        break;
      case 0xE:
        v94 = *(_QWORD *)(v11 + 24);
        v109 = *(_QWORD *)(v11 + 32);
        v27 = sub_15A9FE0((__int64)a3, v94);
        v28 = v94;
        v29 = 1;
        v30 = v27;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v28 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v67 = *(_QWORD *)(v28 + 32);
              v28 = *(_QWORD *)(v28 + 24);
              v29 *= v67;
              continue;
            case 1:
              v51 = 16;
              goto LABEL_88;
            case 2:
              v51 = 32;
              goto LABEL_88;
            case 3:
            case 9:
              v51 = 64;
              goto LABEL_88;
            case 4:
              v51 = 80;
              goto LABEL_88;
            case 5:
            case 6:
              v51 = 128;
              goto LABEL_88;
            case 7:
              v88 = v29;
              v63 = 0;
              v102 = v30;
              goto LABEL_112;
            case 0xB:
              v51 = *(_DWORD *)(v28 + 8) >> 8;
              goto LABEL_88;
            case 0xD:
              v90 = v29;
              v104 = v30;
              v66 = (_QWORD *)sub_15A9930((__int64)a3, v28);
              v30 = v104;
              v29 = v90;
              v51 = 8LL * *v66;
              goto LABEL_88;
            case 0xE:
              v74 = v29;
              v77 = v30;
              v81 = *(_QWORD *)(v28 + 24);
              v103 = *(_QWORD *)(v28 + 32);
              v89 = (unsigned int)sub_15A9FE0((__int64)a3, v81);
              v65 = sub_127FA20((__int64)a3, v81);
              v30 = v77;
              v29 = v74;
              v51 = 8 * v103 * v89 * ((v89 + ((unsigned __int64)(v65 + 7) >> 3) - 1) / v89);
              goto LABEL_88;
            case 0xF:
              v88 = v29;
              v102 = v30;
              v63 = *(_DWORD *)(v28 + 8) >> 8;
LABEL_112:
              v64 = sub_15A9520((__int64)a3, v63);
              v30 = v102;
              v29 = v88;
              v51 = (unsigned int)(8 * v64);
LABEL_88:
              v12 = 8 * v30 * v109 * ((v30 + ((unsigned __int64)(v51 * v29 + 7) >> 3) - 1) / v30);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v12 = 8 * (unsigned int)sub_15A9520((__int64)a3, *(_DWORD *)(v11 + 8) >> 8);
        break;
    }
    break;
  }
  v13 = a2;
  v14 = 1;
  v15 = v12 * v6;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v13 + 8) )
    {
      case 1:
        v16 = 16;
        goto LABEL_12;
      case 2:
        v16 = 32;
        goto LABEL_12;
      case 3:
      case 9:
        v16 = 64;
        goto LABEL_12;
      case 4:
        v16 = 80;
        goto LABEL_12;
      case 5:
      case 6:
        v17 = *((_BYTE *)v10 + 8);
        v18 = v14 << 7;
        if ( v17 != 13 )
          goto LABEL_19;
        goto LABEL_13;
      case 7:
        v95 = v15;
        v31 = 0;
        v110 = v14;
        goto LABEL_45;
      case 0xB:
        v16 = *(_DWORD *)(v13 + 8) >> 8;
        goto LABEL_12;
      case 0xD:
        v97 = v15;
        v112 = v14;
        v39 = (_QWORD *)sub_15A9930((__int64)a3, v13);
        v14 = v112;
        v15 = v97;
        v16 = 8LL * *v39;
        goto LABEL_12;
      case 0xE:
        v33 = *(_QWORD *)(v13 + 32);
        v85 = v15;
        v96 = v14;
        v111 = *(_QWORD *)(v13 + 24);
        v34 = sub_15A9FE0((__int64)a3, v111);
        v15 = v85;
        v14 = v96;
        v35 = 1;
        v36 = v111;
        v37 = v34;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v36 + 8) )
          {
            case 1:
              v62 = 16;
              goto LABEL_109;
            case 2:
              v62 = 32;
              goto LABEL_109;
            case 3:
            case 9:
              v62 = 64;
              goto LABEL_109;
            case 4:
              v62 = 80;
              goto LABEL_109;
            case 5:
            case 6:
              v62 = 128;
              goto LABEL_109;
            case 7:
              v82 = v35;
              v69 = 0;
              v91 = v37;
              v105 = v15;
              v120 = v14;
              goto LABEL_125;
            case 0xB:
              v62 = *(_DWORD *)(v36 + 8) >> 8;
              goto LABEL_109;
            case 0xD:
              v84 = v35;
              v93 = v37;
              v107 = v15;
              v122 = v14;
              v72 = (_QWORD *)sub_15A9930((__int64)a3, v36);
              v14 = v122;
              v15 = v107;
              v37 = v93;
              v35 = v84;
              v62 = 8LL * *v72;
              goto LABEL_109;
            case 0xE:
              v73 = v35;
              v75 = v37;
              v78 = v85;
              v83 = v96;
              v92 = *(_QWORD *)(v36 + 24);
              v121 = *(_QWORD *)(v36 + 32);
              v106 = (unsigned int)sub_15A9FE0((__int64)a3, v92);
              v71 = sub_127FA20((__int64)a3, v92);
              v14 = v83;
              v15 = v78;
              v37 = v75;
              v35 = v73;
              v62 = 8 * v121 * v106 * ((v106 + ((unsigned __int64)(v71 + 7) >> 3) - 1) / v106);
              goto LABEL_109;
            case 0xF:
              v82 = v35;
              v91 = v37;
              v105 = v15;
              v69 = *(_DWORD *)(v36 + 8) >> 8;
              v120 = v14;
LABEL_125:
              v70 = sub_15A9520((__int64)a3, v69);
              v14 = v120;
              v15 = v105;
              v37 = v91;
              v35 = v82;
              v62 = (unsigned int)(8 * v70);
LABEL_109:
              v16 = 8 * v37 * v33 * ((v37 + ((unsigned __int64)(v62 * v35 + 7) >> 3) - 1) / v37);
              goto LABEL_12;
            case 0x10:
              v68 = *(_QWORD *)(v36 + 32);
              v36 = *(_QWORD *)(v36 + 24);
              v35 *= v68;
              continue;
            default:
              goto LABEL_6;
          }
        }
      case 0xF:
        v95 = v15;
        v110 = v14;
        v31 = *(_DWORD *)(v13 + 8) >> 8;
LABEL_45:
        v32 = sub_15A9520((__int64)a3, v31);
        v14 = v110;
        v15 = v95;
        v16 = (unsigned int)(8 * v32);
LABEL_12:
        v17 = *((_BYTE *)v10 + 8);
        v18 = v14 * v16;
        if ( v17 == 13 )
        {
LABEL_13:
          if ( a2 == *v10[2] )
          {
            v123 = 0;
            return sub_15A3AE0((_QWORD *)v9, &v123, 1, 0);
          }
          if ( v18 != v15 )
            goto LABEL_15;
          goto LABEL_62;
        }
LABEL_19:
        if ( v18 == v15 )
        {
          if ( v17 == 16 )
            v17 = *(_BYTE *)(*v10[2] + 8);
          if ( v17 == 15 )
          {
            v40 = *(_BYTE *)(a2 + 8);
            if ( v40 == 16 )
              v40 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
            if ( v40 == 15 )
            {
              v9 = sub_15A46C0(47, (__int64 ***)v9, (__int64 **)a2, 0);
              goto LABEL_72;
            }
            v10 = (__int64 **)sub_15A9650((__int64)a3, (__int64)v10);
            v9 = sub_15A46C0(45, (__int64 ***)v9, v10, 0);
          }
LABEL_62:
          v41 = *(_BYTE *)(a2 + 8);
          if ( v41 == 16 )
            v41 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
          v42 = (__int64 **)a2;
          if ( v41 == 15 )
            v42 = (__int64 **)sub_15A9650((__int64)a3, a2);
          if ( v10 != v42 )
            v9 = sub_15A46C0(47, (__int64 ***)v9, v42, 0);
          v43 = *(_BYTE *)(a2 + 8);
          if ( v43 == 16 )
            v43 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
          if ( v43 == 15 )
            v9 = sub_15A46C0(46, (__int64 ***)v9, (__int64 **)a2, 0);
LABEL_72:
          if ( *(_BYTE *)(v9 + 16) != 5 )
            return v9;
          goto LABEL_31;
        }
        if ( v17 == 16 )
        {
          if ( *(_BYTE *)(*v10[2] + 8) != 15 )
            goto LABEL_15;
          goto LABEL_22;
        }
        if ( v17 == 15 )
        {
LABEL_22:
          v108 = v15;
          v10 = (__int64 **)sub_15A9650((__int64)a3, (__int64)v10);
          v21 = sub_15A46C0(45, (__int64 ***)v9, v10, 0);
          LODWORD(v15) = v108;
          v9 = v21;
          v17 = *((_BYTE *)v10 + 8);
        }
        if ( v17 != 11 )
        {
LABEL_15:
          v10 = (__int64 **)sub_1644900(*v10, v15);
          v9 = sub_15A46C0(47, (__int64 ***)v9, v10, 0);
          if ( *a3 )
          {
            v19 = (__int64)v10;
            v20 = 1;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v19 + 8) )
              {
                case 1:
                  v44 = 16;
                  goto LABEL_79;
                case 2:
                  v44 = 32;
                  goto LABEL_79;
                case 3:
                case 9:
                  v44 = 64;
                  goto LABEL_79;
                case 4:
                  v44 = 80;
                  goto LABEL_79;
                case 5:
                case 6:
                  v44 = 128;
                  goto LABEL_79;
                case 7:
                  v119 = v20;
                  v61 = sub_15A9520((__int64)a3, 0);
                  v20 = v119;
                  v44 = (unsigned int)(8 * v61);
                  goto LABEL_79;
                case 0xB:
                  goto LABEL_86;
                case 0xD:
                  v118 = v20;
                  v60 = (_QWORD *)sub_15A9930((__int64)a3, v19);
                  v20 = v118;
                  v44 = 8LL * *v60;
                  goto LABEL_79;
                case 0xE:
                  v80 = v20;
                  v87 = *(_QWORD *)(v19 + 24);
                  v117 = *(_QWORD *)(v19 + 32);
                  v101 = (unsigned int)sub_15A9FE0((__int64)a3, v87);
                  v59 = sub_127FA20((__int64)a3, v87);
                  v20 = v80;
                  v44 = 8 * v117 * v101 * ((v101 + ((unsigned __int64)(v59 + 7) >> 3) - 1) / v101);
                  goto LABEL_79;
                case 0xF:
                  v116 = v20;
                  v58 = sub_15A9520((__int64)a3, *(_DWORD *)(v19 + 8) >> 8);
                  v20 = v116;
                  v44 = (unsigned int)(8 * v58);
                  goto LABEL_79;
                case 0x10:
                  v57 = *(_QWORD *)(v19 + 32);
                  v19 = *(_QWORD *)(v19 + 24);
                  v20 *= v57;
                  continue;
                default:
                  goto LABEL_6;
              }
            }
          }
          goto LABEL_25;
        }
        if ( *a3 )
        {
          v19 = (__int64)v10;
          v20 = 1;
LABEL_86:
          v44 = *(_DWORD *)(v19 + 8) >> 8;
LABEL_79:
          v45 = v20 * v44;
          v46 = a2;
          v47 = 1;
          v48 = (unsigned __int64)(v45 + 7) >> 3;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v46 + 8) )
            {
              case 1:
                v49 = 16;
                goto LABEL_82;
              case 2:
                v49 = 32;
                goto LABEL_82;
              case 3:
              case 9:
                v49 = 64;
                goto LABEL_82;
              case 4:
                v49 = 80;
                goto LABEL_82;
              case 5:
              case 6:
                v49 = 128;
                goto LABEL_82;
              case 7:
                v98 = v47;
                v52 = 0;
                v113 = v48;
                goto LABEL_94;
              case 0xB:
                v49 = *(_DWORD *)(v46 + 8) >> 8;
                goto LABEL_82;
              case 0xD:
                v100 = v47;
                v115 = v48;
                v56 = (_QWORD *)sub_15A9930((__int64)a3, v46);
                v48 = v115;
                v47 = v100;
                v49 = 8LL * *v56;
                goto LABEL_82;
              case 0xE:
                v76 = v47;
                v79 = v48;
                v86 = *(_QWORD *)(v46 + 24);
                v114 = *(_QWORD *)(v46 + 32);
                v99 = (unsigned int)sub_15A9FE0((__int64)a3, v86);
                v55 = sub_127FA20((__int64)a3, v86);
                v48 = v79;
                v47 = v76;
                v49 = 8 * v114 * v99 * ((v99 + ((unsigned __int64)(v55 + 7) >> 3) - 1) / v99);
                goto LABEL_82;
              case 0xF:
                v98 = v47;
                v113 = v48;
                v52 = *(_DWORD *)(v46 + 8) >> 8;
LABEL_94:
                v53 = sub_15A9520((__int64)a3, v52);
                v48 = v113;
                v47 = v98;
                v49 = (unsigned int)(8 * v53);
LABEL_82:
                v50 = sub_15A0680(*(_QWORD *)v9, 8 * (v48 - ((unsigned __int64)(v49 * v47 + 7) >> 3)), 0);
                v9 = sub_15A2D80((__int64 *)v9, v50, 0, a4, a5, a6);
                break;
              case 0x10:
                v54 = *(_QWORD *)(v46 + 32);
                v46 = *(_QWORD *)(v46 + 24);
                v47 *= v54;
                continue;
              default:
                goto LABEL_6;
            }
            break;
          }
        }
LABEL_25:
        v22 = (__int64 **)sub_1644900(*v10, v18);
        v9 = sub_15A4670((__int64 ***)v9, v22);
        if ( (__int64 **)a2 != v22 )
        {
          v23 = *(_BYTE *)(a2 + 8);
          if ( v23 == 16 )
            v23 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
          if ( v23 == 15 )
            v9 = sub_15A46C0(46, (__int64 ***)v9, (__int64 **)a2, 0);
          else
            v9 = sub_15A46C0(47, (__int64 ***)v9, (__int64 **)a2, 0);
        }
        if ( v9 )
        {
LABEL_31:
          v24 = sub_14DBA30(v9, (__int64)a3, 0);
          if ( v24 )
            return v24;
        }
        return v9;
      case 0x10:
        v38 = *(_QWORD *)(v13 + 32);
        v13 = *(_QWORD *)(v13 + 24);
        v14 *= v38;
        continue;
      default:
        goto LABEL_6;
    }
  }
}
