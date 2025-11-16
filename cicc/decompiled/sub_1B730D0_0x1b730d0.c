// Function: sub_1B730D0
// Address: 0x1b730d0
//
__int64 __fastcall sub_1B730D0(
        __int64 *a1,
        int a2,
        unsigned __int64 a3,
        __int64 a4,
        _BYTE *a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v13; // r15
  _QWORD **v15; // r13
  __int64 v16; // r12
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // r15
  __int64 v22; // rax
  unsigned int v23; // ebx
  __int64 v24; // rbx
  __int64 ***v25; // r15
  unsigned __int8 *v26; // rsi
  __int64 *v27; // rdi
  __int64 **v28; // rax
  __int64 v29; // rax
  unsigned __int8 *v30; // rsi
  __int64 **v31; // rbx
  unsigned __int8 *v32; // rsi
  __int64 *v33; // rbx
  unsigned __int64 *v34; // r15
  __int64 v35; // rax
  unsigned __int64 v36; // rcx
  __int64 v37; // rsi
  __int64 v38; // rdx
  unsigned __int8 *v39; // rsi
  __int64 v40; // r15
  double v41; // xmm4_8
  double v42; // xmm5_8
  __int64 **v43; // rdx
  __int64 v45; // rbx
  unsigned int v46; // eax
  __int64 v47; // rsi
  __int64 v48; // r8
  unsigned __int64 v49; // r9
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rbx
  unsigned int v53; // eax
  __int64 v54; // rsi
  __int64 v55; // r8
  unsigned __int64 v56; // r9
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rsi
  __int64 v60; // rax
  __int64 v61; // rsi
  __int64 v62; // rdx
  unsigned __int8 *v63; // rsi
  __int64 v64; // rax
  unsigned __int64 *v65; // rbx
  __int64 **v66; // rax
  unsigned __int64 v67; // rcx
  __int64 v68; // rsi
  unsigned __int8 *v69; // rsi
  __int64 v70; // rax
  unsigned __int64 v71; // r8
  __int64 v72; // rax
  __int64 v73; // rax
  unsigned int v74; // esi
  int v75; // eax
  __int64 v76; // rax
  _QWORD *v77; // rax
  unsigned int v78; // esi
  int v79; // eax
  __int64 v80; // rax
  __int64 v81; // rax
  _QWORD *v82; // rax
  unsigned __int64 v83; // [rsp+0h] [rbp-110h]
  unsigned __int64 v84; // [rsp+8h] [rbp-108h]
  __int64 v85; // [rsp+8h] [rbp-108h]
  __int64 v86; // [rsp+10h] [rbp-100h]
  __int64 v87; // [rsp+10h] [rbp-100h]
  __int64 v89; // [rsp+20h] [rbp-F0h]
  __int64 v90; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v91; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v92; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v93; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v94; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v95; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v96; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v97; // [rsp+28h] [rbp-E8h]
  unsigned int v98; // [rsp+30h] [rbp-E0h]
  __int64 v99; // [rsp+30h] [rbp-E0h]
  __int64 v100; // [rsp+30h] [rbp-E0h]
  __int64 *v101; // [rsp+30h] [rbp-E0h]
  __int64 v102; // [rsp+30h] [rbp-E0h]
  __int64 v103; // [rsp+30h] [rbp-E0h]
  __int64 v104; // [rsp+30h] [rbp-E0h]
  __int64 v105; // [rsp+30h] [rbp-E0h]
  __int64 v106; // [rsp+30h] [rbp-E0h]
  __int64 v107; // [rsp+30h] [rbp-E0h]
  unsigned __int8 *v109; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v110[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v111; // [rsp+60h] [rbp-B0h]
  unsigned __int8 *v112[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v113; // [rsp+80h] [rbp-90h]
  unsigned __int8 *v114; // [rsp+90h] [rbp-80h] BYREF
  __int64 v115; // [rsp+98h] [rbp-78h]
  unsigned __int64 *v116; // [rsp+A0h] [rbp-70h]
  __int64 v117; // [rsp+A8h] [rbp-68h]
  __int64 v118; // [rsp+B0h] [rbp-60h]
  int v119; // [rsp+B8h] [rbp-58h]
  __int64 v120; // [rsp+C0h] [rbp-50h]
  __int64 v121; // [rsp+C8h] [rbp-48h]

  v13 = 1;
  v15 = (_QWORD **)a3;
  v16 = (__int64)a1;
  v17 = *a1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v17 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v50 = *(_QWORD *)(v17 + 32);
        v17 = *(_QWORD *)(v17 + 24);
        v13 *= v50;
        continue;
      case 1:
        v18 = 16;
        break;
      case 2:
        v18 = 32;
        break;
      case 3:
      case 9:
        v18 = 64;
        break;
      case 4:
        v18 = 80;
        break;
      case 5:
      case 6:
        v18 = 128;
        break;
      case 7:
        v18 = 8 * (unsigned int)sub_15A9520((__int64)a5, 0);
        break;
      case 0xB:
        v18 = *(_DWORD *)(v17 + 8) >> 8;
        break;
      case 0xD:
        v18 = 8LL * *(_QWORD *)sub_15A9930((__int64)a5, v17);
        break;
      case 0xE:
        v45 = *(_QWORD *)(v17 + 32);
        v99 = *(_QWORD *)(v17 + 24);
        v46 = sub_15A9FE0((__int64)a5, v99);
        v47 = v99;
        v48 = 1;
        v49 = v46;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v47 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v73 = *(_QWORD *)(v47 + 32);
              v47 = *(_QWORD *)(v47 + 24);
              v48 *= v73;
              continue;
            case 1:
              v70 = 16;
              goto LABEL_88;
            case 2:
              v70 = 32;
              goto LABEL_88;
            case 3:
            case 9:
              v70 = 64;
              goto LABEL_88;
            case 4:
              v70 = 80;
              goto LABEL_88;
            case 5:
            case 6:
              v70 = 128;
              goto LABEL_88;
            case 7:
              v95 = v49;
              v74 = 0;
              v102 = v48;
              goto LABEL_97;
            case 0xB:
              v70 = *(_DWORD *)(v47 + 8) >> 8;
              goto LABEL_88;
            case 0xD:
              v97 = v49;
              v104 = v48;
              v77 = (_QWORD *)sub_15A9930((__int64)a5, v47);
              v48 = v104;
              v49 = v97;
              v70 = 8LL * *v77;
              goto LABEL_88;
            case 0xE:
              v84 = v49;
              v86 = v48;
              v90 = *(_QWORD *)(v47 + 24);
              v103 = *(_QWORD *)(v47 + 32);
              v96 = (unsigned int)sub_15A9FE0((__int64)a5, v90);
              v76 = sub_127FA20((__int64)a5, v90);
              v48 = v86;
              v49 = v84;
              v70 = 8 * v103 * v96 * ((v96 + ((unsigned __int64)(v76 + 7) >> 3) - 1) / v96);
              goto LABEL_88;
            case 0xF:
              v95 = v49;
              v102 = v48;
              v74 = *(_DWORD *)(v47 + 8) >> 8;
LABEL_97:
              v75 = sub_15A9520((__int64)a5, v74);
              v48 = v102;
              v49 = v95;
              v70 = (unsigned int)(8 * v75);
LABEL_88:
              v71 = (unsigned __int64)(v70 * v48 + 7) >> 3;
              a3 = (v49 + v71 - 1) % v49;
              v18 = 8 * v49 * v45 * ((v49 + v71 - 1) / v49);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v18 = 8 * (unsigned int)sub_15A9520((__int64)a5, *(_DWORD *)(v17 + 8) >> 8);
        break;
    }
    break;
  }
  v19 = v13;
  v20 = (__int64)v15;
  v21 = 1;
  v94 = (unsigned __int64)(v18 * v19 + 7) >> 3;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v20 + 8) )
    {
      case 1:
        v22 = 16;
        goto LABEL_10;
      case 2:
        v22 = 32;
        goto LABEL_10;
      case 3:
      case 9:
        v22 = 64;
        goto LABEL_10;
      case 4:
        v22 = 80;
        goto LABEL_10;
      case 5:
      case 6:
        v22 = 128;
        goto LABEL_10;
      case 7:
        v22 = 8 * (unsigned int)sub_15A9520((__int64)a5, 0);
        goto LABEL_10;
      case 0xB:
        v22 = *(_DWORD *)(v20 + 8) >> 8;
        goto LABEL_10;
      case 0xD:
        v22 = 8LL * *(_QWORD *)sub_15A9930((__int64)a5, v20);
        goto LABEL_10;
      case 0xE:
        v52 = *(_QWORD *)(v20 + 32);
        v100 = *(_QWORD *)(v20 + 24);
        v53 = sub_15A9FE0((__int64)a5, v100);
        v54 = v100;
        v55 = 1;
        v56 = v53;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v54 + 8) )
          {
            case 1:
              v72 = 16;
              goto LABEL_91;
            case 2:
              v72 = 32;
              goto LABEL_91;
            case 3:
            case 9:
              v72 = 64;
              goto LABEL_91;
            case 4:
              v72 = 80;
              goto LABEL_91;
            case 5:
            case 6:
              v72 = 128;
              goto LABEL_91;
            case 7:
              v91 = v56;
              v78 = 0;
              v105 = v55;
              goto LABEL_107;
            case 0xB:
              v72 = *(_DWORD *)(v54 + 8) >> 8;
              goto LABEL_91;
            case 0xD:
              v93 = v56;
              v107 = v55;
              v82 = (_QWORD *)sub_15A9930((__int64)a5, v54);
              v55 = v107;
              v56 = v93;
              v72 = 8LL * *v82;
              goto LABEL_91;
            case 0xE:
              v83 = v56;
              v85 = v55;
              v87 = *(_QWORD *)(v54 + 24);
              v106 = *(_QWORD *)(v54 + 32);
              v92 = (unsigned int)sub_15A9FE0((__int64)a5, v87);
              v81 = sub_127FA20((__int64)a5, v87);
              v55 = v85;
              v56 = v83;
              v72 = 8 * v106 * v92 * ((v92 + ((unsigned __int64)(v81 + 7) >> 3) - 1) / v92);
              goto LABEL_91;
            case 0xF:
              v91 = v56;
              v105 = v55;
              v78 = *(_DWORD *)(v54 + 8) >> 8;
LABEL_107:
              v79 = sub_15A9520((__int64)a5, v78);
              v55 = v105;
              v56 = v91;
              v72 = (unsigned int)(8 * v79);
LABEL_91:
              v22 = 8 * v56 * v52 * ((v56 + ((unsigned __int64)(v72 * v55 + 7) >> 3) - 1) / v56);
              goto LABEL_10;
            case 0x10:
              v80 = *(_QWORD *)(v54 + 32);
              v54 = *(_QWORD *)(v54 + 24);
              v55 *= v80;
              continue;
            default:
              goto LABEL_4;
          }
        }
      case 0xF:
        v22 = 8 * (unsigned int)sub_15A9520((__int64)a5, *(_DWORD *)(v20 + 8) >> 8);
LABEL_10:
        v23 = ((unsigned __int64)(v22 * v21 + 7) >> 3) + a2;
        v98 = v23;
        if ( v23 <= (unsigned int)v94 )
          return sub_1B725B0((__int64 ***)v16, a2, (__int64)v15, a4, a5);
        if ( !v23 || ((v23 - 1) & v23) != 0 )
          v98 = (((((((((v23 | ((unsigned __int64)v23 >> 1)) >> 2) | v23 | ((unsigned __int64)v23 >> 1)) >> 4)
                   | ((v23 | ((unsigned __int64)v23 >> 1)) >> 2)
                   | v23
                   | ((unsigned __int64)v23 >> 1)) >> 8)
                 | ((((v23 | ((unsigned __int64)v23 >> 1)) >> 2) | v23 | ((unsigned __int64)v23 >> 1)) >> 4)
                 | ((v23 | ((unsigned __int64)v23 >> 1)) >> 2)
                 | v23
                 | ((unsigned __int64)v23 >> 1)) >> 16)
               | ((((((v23 | ((unsigned __int64)v23 >> 1)) >> 2) | v23 | ((unsigned __int64)v23 >> 1)) >> 4)
                 | ((v23 | ((unsigned __int64)v23 >> 1)) >> 2)
                 | v23
                 | ((unsigned __int64)v23 >> 1)) >> 8)
               | ((((v23 | ((unsigned __int64)v23 >> 1)) >> 2) | v23 | ((unsigned __int64)v23 >> 1)) >> 4)
               | ((v23 | ((unsigned __int64)v23 >> 1)) >> 2)
               | v23
               | (v23 >> 1))
              + 1;
        v24 = a1[5];
        v25 = (__int64 ***)*(a1 - 3);
        v89 = a1[4];
        v115 = v24;
        v114 = 0;
        v117 = sub_157E9C0(v24);
        v118 = 0;
        v119 = 0;
        v120 = 0;
        v121 = 0;
        v116 = (unsigned __int64 *)v89;
        if ( v89 != v24 + 40 )
        {
          if ( !v89 )
            BUG();
          v26 = *(unsigned __int8 **)(v89 + 24);
          v112[0] = v26;
          if ( v26 )
          {
            sub_1623A60((__int64)v112, (__int64)v26, 2);
            if ( v114 )
              sub_161E7C0((__int64)&v114, (__int64)v114);
            v114 = v112[0];
            if ( v112[0] )
              sub_1623210((__int64)v112, v112[0], (__int64)&v114);
          }
        }
        v27 = (__int64 *)sub_1644900(*v15, 8 * v98);
        v28 = *v25;
        if ( *((_BYTE *)*v25 + 8) == 16 )
          v28 = (__int64 **)*v28[2];
        v29 = sub_1646BA0(v27, *((_DWORD *)v28 + 2) >> 8);
        v30 = *(unsigned __int8 **)(v16 + 48);
        v31 = (__int64 **)v29;
        v112[0] = v30;
        if ( v30 )
        {
          sub_1623A60((__int64)v112, (__int64)v30, 2);
          v32 = v114;
          if ( !v114 )
            goto LABEL_26;
        }
        else
        {
          v32 = v114;
          if ( !v114 )
            goto LABEL_28;
        }
        sub_161E7C0((__int64)&v114, (__int64)v32);
LABEL_26:
        v114 = v112[0];
        if ( v112[0] )
          sub_1623210((__int64)v112, v112[0], (__int64)&v114);
LABEL_28:
        v111 = 257;
        if ( v31 != *v25 )
        {
          if ( *((_BYTE *)v25 + 16) > 0x10u )
          {
            v113 = 257;
            v64 = sub_15FDBD0(47, (__int64)v25, (__int64)v31, (__int64)v112, 0);
            v25 = (__int64 ***)v64;
            if ( v115 )
            {
              v65 = v116;
              sub_157E9D0(v115 + 40, v64);
              v66 = v25[3];
              v67 = *v65;
              v25[4] = (__int64 **)v65;
              v67 &= 0xFFFFFFFFFFFFFFF8LL;
              v25[3] = (__int64 **)(v67 | (unsigned __int8)v66 & 7);
              *(_QWORD *)(v67 + 8) = v25 + 3;
              *v65 = *v65 & 7 | (unsigned __int64)(v25 + 3);
            }
            sub_164B780((__int64)v25, v110);
            if ( v114 )
            {
              v109 = v114;
              sub_1623A60((__int64)&v109, (__int64)v114, 2);
              v68 = (__int64)v25[6];
              if ( v68 )
                sub_161E7C0((__int64)(v25 + 6), v68);
              v69 = v109;
              v25[6] = (__int64 **)v109;
              if ( v69 )
                sub_1623210((__int64)&v109, v69, (__int64)(v25 + 6));
            }
          }
          else
          {
            v25 = (__int64 ***)sub_15A46C0(47, v25, v31, 0);
          }
        }
        v113 = 257;
        v33 = sub_1648A60(64, 1u);
        if ( v33 )
          sub_15F9210((__int64)v33, (__int64)(*v25)[3], (__int64)v25, 0, 0, 0);
        if ( v115 )
        {
          v34 = v116;
          sub_157E9D0(v115 + 40, (__int64)v33);
          v35 = v33[3];
          v36 = *v34;
          v33[4] = (__int64)v34;
          v36 &= 0xFFFFFFFFFFFFFFF8LL;
          v33[3] = v36 | v35 & 7;
          *(_QWORD *)(v36 + 8) = v33 + 3;
          *v34 = *v34 & 7 | (unsigned __int64)(v33 + 3);
        }
        sub_164B780((__int64)v33, (__int64 *)v112);
        if ( v114 )
        {
          v110[0] = (__int64)v114;
          sub_1623A60((__int64)v110, (__int64)v114, 2);
          v37 = v33[6];
          v38 = (__int64)(v33 + 6);
          if ( v37 )
          {
            sub_161E7C0((__int64)(v33 + 6), v37);
            v38 = (__int64)(v33 + 6);
          }
          v39 = (unsigned __int8 *)v110[0];
          v33[6] = v110[0];
          if ( v39 )
            sub_1623210((__int64)v110, v39, v38);
        }
        v40 = (__int64)v33;
        sub_164B7C0((__int64)v33, v16);
        sub_15F8F50((__int64)v33, 1 << (*(unsigned __int16 *)(v16 + 18) >> 1) >> 1);
        if ( *a5 )
        {
          v113 = 257;
          v57 = sub_15A0680(*v33, 8 * (v98 - (unsigned int)v94), 0);
          v40 = sub_156E320((__int64 *)&v114, (__int64)v33, v57, (__int64)v112, 0);
        }
        v43 = *(__int64 ***)v16;
        v111 = 257;
        if ( v43 != *(__int64 ***)v40 )
        {
          if ( *(_BYTE *)(v40 + 16) > 0x10u )
          {
            v113 = 257;
            v58 = sub_15FDBD0(36, v40, (__int64)v43, (__int64)v112, 0);
            v40 = v58;
            if ( v115 )
            {
              v101 = (__int64 *)v116;
              sub_157E9D0(v115 + 40, v58);
              v59 = *v101;
              v60 = *(_QWORD *)(v40 + 24) & 7LL;
              *(_QWORD *)(v40 + 32) = v101;
              v59 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v40 + 24) = v59 | v60;
              *(_QWORD *)(v59 + 8) = v40 + 24;
              *v101 = *v101 & 7 | (v40 + 24);
            }
            sub_164B780(v40, v110);
            if ( v114 )
            {
              v109 = v114;
              sub_1623A60((__int64)&v109, (__int64)v114, 2);
              v61 = *(_QWORD *)(v40 + 48);
              v62 = v40 + 48;
              if ( v61 )
              {
                sub_161E7C0(v40 + 48, v61);
                v62 = v40 + 48;
              }
              v63 = v109;
              *(_QWORD *)(v40 + 48) = v109;
              if ( v63 )
                sub_1623210((__int64)&v109, v63, v62);
            }
          }
          else
          {
            v40 = sub_15A46C0(36, (__int64 ***)v40, v43, 0);
          }
        }
        sub_164D160(v16, v40, a6, a7, a8, a9, v41, v42, a12, a13);
        if ( v114 )
          sub_161E7C0((__int64)&v114, (__int64)v114);
        v16 = (__int64)v33;
        return sub_1B725B0((__int64 ***)v16, a2, (__int64)v15, a4, a5);
      case 0x10:
        v51 = *(_QWORD *)(v20 + 32);
        v20 = *(_QWORD *)(v20 + 24);
        v21 *= v51;
        continue;
      default:
LABEL_4:
        MEMORY[0] = a3;
        BUG();
    }
  }
}
