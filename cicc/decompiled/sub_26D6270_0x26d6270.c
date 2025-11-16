// Function: sub_26D6270
// Address: 0x26d6270
//
__int64 __fastcall sub_26D6270(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r13
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // r12
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  _QWORD *v16; // rdi
  char v17; // al
  __int64 v18; // rdx
  bool v19; // r12
  __int64 *v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r12
  unsigned int v24; // ecx
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // rax
  unsigned __int8 v28; // r12
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned int v33; // ecx
  _QWORD *v34; // r13
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r12
  unsigned int v39; // eax
  _QWORD *v40; // r15
  char v41; // r14
  _QWORD *v42; // r13
  __int64 v43; // rbx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rbx
  unsigned int v47; // eax
  _QWORD *v48; // r9
  _QWORD *v49; // r13
  __int64 v50; // r12
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r15
  unsigned int v54; // eax
  __int64 v55; // r14
  __int64 v56; // r15
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  unsigned __int64 v61; // r12
  _QWORD *v62; // rax
  __int64 v63; // rsi
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rax
  unsigned __int64 v68; // rdx
  char *v69; // r11
  char *v70; // rbx
  __int64 v71; // r12
  _BOOL8 v72; // rax
  bool v73; // zf
  unsigned __int8 v74; // r12
  unsigned int v75; // edi
  __int64 *v76; // rax
  __int64 v77; // rsi
  _QWORD *v78; // rsi
  _QWORD *v79; // r15
  unsigned __int64 v80; // rax
  int v81; // edx
  int v82; // r9d
  unsigned int v83; // ecx
  __int64 *v84; // rax
  __int64 v85; // rdi
  _QWORD *v86; // rsi
  int v88; // eax
  int v89; // r9d
  int v90; // eax
  int v91; // r9d
  __int64 v93; // [rsp+18h] [rbp-2C8h]
  _QWORD *v94; // [rsp+20h] [rbp-2C0h]
  _QWORD *v95; // [rsp+28h] [rbp-2B8h]
  __int64 v96; // [rsp+30h] [rbp-2B0h]
  __int64 v97; // [rsp+38h] [rbp-2A8h]
  __int64 v98; // [rsp+40h] [rbp-2A0h]
  __int64 v99; // [rsp+48h] [rbp-298h]
  _QWORD *v100; // [rsp+50h] [rbp-290h]
  _QWORD *v101; // [rsp+58h] [rbp-288h]
  _QWORD *v102; // [rsp+60h] [rbp-280h]
  __int64 v103; // [rsp+68h] [rbp-278h]
  _QWORD *v104; // [rsp+70h] [rbp-270h]
  __int64 *v105; // [rsp+70h] [rbp-270h]
  __int64 v107; // [rsp+A0h] [rbp-240h]
  int v108; // [rsp+A0h] [rbp-240h]
  unsigned __int8 v109; // [rsp+ADh] [rbp-233h]
  char v110; // [rsp+AEh] [rbp-232h]
  char v111; // [rsp+AFh] [rbp-231h]
  unsigned __int8 v112; // [rsp+AFh] [rbp-231h]
  char v113; // [rsp+B0h] [rbp-230h]
  char *v114; // [rsp+B0h] [rbp-230h]
  __int64 v115; // [rsp+B8h] [rbp-228h]
  char *v116; // [rsp+B8h] [rbp-228h]
  __int64 v117; // [rsp+C0h] [rbp-220h] BYREF
  __int64 v118; // [rsp+C8h] [rbp-218h]
  __int64 v119; // [rsp+D0h] [rbp-210h]
  unsigned int v120; // [rsp+D8h] [rbp-208h]
  _DWORD *v121; // [rsp+E0h] [rbp-200h]
  __int64 v122; // [rsp+E8h] [rbp-1F8h]
  _DWORD v123[6]; // [rsp+F0h] [rbp-1F0h] BYREF
  unsigned __int64 v124; // [rsp+108h] [rbp-1D8h]
  unsigned int v125; // [rsp+110h] [rbp-1D0h]
  unsigned __int64 v126; // [rsp+118h] [rbp-1C8h]
  unsigned int v127; // [rsp+120h] [rbp-1C0h]
  bool v128; // [rsp+128h] [rbp-1B8h]
  __int64 v129; // [rsp+130h] [rbp-1B0h] BYREF
  __int64 v130[11]; // [rsp+138h] [rbp-1A8h] BYREF
  char *v131; // [rsp+190h] [rbp-150h] BYREF
  __int64 v132; // [rsp+198h] [rbp-148h]
  _BYTE v133[80]; // [rsp+1A0h] [rbp-140h] BYREF
  char *v134; // [rsp+1F0h] [rbp-F0h] BYREF
  __int64 v135; // [rsp+1F8h] [rbp-E8h]
  _QWORD v136[10]; // [rsp+200h] [rbp-E0h] BYREF
  char *v137; // [rsp+250h] [rbp-90h] BYREF
  __int128 v138; // [rsp+258h] [rbp-88h] BYREF
  __int64 v139; // [rsp+268h] [rbp-78h]
  char *v140; // [rsp+270h] [rbp-70h]
  char v141; // [rsp+280h] [rbp-60h] BYREF

  v121 = v123;
  v103 = a2 + 72;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v122 = 0;
  v109 = 0;
LABEL_2:
  v131 = v133;
  v132 = 0xA00000000LL;
  v115 = *(_QWORD *)(a2 + 80);
  if ( v115 != v103 )
  {
    while ( 1 )
    {
      if ( !v115 )
      {
        v134 = (char *)v136;
        v137 = (char *)&v138 + 8;
        v135 = 0xA00000000LL;
        *(_QWORD *)&v138 = 0xA00000000LL;
        BUG();
      }
      v111 = 0;
      v135 = 0xA00000000LL;
      v134 = (char *)v136;
      *(_QWORD *)&v138 = 0xA00000000LL;
      v137 = (char *)&v138 + 8;
      v4 = *(_QWORD *)(v115 + 32);
      if ( v115 + 24 != v4 )
        break;
LABEL_23:
      if ( !*(_QWORD *)(a1 + 1712) )
      {
        sub_26BA960((__int64)&v131, v131, v137, &v137[8 * (unsigned int)v138]);
        sub_26C21F0(a1, (__int64)&v137, (unsigned __int8 *)a2, 0);
        goto LABEL_25;
      }
LABEL_24:
      sub_26BA960((__int64)&v131, v131, v134, &v134[8 * (unsigned int)v135]);
      sub_26C21F0(a1, (__int64)&v134, (unsigned __int8 *)a2, 1);
LABEL_25:
      if ( v137 != (char *)&v138 + 8 )
        _libc_free((unsigned __int64)v137);
      if ( v134 != (char *)v136 )
        _libc_free((unsigned __int64)v134);
      v115 = *(_QWORD *)(v115 + 8);
      if ( v115 == v103 )
      {
        v20 = (__int64 *)v131;
        v116 = &v131[8 * (unsigned int)v132];
        if ( v116 == v131 )
        {
          if ( v116 != v133 )
            _libc_free((unsigned __int64)v116);
          goto LABEL_176;
        }
        v112 = 0;
        while ( 1 )
        {
          v29 = *v20;
          v23 = *(_QWORD *)(*v20 - 32);
          if ( v23 && !*(_BYTE *)v23 && *(_QWORD *)(v23 + 24) == *(_QWORD *)(v29 + 80) )
          {
            v21 = v120;
            v139 = 1065353216;
            v137 = (char *)v29;
            v22 = v118;
            v138 = 0;
            if ( !v120 )
              goto LABEL_49;
          }
          else
          {
            v21 = v120;
            v139 = 1065353216;
            v137 = (char *)v29;
            v22 = v118;
            v138 = 0;
            if ( !v120 )
            {
              *(_QWORD *)&v138 = 0;
              if ( sub_B491E0(v29) )
                goto LABEL_138;
              goto LABEL_101;
            }
            v23 = 0;
          }
          v24 = (v21 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
          v25 = v22 + 16LL * v24;
          v26 = *(_QWORD *)v25;
          if ( v29 != *(_QWORD *)v25 )
          {
            v81 = 1;
            while ( v26 != -4096 )
            {
              v82 = v81 + 1;
              v24 = (v21 - 1) & (v81 + v24);
              v25 = v22 + 16LL * v24;
              v26 = *(_QWORD *)v25;
              if ( v29 == *(_QWORD *)v25 )
                goto LABEL_35;
              v81 = v82;
            }
LABEL_49:
            v27 = 0;
            goto LABEL_37;
          }
LABEL_35:
          if ( v25 == v22 + 16 * v21 )
            goto LABEL_49;
          v27 = *(_QWORD *)&v121[4 * *(unsigned int *)(v25 + 8) + 2];
LABEL_37:
          *(_QWORD *)&v138 = v27;
          if ( a2 == v23 )
            goto LABEL_44;
          if ( sub_B491E0(v29) )
          {
LABEL_138:
            sub_26CB210((__int64)&v134, a1, v29, &v129);
            v69 = v134;
            v114 = (char *)v135;
            if ( (char *)v135 != v134 )
            {
              v105 = v20;
              v70 = v134;
              do
              {
                while ( 1 )
                {
                  v79 = *(_QWORD **)v70;
                  if ( *(_DWORD *)(a1 + 1520) != 1 )
                    break;
                  v70 += 8;
                  v80 = sub_D844E0(*(_QWORD *)(a1 + 1280));
                  sub_26CF830(a1, v29, (unsigned __int64)v79, a3, v80);
                  if ( v114 == v70 )
                    goto LABEL_153;
                }
                v71 = v129;
                if ( (unsigned __int8)sub_2A60EC0(
                                        *(_QWORD *)v70,
                                        *(_QWORD *)(a1 + 1280),
                                        *(unsigned __int8 *)(a1 + 1704)) )
                {
                  v72 = sub_EF9210(v79);
                  v73 = *(_BYTE *)(a1 + 1705) == 0;
                  v137 = (char *)v29;
                  *(_QWORD *)&v138 = v79;
                  *((_QWORD *)&v138 + 1) = v72;
                  LODWORD(v139) = 1065353216;
                  if ( v73 )
                  {
                    if ( (_DWORD)qword_4FF62C8 )
                    {
                      v74 = sub_26CC080(a1, a2, (__int64 *)&v137, v71, (unsigned __int64 *)&v129, 0);
                      if ( v74 )
                      {
                        if ( v120 )
                        {
                          v75 = (v120 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                          v76 = (__int64 *)(v118 + 16LL * v75);
                          v77 = *v76;
                          if ( v29 == *v76 )
                          {
LABEL_146:
                            if ( v76 != (__int64 *)(v118 + 16LL * v120) )
                            {
                              v78 = &v121[4 * *((unsigned int *)v76 + 2)];
                              if ( v78 != (_QWORD *)&v121[4 * (unsigned int)v122] )
                                sub_26BAD40((__int64)&v117, v78);
                            }
                          }
                          else
                          {
                            v88 = 1;
                            while ( v77 != -4096 )
                            {
                              v89 = v88 + 1;
                              v75 = (v120 - 1) & (v88 + v75);
                              v76 = (__int64 *)(v118 + 16LL * v75);
                              v77 = *v76;
                              if ( v29 == *v76 )
                                goto LABEL_146;
                              v88 = v89;
                            }
                          }
                        }
                        v112 = v74;
                      }
                    }
                  }
                }
                v70 += 8;
              }
              while ( v114 != v70 );
LABEL_153:
              v20 = v105;
              v69 = v134;
            }
            if ( v69 )
              j_j___libc_free_0((unsigned __int64)v69);
LABEL_44:
            if ( v116 == (char *)++v20 )
              goto LABEL_103;
            continue;
          }
          if ( v23 && sub_B92180(v23) && !sub_B2FC80(v23) )
          {
            if ( !*(_BYTE *)(a1 + 1705) )
            {
              v28 = sub_26C3F00(a1, (__int64)&v137, 0);
              if ( v28 )
              {
                if ( v120 )
                {
                  v83 = (v120 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                  v84 = (__int64 *)(v118 + 16LL * v83);
                  v85 = *v84;
                  if ( v29 == *v84 )
                  {
LABEL_170:
                    if ( v84 != (__int64 *)(v118 + 16LL * v120) )
                    {
                      v86 = &v121[4 * *((unsigned int *)v84 + 2)];
                      if ( v86 != (_QWORD *)&v121[4 * (unsigned int)v122] )
                        sub_26BAD40((__int64)&v117, v86);
                    }
                  }
                  else
                  {
                    v90 = 1;
                    while ( v85 != -4096 )
                    {
                      v91 = v90 + 1;
                      v83 = (v120 - 1) & (v90 + v83);
                      v84 = (__int64 *)(v118 + 16LL * v83);
                      v85 = *v84;
                      if ( v29 == *v84 )
                        goto LABEL_170;
                      v90 = v91;
                    }
                  }
                }
                v112 = v28;
              }
            }
            goto LABEL_44;
          }
LABEL_101:
          if ( *(_DWORD *)(a1 + 1520) != 1 )
            goto LABEL_44;
          ++v20;
          v61 = sub_D844E0(*(_QWORD *)(a1 + 1280));
          v62 = sub_26CAE90((_QWORD *)a1, v29);
          sub_26CF830(a1, v29, (unsigned __int64)v62, a3, v61);
          if ( v116 == (char *)v20 )
          {
LABEL_103:
            v109 |= v112;
            if ( v131 != v133 )
              _libc_free((unsigned __int64)v131);
            if ( v112 )
            {
              v109 = v112;
              goto LABEL_2;
            }
            goto LABEL_176;
          }
        }
      }
    }
    v5 = a1;
    v6 = v115 + 24;
    v7 = v5;
    while ( 1 )
    {
      if ( !v4 )
        BUG();
      v8 = *(_BYTE *)(v4 - 24);
      if ( v8 == 85 )
      {
        v9 = *(_QWORD *)(v4 - 56);
        if ( v9 && !*(_BYTE *)v9 && *(_QWORD *)(v9 + 24) == *(_QWORD *)(v4 + 56) && (*(_BYTE *)(v9 + 33) & 0x20) != 0 )
          goto LABEL_8;
      }
      else if ( v8 != 34 && v8 != 40 )
      {
        goto LABEL_8;
      }
      v10 = v4 - 24;
      v13 = sub_26CAE90((_QWORD *)v7, v4 - 24);
      if ( !v13 )
      {
        if ( sub_26C3E80(v7, v4 - 24) )
        {
          v59 = (unsigned int)v135;
          v60 = (unsigned int)v135 + 1LL;
          if ( v60 > HIDWORD(v135) )
          {
            sub_C8D5F0((__int64)&v134, v136, v60, 8u, v57, v58);
            v59 = (unsigned int)v135;
          }
          *(_QWORD *)&v134[8 * v59] = v10;
          LODWORD(v135) = v135 + 1;
        }
        goto LABEL_8;
      }
      v14 = (unsigned int)v135;
      v15 = (unsigned int)v135 + 1LL;
      if ( v15 > HIDWORD(v135) )
      {
        sub_C8D5F0((__int64)&v134, v136, v15, 8u, v11, v12);
        v14 = (unsigned int)v135;
      }
      *(_QWORD *)&v134[8 * v14] = v10;
      LODWORD(v135) = v135 + 1;
      v113 = unk_4F838D3;
      if ( !unk_4F838D3 || !v13[8] )
      {
        v30 = v13[20];
        if ( !v13[14] )
        {
          if ( !v30 )
            goto LABEL_91;
          v32 = v13[18];
LABEL_54:
          v104 = (_QWORD *)(v32 + 48);
          if ( *(_QWORD *)(v32 + 64) == v32 + 48 )
          {
LABEL_91:
            if ( !v13[7] && !v113 )
              goto LABEL_20;
            goto LABEL_19;
          }
          v107 = 0;
          v100 = v13;
          v99 = v6;
          v98 = v4 - 24;
          v97 = v4;
          v96 = v7;
          v34 = *(_QWORD **)(v32 + 64);
          while ( 2 )
          {
            if ( v113 )
            {
              v35 = v34[14];
              if ( v35 )
              {
LABEL_88:
                v107 += v35;
                v34 = (_QWORD *)sub_220EF30((__int64)v34);
                if ( v104 == v34 )
                {
                  v13 = v100;
                  v6 = v99;
                  v10 = v98;
                  v4 = v97;
                  v7 = v96;
                  goto LABEL_90;
                }
                continue;
              }
            }
            break;
          }
          v36 = v34[26];
          if ( v34[20] )
          {
            v37 = v34[18];
            if ( !v36
              || (v38 = v34[24], v39 = *(_DWORD *)(v38 + 32), *(_DWORD *)(v37 + 32) < v39)
              || *(_DWORD *)(v37 + 32) == v39 && *(_DWORD *)(v37 + 36) < *(_DWORD *)(v38 + 36) )
            {
              v35 = *(_QWORD *)(v37 + 40);
LABEL_87:
              if ( v35 )
                goto LABEL_88;
LABEL_99:
              v35 = v34[13] != 0;
              goto LABEL_88;
            }
          }
          else
          {
            if ( !v36 )
              goto LABEL_99;
            v38 = v34[24];
          }
          v40 = *(_QWORD **)(v38 + 64);
          v102 = (_QWORD *)(v38 + 48);
          if ( v40 == (_QWORD *)(v38 + 48) )
            goto LABEL_99;
          v41 = v113;
          v95 = v34;
          v35 = 0;
          v42 = v40;
          while ( 2 )
          {
            if ( v41 )
            {
              v43 = v42[14];
              if ( v43 )
              {
LABEL_85:
                v35 += v43;
                v42 = (_QWORD *)sub_220EF30((__int64)v42);
                if ( v102 == v42 )
                {
                  v34 = v95;
                  goto LABEL_87;
                }
                continue;
              }
            }
            break;
          }
          v44 = v42[26];
          if ( v42[20] )
          {
            v45 = v42[18];
            if ( !v44
              || (v46 = v42[24], v47 = *(_DWORD *)(v46 + 32), *(_DWORD *)(v45 + 32) < v47)
              || *(_DWORD *)(v45 + 32) == v47 && *(_DWORD *)(v45 + 36) < *(_DWORD *)(v46 + 36) )
            {
              v43 = *(_QWORD *)(v45 + 40);
LABEL_84:
              if ( v43 )
                goto LABEL_85;
LABEL_122:
              v43 = v42[13] != 0;
              goto LABEL_85;
            }
          }
          else
          {
            if ( !v44 )
              goto LABEL_122;
            v46 = v42[24];
          }
          v48 = *(_QWORD **)(v46 + 64);
          v101 = (_QWORD *)(v46 + 48);
          if ( v48 == (_QWORD *)(v46 + 48) )
            goto LABEL_122;
          v110 = v41;
          v43 = 0;
          v93 = v35;
          v94 = v42;
          v49 = v48;
          while ( 2 )
          {
            if ( v110 )
            {
              v50 = v49[14];
              if ( v50 )
                goto LABEL_82;
            }
            v51 = v49[26];
            if ( v49[20] )
            {
              v52 = v49[18];
              if ( !v51
                || (v53 = v49[24], v54 = *(_DWORD *)(v53 + 32), *(_DWORD *)(v52 + 32) < v54)
                || *(_DWORD *)(v52 + 32) == v54 && *(_DWORD *)(v52 + 36) < *(_DWORD *)(v53 + 36) )
              {
                v50 = *(_QWORD *)(v52 + 40);
                goto LABEL_81;
              }
LABEL_78:
              v55 = *(_QWORD *)(v53 + 64);
              v56 = v53 + 48;
              if ( v55 != v56 )
              {
                v50 = 0;
                do
                {
                  v50 += sub_EF9210((_QWORD *)(v55 + 48));
                  v55 = sub_220EF30(v55);
                }
                while ( v56 != v55 );
LABEL_81:
                if ( v50 )
                {
LABEL_82:
                  v43 += v50;
                  v49 = (_QWORD *)sub_220EF30((__int64)v49);
                  if ( v101 == v49 )
                  {
                    v42 = v94;
                    v41 = v110;
                    v35 = v93;
                    goto LABEL_84;
                  }
                  continue;
                }
              }
            }
            else if ( v51 )
            {
              v53 = v49[24];
              goto LABEL_78;
            }
            break;
          }
          v50 = v49[13] != 0;
          goto LABEL_82;
        }
        v31 = v13[12];
        if ( v30 )
        {
          v32 = v13[18];
          v33 = *(_DWORD *)(v32 + 32);
          if ( *(_DWORD *)(v31 + 32) >= v33
            && (*(_DWORD *)(v31 + 32) != v33 || *(_DWORD *)(v31 + 36) >= *(_DWORD *)(v32 + 36)) )
          {
            goto LABEL_54;
          }
        }
        v107 = *(_QWORD *)(v31 + 40);
LABEL_90:
        if ( !v107 )
          goto LABEL_91;
      }
LABEL_19:
      v129 = v10;
      v130[0] = (__int64)v13;
      sub_26D54A0((__int64)&v117, &v129, v130);
LABEL_20:
      v16 = v13;
      v17 = sub_2A60EC0(v13, *(_QWORD *)(v7 + 1280), *(unsigned __int8 *)(v7 + 1704));
      v19 = v17;
      if ( v17 )
      {
        v4 = *(_QWORD *)(v4 + 8);
        v111 = v17;
        if ( v6 == v4 )
        {
LABEL_22:
          a1 = v7;
          if ( v111 )
            goto LABEL_24;
          goto LABEL_23;
        }
      }
      else
      {
        if ( !byte_4FF79C8 )
          goto LABEL_8;
        v63 = *(_QWORD *)(v4 - 56);
        if ( !v63 || *(_BYTE *)v63 || *(_QWORD *)(v63 + 24) != *(_QWORD *)(v4 + 56) )
          goto LABEL_8;
        if ( !*(_QWORD *)(v7 + 1456) )
          sub_4263D6(v16, v63, v18);
        v108 = (*(__int64 (__fastcall **)(__int64))(v7 + 1464))(v7 + 1440);
        sub_30D6B30(&v129, v63, &v129, v64);
        sub_30DF350(
          (unsigned int)v123,
          v10,
          (unsigned int)&v129,
          v108,
          (unsigned int)sub_26B9F70,
          v7 + 1408,
          (__int64)sub_24258E0,
          v7 + 1472,
          0,
          0,
          0,
          0);
        if ( v123[0] == 0x7FFFFFFF )
        {
          if ( !v128 )
            goto LABEL_8;
        }
        else if ( v123[0] == 0x80000000 )
        {
          if ( !v128 )
          {
LABEL_118:
            v67 = (unsigned int)v138;
            v68 = (unsigned int)v138 + 1LL;
            if ( v68 > DWORD1(v138) )
            {
              sub_C8D5F0((__int64)&v137, (char *)&v138 + 8, v68, 8u, v65, v66);
              v67 = (unsigned int)v138;
            }
            *(_QWORD *)&v137[8 * v67] = v10;
            LODWORD(v138) = v138 + 1;
            goto LABEL_8;
          }
          v19 = v128;
        }
        else
        {
          v19 = v123[0] <= SLODWORD(qword_4FF7320[17]);
          if ( !v128 )
            goto LABEL_117;
        }
        v128 = 0;
        if ( v127 > 0x40 && v126 )
          j_j___libc_free_0_0(v126);
        if ( v125 > 0x40 && v124 )
          j_j___libc_free_0_0(v124);
LABEL_117:
        if ( v19 )
          goto LABEL_118;
LABEL_8:
        v4 = *(_QWORD *)(v4 + 8);
        if ( v6 == v4 )
          goto LABEL_22;
      }
    }
  }
LABEL_176:
  if ( !unk_4F838D3 )
  {
    sub_26C1C00((__int64)&v137, (__int64)&v117);
    sub_26C8F20(a1, (__int64)&v137, (unsigned __int8 *)a2);
    if ( v140 != &v141 )
      _libc_free((unsigned __int64)v140);
    sub_C7D6A0(v138, 16LL * (unsigned int)v139, 8);
  }
  if ( v121 != v123 )
    _libc_free((unsigned __int64)v121);
  sub_C7D6A0(v118, 16LL * v120, 8);
  return v109;
}
