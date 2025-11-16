// Function: sub_3032470
// Address: 0x3032470
//
void __fastcall sub_3032470(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // edx
  __int64 v13; // r15
  __int16 *v14; // rax
  unsigned __int16 v15; // cx
  __int16 v16; // si
  __int16 v17; // dx
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned int v20; // r15d
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r13
  __int64 v26; // rdx
  __int64 v27; // r12
  __int64 *v28; // rdx
  __int64 *v29; // rax
  __int64 v30; // r12
  unsigned __int64 v31; // r13
  __int64 v32; // rax
  unsigned __int16 v33; // cx
  __int64 v34; // rax
  bool v35; // al
  __int64 v36; // rcx
  __int64 v37; // rax
  int v38; // edx
  unsigned int v39; // eax
  __int64 v40; // r15
  __int128 v41; // rax
  int v42; // r9d
  __int64 v43; // rax
  __int64 v44; // rdx
  unsigned __int64 v45; // rdx
  unsigned __int64 *v46; // rax
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 *v49; // rax
  __int16 *v50; // rax
  unsigned __int16 v51; // dx
  __int64 v52; // rax
  __int64 v53; // rdx
  unsigned __int16 *v54; // rcx
  int v55; // r10d
  __int16 v56; // ax
  __int16 v57; // r15
  __int64 v58; // rdx
  __int64 v59; // r13
  unsigned __int64 v60; // rcx
  __int16 v61; // bx
  __int64 v62; // rdx
  __int64 v63; // r15
  int v64; // r12d
  int v65; // r14d
  __int64 v66; // r13
  __int64 **v67; // rax
  __int64 *v68; // rdx
  __int64 *v69; // rsi
  __int64 v70; // r12
  __int64 v71; // r13
  int v72; // eax
  int v73; // edx
  int v74; // r9d
  __int64 v75; // rax
  int v76; // r10d
  __int64 v77; // r9
  __int128 v78; // rax
  unsigned int v79; // r13d
  int v80; // r15d
  __int64 v81; // r14
  unsigned __int64 v82; // rbx
  unsigned __int64 j; // rcx
  unsigned __int64 v84; // r12
  __int64 *v85; // rdx
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // r12
  __int64 v89; // rdx
  __int64 v90; // r13
  __int64 v91; // rax
  __int64 *v92; // rax
  __int64 *v93; // rdi
  __int64 v94; // rdi
  __int64 v95; // rax
  __int64 v96; // rax
  char v97; // dl
  char v98; // cl
  const char *v99; // rax
  int v100; // esi
  const char *v101; // rdx
  char v102; // r9
  int i; // edx
  unsigned __int16 v104; // dx
  __int16 v105; // si
  __int16 v106; // cx
  unsigned __int16 v107; // cx
  __int16 v108; // si
  __int16 v109; // dx
  __int128 v110; // [rsp-20h] [rbp-440h]
  __int64 v111; // [rsp-8h] [rbp-428h]
  __int64 v112; // [rsp+8h] [rbp-418h]
  unsigned __int64 v113; // [rsp+10h] [rbp-410h]
  __int64 v114; // [rsp+18h] [rbp-408h]
  unsigned __int64 v115; // [rsp+20h] [rbp-400h]
  int v116; // [rsp+28h] [rbp-3F8h]
  unsigned int v117; // [rsp+2Ch] [rbp-3F4h]
  __int16 v119; // [rsp+42h] [rbp-3DEh]
  __int64 v120; // [rsp+48h] [rbp-3D8h]
  __int64 v121; // [rsp+48h] [rbp-3D8h]
  int v122; // [rsp+50h] [rbp-3D0h]
  int v123; // [rsp+58h] [rbp-3C8h]
  int v124; // [rsp+5Ch] [rbp-3C4h]
  __int64 v125; // [rsp+60h] [rbp-3C0h]
  __int64 **v126; // [rsp+60h] [rbp-3C0h]
  __int64 v128; // [rsp+70h] [rbp-3B0h]
  __int64 v129; // [rsp+70h] [rbp-3B0h]
  int v130; // [rsp+70h] [rbp-3B0h]
  int v131; // [rsp+70h] [rbp-3B0h]
  __int64 v132; // [rsp+78h] [rbp-3A8h]
  int v133; // [rsp+78h] [rbp-3A8h]
  __int64 v134; // [rsp+80h] [rbp-3A0h] BYREF
  int v135; // [rsp+88h] [rbp-398h]
  _WORD *v136; // [rsp+90h] [rbp-390h] BYREF
  __int64 v137; // [rsp+98h] [rbp-388h]
  __int64 v138; // [rsp+A0h] [rbp-380h]
  _WORD v139[36]; // [rsp+A8h] [rbp-378h] BYREF
  __int64 *v140; // [rsp+F0h] [rbp-330h] BYREF
  __int64 v141; // [rsp+F8h] [rbp-328h]
  __int64 v142; // [rsp+100h] [rbp-320h] BYREF
  _WORD v143[36]; // [rsp+108h] [rbp-318h] BYREF
  __int64 *v144; // [rsp+150h] [rbp-2D0h] BYREF
  __int64 v145; // [rsp+158h] [rbp-2C8h]
  __int64 v146; // [rsp+160h] [rbp-2C0h] BYREF
  _WORD v147[60]; // [rsp+168h] [rbp-2B8h] BYREF
  _QWORD *v148; // [rsp+1E0h] [rbp-240h] BYREF
  __int64 v149; // [rsp+1E8h] [rbp-238h]
  _QWORD v150[70]; // [rsp+1F0h] [rbp-230h] BYREF

  v4 = *(_QWORD **)(a1 + 40);
  v5 = *(_QWORD *)(*v4 + 96LL);
  v6 = *(_QWORD **)(v5 + 24);
  if ( *(_DWORD *)(v5 + 32) > 0x40u )
    v6 = (_QWORD *)*v6;
  v124 = (int)v6;
  v7 = *(_QWORD *)(a1 + 80);
  v134 = v7;
  if ( v7 )
  {
    sub_B96E90((__int64)&v134, v7, 1);
    v4 = *(_QWORD **)(a1 + 40);
  }
  v135 = *(_DWORD *)(a1 + 72);
  v8 = *(_QWORD *)(v4[5] + 96LL);
  v9 = *(_QWORD *)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = **(_QWORD **)(v8 + 24);
  if ( v124 == 9066 )
  {
    v98 = BYTE6(v9) & 0x38;
    if ( (v9 & 0x38000000000000LL) == 0 )
    {
      v99 = "\r";
      v100 = 13;
      v101 = "\r";
      while ( BYTE1(v9) != v100 )
      {
        v101 += 4;
        if ( &unk_44C7AC4 == (_UNKNOWN *)v101 )
        {
          v102 = 0;
          goto LABEL_94;
        }
        v100 = *(_DWORD *)v101;
      }
      v102 = 1;
LABEL_94:
      for ( i = 13; BYTE2(v9) != i; i = *(_DWORD *)v99 )
      {
        v99 += 4;
        if ( &unk_44C7AC4 == (_UNKNOWN *)v99 )
          goto LABEL_98;
      }
      if ( BYTE4(v9) != 19 || !v102 )
LABEL_98:
        sub_C64ED0("Invalid atype, btype and shape for .scale_vec::1X", 1u);
      goto LABEL_9;
    }
    if ( v98 != 8 )
    {
      if ( v98 == 16 && (BYTE1(v9) != 17 || BYTE2(v9) != 17) )
        sub_C64ED0("Invalid shape, atype and btype combination for .scale_vec::4X", 1u);
      goto LABEL_9;
    }
    v97 = BYTE6(v9) & 7;
    if ( BYTE4(v9) != 20 )
      goto LABEL_85;
  }
  else
  {
    if ( v124 != 9144 || (BYTE6(v9) & 0x38) != 8 )
      goto LABEL_9;
    v97 = BYTE6(v9) & 7;
    if ( BYTE4(v9) != 21 )
      goto LABEL_85;
  }
  if ( BYTE1(v9) != 17 || BYTE2(v9) != 17 || v97 )
LABEL_85:
    sub_C64ED0("Invalid shape, atype, btype and blocks_scale format combination for .scale_vec::2X", 1u);
LABEL_9:
  v150[0] = sub_3400BD0(a2, v9, (unsigned int)&v134, 8, 0, 1, 0);
  v150[1] = v12;
  v148 = v150;
  v149 = 0x2000000001LL;
  if ( v124 != 9062 )
  {
    if ( v124 == 9066 )
    {
      v13 = *(_QWORD *)(a1 + 40);
      v14 = *(__int16 **)(a1 + 48);
      v107 = *v14;
      v108 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v13 + 320) + 48LL) + 16LL * *(unsigned int *)(v13 + 328));
      v109 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v13 + 360) + 48LL) + 16LL * *(unsigned int *)(v13 + 368));
      v140 = (__int64 *)v143;
      v142 = 32;
      v143[0] = v107;
      v143[1] = v108;
      v143[2] = v109;
      v141 = 3;
      if ( (v9 & 8) == 0 )
        sub_C64ED0("nvvm.mma.blockscale currently supports non-sync aligned variants only!", 1u);
      if ( v107 == 149 && v109 == 58 && v108 == 60 )
      {
        v116 = 5746;
        goto LABEL_17;
      }
    }
    else
    {
      if ( v124 != 9144 )
        sub_C64ED0("Unexpected intrinsic ID here!", 1u);
      v13 = *(_QWORD *)(a1 + 40);
      v14 = *(__int16 **)(a1 + 48);
      v15 = *v14;
      v16 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v13 + 400) + 48LL) + 16LL * *(unsigned int *)(v13 + 408));
      v17 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v13 + 440) + 48LL) + 16LL * *(unsigned int *)(v13 + 448));
      v144 = (__int64 *)v147;
      v146 = 32;
      v147[0] = v15;
      v147[1] = v16;
      v147[2] = v17;
      v145 = 3;
      if ( (v9 & 8) == 0 )
        sub_C64ED0("nvvm.mma.sparse.blockscale currently supports non-sync aligned variants only!", 1u);
      if ( v15 == 149 && v16 == 60 && v17 == 60 )
      {
        v116 = 5747;
LABEL_17:
        v18 = *(unsigned int *)(a1 + 64);
        v123 = v18;
        if ( (_DWORD)v18 != 2 )
        {
LABEL_18:
          v19 = v13;
          v20 = 2;
          while ( 1 )
          {
            v29 = (__int64 *)(v19 + 40LL * v20);
            v18 = *v29;
            v30 = *v29;
            v31 = v29[1];
            v132 = *v29;
            v128 = *((unsigned int *)v29 + 2);
            v32 = *(_QWORD *)(*v29 + 48) + 16 * v128;
            v33 = *(_WORD *)v32;
            v34 = *(_QWORD *)(v32 + 8);
            LOWORD(v144) = v33;
            v145 = v34;
            if ( v33 )
            {
              if ( (unsigned __int16)(v33 - 17) <= 0xD3u )
              {
                v94 = v125;
                v122 = 0;
                LOWORD(v94) = word_4456580[v33 - 1];
                v125 = v94;
                goto LABEL_74;
              }
            }
            else
            {
              v120 = v19;
              v35 = sub_30070B0((__int64)&v144);
              v19 = v120;
              if ( v35 )
              {
                v37 = sub_3009970((__int64)&v144, v18, v120, v36, v10);
                v33 = (unsigned __int16)v144;
                v125 = v37;
                v122 = v38;
                if ( !(_WORD)v144 )
                {
                  if ( !sub_3007100((__int64)&v144) )
                    goto LABEL_33;
LABEL_86:
                  sub_CA17B0(
                    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be drop"
                    "ped, use EVT::getVectorElementCount() instead");
                  if ( !(_WORD)v144 )
                  {
LABEL_33:
                    v39 = sub_3007130((__int64)&v144, v18);
                    goto LABEL_34;
                  }
                  if ( (unsigned __int16)((_WORD)v144 - 176) <= 0x34u )
                    sub_CA17B0(
                      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dr"
                      "opped, use MVT::getVectorElementCount() instead");
LABEL_75:
                  v39 = word_4456340[(unsigned __int16)v144 - 1];
LABEL_34:
                  v18 = v39;
                  v11 = 0;
                  v121 = v39;
                  if ( v39 )
                  {
                    v117 = v20;
                    v40 = 0;
                    do
                    {
                      *(_QWORD *)&v41 = sub_3400D50(a2, v40, &v134, 0);
                      v18 = 158;
                      v31 = v128 | v31 & 0xFFFFFFFF00000000LL;
                      *((_QWORD *)&v110 + 1) = v31;
                      *(_QWORD *)&v110 = v132;
                      v10 = sub_3406EB0(a2, 158, (unsigned int)&v134, v125, v122, v42, v110, v41);
                      v43 = (unsigned int)v149;
                      v11 = v44;
                      v45 = (unsigned int)v149 + 1LL;
                      if ( v45 > HIDWORD(v149) )
                      {
                        v18 = (__int64)v150;
                        v113 = v10;
                        v114 = v11;
                        sub_C8D5F0((__int64)&v148, v150, v45, 0x10u, v10, v11);
                        v43 = (unsigned int)v149;
                        v10 = v113;
                        v11 = v114;
                      }
                      v46 = &v148[2 * v43];
                      ++v40;
                      *v46 = v10;
                      v46[1] = v11;
                      LODWORD(v149) = v149 + 1;
                    }
                    while ( v40 != v121 );
                    v20 = v117;
                  }
                  goto LABEL_27;
                }
LABEL_74:
                if ( (unsigned __int16)(v33 - 176) <= 0x34u )
                  goto LABEL_86;
                goto LABEL_75;
              }
            }
            if ( v124 == 9144 && v20 == 9 )
            {
              v21 = *(_QWORD *)(*(_QWORD *)(v19 + 360) + 96LL);
              v18 = *(_QWORD *)(v21 + 24);
              if ( *(_DWORD *)(v21 + 32) > 0x40u )
                v18 = *(_QWORD *)v18;
              HIWORD(v22) = v119;
              LOWORD(v22) = 7;
              v23 = sub_3400BD0(a2, v18, (unsigned int)&v134, v22, 0, 1, 0);
              v25 = v24;
              v26 = (unsigned int)v149;
              v27 = v23;
              v10 = (unsigned int)v149 + 1LL;
              if ( v10 > HIDWORD(v149) )
              {
                v18 = (__int64)v150;
                sub_C8D5F0((__int64)&v148, v150, (unsigned int)v149 + 1LL, 0x10u, v10, v11);
                v26 = (unsigned int)v149;
              }
              v28 = &v148[2 * v26];
              *v28 = v27;
              v28[1] = v25;
              LODWORD(v149) = v149 + 1;
            }
            else
            {
              v47 = (unsigned int)v149;
              v48 = (unsigned int)v149 + 1LL;
              if ( v48 > HIDWORD(v149) )
              {
                v18 = (__int64)v150;
                sub_C8D5F0((__int64)&v148, v150, v48, 0x10u, v10, v11);
                v47 = (unsigned int)v149;
              }
              v49 = &v148[2 * v47];
              *v49 = v30;
              v49[1] = v31;
              LODWORD(v149) = v149 + 1;
            }
LABEL_27:
            if ( ++v20 == v123 )
            {
              v50 = *(__int16 **)(a1 + 48);
              goto LABEL_44;
            }
            v19 = *(_QWORD *)(a1 + 40);
          }
        }
        v51 = *v14;
        v95 = *((_QWORD *)v14 + 1);
        LOWORD(v136) = v51;
        v137 = v95;
LABEL_77:
        if ( (unsigned __int16)(v51 - 176) > 0x34u )
          goto LABEL_78;
        goto LABEL_99;
      }
    }
LABEL_164:
    BUG();
  }
  v13 = *(_QWORD *)(a1 + 40);
  v50 = *(__int16 **)(a1 + 48);
  v104 = *v50;
  v105 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v13 + 80) + 48LL) + 16LL * *(unsigned int *)(v13 + 88));
  v106 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v13 + 120) + 48LL) + 16LL * *(unsigned int *)(v13 + 128));
  v136 = v139;
  v138 = 32;
  v139[0] = v104;
  v139[1] = v105;
  v139[2] = v106;
  v137 = 3;
  if ( v104 == 149 )
  {
    if ( v106 != 7 || v105 != 58 )
      goto LABEL_164;
    v116 = 2900;
  }
  else if ( v104 <= 0x95u )
  {
    if ( v104 == 58 )
    {
      if ( v106 != 7 || v105 != 7 )
        goto LABEL_164;
      v116 = 2899;
    }
    else
    {
      if ( v104 != 64 )
        goto LABEL_164;
      if ( v105 == 7 )
      {
        if ( v106 != 60 )
          goto LABEL_164;
        v116 = 2904;
      }
      else if ( v105 == 58 )
      {
        if ( v106 != 58 )
          goto LABEL_164;
        v116 = 2905;
      }
      else
      {
        if ( v106 != 7 || v105 != 60 )
          goto LABEL_164;
        v116 = 2906;
      }
    }
  }
  else if ( v104 == 153 )
  {
    if ( v105 == 58 )
    {
      if ( v106 != 64 )
        goto LABEL_164;
      v116 = 2901;
    }
    else if ( v105 == 60 )
    {
      if ( v106 != 60 )
        goto LABEL_164;
      v116 = 2902;
    }
    else
    {
      if ( v106 != 58 || v105 != 64 )
        goto LABEL_164;
      v116 = 2903;
    }
  }
  else
  {
    if ( v104 != 167 || v106 != 13 || v105 != 13 )
      goto LABEL_164;
    v116 = 2898;
  }
  v18 = *(unsigned int *)(a1 + 64);
  v123 = v18;
  if ( (_DWORD)v18 != 2 )
    goto LABEL_18;
LABEL_44:
  v51 = *v50;
  v52 = *((_QWORD *)v50 + 1);
  LOWORD(v136) = v51;
  v137 = v52;
  if ( v51 )
    goto LABEL_77;
  if ( !sub_3007100((__int64)&v136) )
  {
LABEL_46:
    v55 = sub_3007130((__int64)&v136, v18);
LABEL_47:
    v133 = v55;
    v56 = sub_3009970((__int64)&v136, v18, v53, (__int64)v54, v10);
    v55 = v133;
    v57 = v56;
    v59 = v58;
    goto LABEL_48;
  }
LABEL_99:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v136 )
    goto LABEL_46;
  if ( (unsigned __int16)((_WORD)v136 - 176) > 0x34u )
  {
    v96 = (unsigned __int16)v136 - 1;
    v55 = word_4456340[v96];
    goto LABEL_79;
  }
  sub_CA17B0(
    "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::ge"
    "tVectorElementCount() instead");
LABEL_78:
  v54 = word_4456340;
  v53 = (unsigned __int16)v136;
  v96 = (unsigned __int16)v136 - 1;
  v55 = word_4456340[v96];
  if ( !(_WORD)v136 )
    goto LABEL_47;
LABEL_79:
  v59 = 0;
  v57 = word_4456580[v96];
LABEL_48:
  v140 = &v142;
  v141 = 0x500000000LL;
  if ( v55 )
  {
    v129 = a2;
    v60 = 5;
    v61 = v57;
    v62 = 0;
    v63 = v59;
    v64 = 0;
    v65 = v55;
    v66 = v112;
    v67 = &v140;
    while ( 1 )
    {
      LOWORD(v66) = v61;
      if ( v62 + 1 > v60 )
      {
        v126 = v67;
        sub_C8D5F0((__int64)v67, &v142, v62 + 1, 0x10u, v10, v11);
        v62 = (unsigned int)v141;
        v67 = v126;
      }
      v68 = &v140[2 * v62];
      ++v64;
      *v68 = v66;
      v68[1] = v63;
      v62 = (unsigned int)(v141 + 1);
      LODWORD(v141) = v141 + 1;
      if ( v64 == v65 )
        break;
      v60 = HIDWORD(v141);
    }
    v55 = v65;
    a2 = v129;
    v69 = v140;
  }
  else
  {
    v69 = &v142;
  }
  v70 = (__int64)v148;
  v71 = (unsigned int)v149;
  v130 = v55;
  v72 = sub_33E5830(a2, v69);
  v75 = sub_33E66D0(a2, v116, (unsigned int)&v134, v72, v73, v74, v70, v71);
  v76 = v130;
  v144 = &v146;
  v77 = v111;
  v145 = 0x800000000LL;
  if ( v130 )
  {
    v131 = a2;
    *((_QWORD *)&v78 + 1) = 0;
    v79 = 0;
    v80 = v76;
    v81 = v75;
    v82 = v115;
    for ( j = 8; ; j = HIDWORD(v145) )
    {
      v84 = v82 & 0xFFFFFFFF00000000LL | v79;
      v82 = v84;
      if ( *((_QWORD *)&v78 + 1) + 1LL > j )
      {
        sub_C8D5F0((__int64)&v144, &v146, *((_QWORD *)&v78 + 1) + 1LL, 0x10u, 0xFFFFFFFF00000000LL, v77);
        *((_QWORD *)&v78 + 1) = (unsigned int)v145;
      }
      v85 = &v144[2 * *((_QWORD *)&v78 + 1)];
      ++v79;
      *v85 = v81;
      v85[1] = v84;
      *((_QWORD *)&v78 + 1) = (unsigned int)(v145 + 1);
      LODWORD(v145) = v145 + 1;
      if ( v79 == v80 )
        break;
    }
    LODWORD(a2) = v131;
    *(_QWORD *)&v78 = v144;
  }
  else
  {
    *((_QWORD *)&v78 + 1) = 0;
    *(_QWORD *)&v78 = &v146;
  }
  v88 = sub_33FC220(a2, 156, (unsigned int)&v134, (_DWORD)v136, v137, v77, v78);
  v90 = v89;
  v91 = *(unsigned int *)(a3 + 8);
  if ( v91 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v91 + 1, 0x10u, v86, v87);
    v91 = *(unsigned int *)(a3 + 8);
  }
  v92 = (__int64 *)(*(_QWORD *)a3 + 16 * v91);
  *v92 = v88;
  v92[1] = v90;
  v93 = v144;
  ++*(_DWORD *)(a3 + 8);
  if ( v93 != &v146 )
    _libc_free((unsigned __int64)v93);
  if ( v140 != &v142 )
    _libc_free((unsigned __int64)v140);
  if ( v148 != v150 )
    _libc_free((unsigned __int64)v148);
  if ( v134 )
    sub_B91220((__int64)&v134, v134);
}
