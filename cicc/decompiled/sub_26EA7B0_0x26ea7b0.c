// Function: sub_26EA7B0
// Address: 0x26ea7b0
//
void __fastcall sub_26EA7B0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned __int8 v6; // al
  bool v7; // dl
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // rax
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // rax
  _QWORD *v20; // rax
  unsigned __int8 *v21; // rsi
  __int64 v22; // r9
  unsigned __int8 *v23; // r12
  unsigned __int8 *v24; // rax
  int v25; // ecx
  unsigned __int8 *v26; // rdx
  __int64 v27; // r14
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r8
  __int64 v35; // rsi
  int v36; // edi
  __int64 v37; // rax
  unsigned int v38; // edi
  _QWORD *v39; // rbx
  __int64 v40; // r9
  __int64 v41; // r12
  int v42; // eax
  unsigned int v43; // edx
  unsigned __int8 v44; // cl
  int v45; // r12d
  unsigned __int8 *v46; // r14
  unsigned __int8 *v47; // r12
  __int64 v48; // rdx
  unsigned int v49; // esi
  __int64 v50; // rsi
  __int64 v51; // rax
  unsigned __int8 v52; // dl
  __int64 *v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  _QWORD *v56; // rax
  unsigned __int8 **v57; // r12
  __int64 v58; // rbx
  __int64 v59; // r12
  unsigned int v60; // r14d
  int v61; // r15d
  __int64 v62; // rax
  unsigned __int8 *v63; // rsi
  __int64 v64; // rax
  char v65; // cl
  unsigned __int8 v66; // dl
  __int64 *v67; // rax
  __int64 v68; // rax
  unsigned int v69; // eax
  unsigned int v70; // edx
  __int64 v71; // rax
  __int64 v72; // rdx
  int v73; // r15d
  __int64 v74; // rax
  _QWORD *v75; // rax
  unsigned __int8 **v76; // r14
  __int64 v77; // r12
  __int64 v78; // rax
  __int64 *v79; // rax
  __int64 v80; // rsi
  unsigned __int8 *v81; // rsi
  __int64 v82; // rax
  __int64 v83; // rcx
  __int64 v84; // rax
  __int64 *v85; // rdi
  _QWORD *v86; // rax
  __int64 v87; // rsi
  __int64 v88; // rdx
  unsigned __int8 *v89; // rsi
  __int64 v90; // rax
  __int64 v91; // rcx
  __int64 v92; // rax
  __int64 *v93; // rdi
  _QWORD *v94; // rax
  __int64 v95; // rsi
  unsigned __int8 *v96; // rsi
  __int64 v97; // rsi
  unsigned __int8 *v98; // rsi
  unsigned int v99; // eax
  unsigned __int64 v100; // r8
  unsigned __int64 v101; // r13
  __int64 v102; // r12
  __int64 v103; // rdi
  __int64 v104; // rax
  __int64 v105; // rdx
  __int64 v106; // [rsp-10h] [rbp-1F0h]
  __int64 v107; // [rsp+0h] [rbp-1E0h]
  const char *v109; // [rsp+10h] [rbp-1D0h]
  __int64 v111; // [rsp+20h] [rbp-1C0h]
  __int64 *v112; // [rsp+28h] [rbp-1B8h]
  __int64 v113; // [rsp+30h] [rbp-1B0h]
  __int64 v114; // [rsp+38h] [rbp-1A8h]
  __int64 v115; // [rsp+40h] [rbp-1A0h]
  int v116; // [rsp+60h] [rbp-180h]
  unsigned int v117; // [rsp+64h] [rbp-17Ch]
  unsigned int v118; // [rsp+68h] [rbp-178h]
  __int64 i; // [rsp+78h] [rbp-168h]
  __int64 v120; // [rsp+98h] [rbp-148h] BYREF
  __int64 v121[4]; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v122[4]; // [rsp+C0h] [rbp-120h] BYREF
  __int16 v123; // [rsp+E0h] [rbp-100h]
  unsigned __int8 *v124[4]; // [rsp+F0h] [rbp-F0h] BYREF
  __int16 v125; // [rsp+110h] [rbp-D0h]
  unsigned __int8 *v126; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v127; // [rsp+128h] [rbp-B8h]
  _BYTE v128[32]; // [rsp+130h] [rbp-B0h] BYREF
  __int64 v129; // [rsp+150h] [rbp-90h]
  __int64 v130; // [rsp+158h] [rbp-88h]
  __int64 v131; // [rsp+160h] [rbp-80h]
  _QWORD *v132; // [rsp+168h] [rbp-78h]
  void **v133; // [rsp+170h] [rbp-70h]
  void **v134; // [rsp+178h] [rbp-68h]
  __int64 v135; // [rsp+180h] [rbp-60h]
  int v136; // [rsp+188h] [rbp-58h]
  __int16 v137; // [rsp+18Ch] [rbp-54h]
  char v138; // [rsp+18Eh] [rbp-52h]
  __int64 v139; // [rsp+190h] [rbp-50h]
  __int64 v140; // [rsp+198h] [rbp-48h]
  void *v141; // [rsp+1A0h] [rbp-40h] BYREF
  void *v142; // [rsp+1A8h] [rbp-38h] BYREF

  v112 = *(__int64 **)(a2 + 40);
  v120 = sub_B2BE50(a2);
  v109 = sub_BD5D20(a2);
  v111 = v2;
  v3 = sub_B92180(a2);
  if ( v3 )
  {
    v4 = v3;
    v5 = v3 - 16;
    v6 = *(_BYTE *)(v3 - 16);
    v7 = (v6 & 2) != 0;
    if ( (v6 & 2) != 0 )
      v8 = *(_QWORD *)(v4 - 32);
    else
      v8 = v5 - 8LL * ((v6 >> 2) & 0xF);
    v9 = *(_QWORD *)(v8 + 24);
    if ( v9 )
    {
      v10 = sub_B91420(v9);
      v111 = v11;
      v109 = (const char *)v10;
      if ( v11 )
        goto LABEL_6;
      v6 = *(_BYTE *)(v4 - 16);
      v7 = (v6 & 2) != 0;
    }
    if ( v7 )
      v102 = *(_QWORD *)(v4 - 32);
    else
      v102 = v5 - 8LL * ((v6 >> 2) & 0xF);
    v103 = *(_QWORD *)(v102 + 16);
    if ( v103 )
    {
      v104 = sub_B91420(v103);
      v111 = v105;
      v103 = v104;
    }
    else
    {
      v111 = 0;
    }
    v109 = (const char *)v103;
  }
LABEL_6:
  v113 = sub_B2F650((__int64)v109, v111);
  for ( i = a1[8]; i; i = *(_QWORD *)i )
  {
    v12 = *(_QWORD *)(i + 8);
    v118 = *(_DWORD *)(i + 16);
    v13 = v12 + 48;
    v14 = sub_AA5190(v12);
    v15 = *(_QWORD *)(v12 + 48);
    v16 = v14;
    v17 = v14 - 24;
    if ( v16 )
      v16 = v17;
    v18 = v15 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_10:
    if ( v13 != v18 )
    {
LABEL_11:
      if ( !v18 )
        BUG();
      v19 = 0;
      if ( (unsigned int)*(unsigned __int8 *)(v18 - 24) - 30 < 0xB )
        v19 = v18 - 24;
      goto LABEL_14;
    }
    while ( 1 )
    {
      v19 = 0;
LABEL_14:
      if ( v16 == v19 )
        break;
      if ( *(_BYTE *)v16 != 84
        && (*(_BYTE *)v16 != 85
         || (v71 = *(_QWORD *)(v16 - 32)) == 0
         || *(_BYTE *)v71
         || *(_QWORD *)(v71 + 24) != *(_QWORD *)(v16 + 80)
         || (*(_BYTE *)(v71 + 33) & 0x20) == 0
         || (unsigned int)(*(_DWORD *)(v71 + 36) - 68) > 3)
        && !sub_B46A10(v16)
        && *(_QWORD *)(v16 + 48) )
      {
        break;
      }
      v72 = *(_QWORD *)(v16 + 32);
      if ( v72 == *(_QWORD *)(v16 + 40) + 48LL || !v72 )
      {
        v16 = 0;
        goto LABEL_10;
      }
      v16 = v72 - 24;
      if ( v13 != v18 )
        goto LABEL_11;
    }
    v20 = (_QWORD *)sub_BD5C60(v16);
    v135 = 0;
    v132 = v20;
    v133 = &v141;
    v134 = &v142;
    LOWORD(v131) = 0;
    v129 = 0;
    v130 = 0;
    v141 = &unk_49DA100;
    v126 = v128;
    v127 = 0x200000000LL;
    v136 = 0;
    v137 = 512;
    v138 = 7;
    v139 = 0;
    v140 = 0;
    v142 = &unk_49DA0B0;
    v129 = *(_QWORD *)(v16 + 40);
    v130 = v16 + 24;
    v21 = *(unsigned __int8 **)sub_B46C60(v16);
    v124[0] = v21;
    if ( v21 && (sub_B96E90((__int64)v124, (__int64)v21, 1), (v23 = v124[0]) != 0) )
    {
      v24 = v126;
      v25 = v127;
      v26 = &v126[16 * (unsigned int)v127];
      if ( v126 != v26 )
      {
        while ( *(_DWORD *)v24 )
        {
          v24 += 16;
          if ( v26 == v24 )
            goto LABEL_118;
        }
        *((unsigned __int8 **)v24 + 1) = v124[0];
LABEL_26:
        sub_B91220((__int64)v124, (__int64)v23);
        goto LABEL_27;
      }
LABEL_118:
      if ( (unsigned int)v127 >= (unsigned __int64)HIDWORD(v127) )
      {
        v100 = (unsigned int)v127 + 1LL;
        v101 = v107 & 0xFFFFFFFF00000000LL;
        v107 &= 0xFFFFFFFF00000000LL;
        if ( HIDWORD(v127) < v100 )
        {
          sub_C8D5F0((__int64)&v126, v128, v100, 0x10u, v100, v22);
          v26 = &v126[16 * (unsigned int)v127];
        }
        *(_QWORD *)v26 = v101;
        *((_QWORD *)v26 + 1) = v23;
        v23 = v124[0];
        LODWORD(v127) = v127 + 1;
      }
      else
      {
        if ( v26 )
        {
          *(_DWORD *)v26 = 0;
          *((_QWORD *)v26 + 1) = v23;
          v25 = v127;
          v23 = v124[0];
        }
        LODWORD(v127) = v25 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v126, 0);
      v23 = v124[0];
    }
    if ( v23 )
      goto LABEL_26;
LABEL_27:
    v27 = 0;
    v28 = sub_B6E160(v112, 0x123u, 0, 0);
    v29 = sub_BCB2E0(v132);
    v121[0] = sub_ACD640(v29, v113, 0);
    v30 = sub_BCB2E0(v132);
    v121[1] = sub_ACD640(v30, v118, 0);
    v31 = sub_BCB2D0(v132);
    v121[2] = sub_ACD640(v31, 0, 0);
    v32 = sub_BCB2E0(v132);
    v33 = sub_ACD640(v32, -1, 0);
    v123 = 257;
    v121[3] = v33;
    if ( v28 )
      v27 = *(_QWORD *)(v28 + 24);
    v125 = 257;
    v34 = v139 + 56 * v140;
    if ( v34 == v139 )
    {
      v116 = 5;
      v38 = 5;
    }
    else
    {
      v35 = v139;
      v36 = 0;
      do
      {
        v37 = *(_QWORD *)(v35 + 40) - *(_QWORD *)(v35 + 32);
        v35 += 56;
        v36 += v37 >> 3;
      }
      while ( v34 != v35 );
      v38 = v36 + 5;
      v116 = v38 & 0x7FFFFFF;
    }
    v114 = v139;
    v115 = v140;
    LOBYTE(v23) = 16 * (_DWORD)v140 != 0;
    v39 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v140) << 32) | v38);
    if ( v39 )
    {
      v117 = v117 & 0xE0000000 | ((_DWORD)v23 << 28) | v116;
      sub_B44260((__int64)v39, **(_QWORD **)(v27 + 16), 56, v117, 0, 0);
      v39[9] = 0;
      sub_B4A290((__int64)v39, v27, v28, v121, 4, (__int64)v124, v114, v115);
      v40 = v106;
    }
    if ( (_BYTE)v137 )
    {
      v79 = (__int64 *)sub_BD5C60((__int64)v39);
      v39[9] = sub_A7A090(v39 + 9, v79, -1, 72);
    }
    if ( *(_BYTE *)v39 > 0x1Cu )
    {
      switch ( *(_BYTE *)v39 )
      {
        case ')':
        case '+':
        case '-':
        case '/':
        case '2':
        case '5':
        case 'J':
        case 'K':
        case 'S':
          goto LABEL_44;
        case 'T':
        case 'U':
        case 'V':
          v41 = v39[1];
          v42 = *(unsigned __int8 *)(v41 + 8);
          v43 = v42 - 17;
          v44 = *(_BYTE *)(v41 + 8);
          if ( (unsigned int)(v42 - 17) <= 1 )
            v44 = *(_BYTE *)(**(_QWORD **)(v41 + 16) + 8LL);
          if ( v44 <= 3u || v44 == 5 || (v44 & 0xFD) == 4 )
            goto LABEL_44;
          if ( (_BYTE)v42 == 15 )
          {
            if ( (*(_BYTE *)(v41 + 9) & 4) == 0 || !sub_BCB420(v39[1]) )
              break;
            v41 = **(_QWORD **)(v41 + 16);
            v42 = *(unsigned __int8 *)(v41 + 8);
            v43 = v42 - 17;
          }
          else if ( (_BYTE)v42 == 16 )
          {
            do
            {
              v41 = *(_QWORD *)(v41 + 24);
              LOBYTE(v42) = *(_BYTE *)(v41 + 8);
            }
            while ( (_BYTE)v42 == 16 );
            v43 = (unsigned __int8)v42 - 17;
          }
          if ( v43 <= 1 )
            LOBYTE(v42) = *(_BYTE *)(**(_QWORD **)(v41 + 16) + 8LL);
          if ( (unsigned __int8)v42 <= 3u || (_BYTE)v42 == 5 || (v42 & 0xFD) == 4 )
          {
LABEL_44:
            v45 = v136;
            if ( v135 )
              sub_B99FD0((__int64)v39, 3u, v135);
            sub_B45150((__int64)v39, v45);
          }
          break;
        default:
          break;
      }
    }
    (*((void (__fastcall **)(void **, _QWORD *, __int64 *, __int64, __int64, __int64))*v134 + 2))(
      v134,
      v39,
      v122,
      v130,
      v131,
      v40);
    v46 = v126;
    v47 = &v126[16 * (unsigned int)v127];
    if ( v126 != v47 )
    {
      do
      {
        v48 = *((_QWORD *)v46 + 1);
        v49 = *(_DWORD *)v46;
        v46 += 16;
        sub_B99FD0((__int64)v39, v49, v48);
      }
      while ( v47 != v46 );
    }
    v50 = v39[6];
    if ( v50 )
    {
      v122[0] = v39[6];
    }
    else
    {
      v90 = sub_B92180(a2);
      v91 = v90;
      if ( v90 )
      {
        v92 = *(_QWORD *)(v90 + 8);
        v93 = (__int64 *)(v92 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v92 & 4) != 0 )
          v93 = (__int64 *)*v93;
        v94 = sub_B01860(v93, 0, 0, v91, 0, 0, 0, 1);
        sub_B10CB0(v124, (__int64)v94);
        v95 = v39[6];
        if ( v95 )
          sub_B91220((__int64)(v39 + 6), v95);
        v96 = v124[0];
        v39[6] = v124[0];
        if ( !v96 )
          goto LABEL_65;
        sub_B976B0((__int64)v124, v96, (__int64)(v39 + 6));
      }
      v50 = v39[6];
      v122[0] = v50;
      if ( !v50 )
        goto LABEL_65;
    }
    sub_B96E90((__int64)v122, v50, 1);
    if ( v122[0] )
    {
      v51 = sub_B10CD0((__int64)v122);
      v52 = *(_BYTE *)(v51 - 16);
      if ( (v52 & 2) != 0 )
        v53 = *(__int64 **)(v51 - 32);
      else
        v53 = (__int64 *)(v51 - 16 - 8LL * ((v52 >> 2) & 0xF));
      v54 = *v53;
      if ( *(_BYTE *)v54 == 20 && *(_DWORD *)(v54 + 4) )
      {
        v55 = sub_B10CD0((__int64)v122);
        v56 = sub_26BDBC0(v55, 0);
        sub_B10CB0(v124, (__int64)v56);
        if ( v122[0] )
          sub_B91220((__int64)v122, v122[0]);
        v122[0] = (__int64)v124[0];
        if ( v124[0] )
        {
          v57 = (unsigned __int8 **)(v39 + 6);
          sub_B976B0((__int64)v124, v124[0], (__int64)v122);
          v124[0] = (unsigned __int8 *)v122[0];
          if ( v122[0] )
          {
            sub_B96E90((__int64)v124, v122[0], 1);
            if ( v57 == v124 )
            {
              if ( v124[0] )
                sub_B91220((__int64)v124, (__int64)v124[0]);
              goto LABEL_63;
            }
            v97 = v39[6];
            if ( !v97 )
            {
LABEL_165:
              v98 = v124[0];
              v39[6] = v124[0];
              if ( v98 )
                sub_B976B0((__int64)v124, v98, (__int64)v57);
              goto LABEL_63;
            }
LABEL_164:
            sub_B91220((__int64)v57, v97);
            goto LABEL_165;
          }
        }
        else
        {
          v57 = (unsigned __int8 **)(v39 + 6);
        }
        if ( v57 != v124 )
        {
          v97 = v39[6];
          if ( v97 )
            goto LABEL_164;
          v39[6] = v124[0];
        }
      }
LABEL_63:
      if ( v122[0] )
        sub_B91220((__int64)v122, v122[0]);
    }
LABEL_65:
    nullsub_61();
    v141 = &unk_49DA100;
    nullsub_63();
    if ( v126 != v128 )
      _libc_free((unsigned __int64)v126);
  }
  v58 = a1[15];
  if ( v58 )
  {
    while ( 1 )
    {
      v59 = *(_QWORD *)(v58 + 8);
      v60 = *(_DWORD *)(v58 + 16);
      v61 = 1;
      v62 = *(_QWORD *)(v59 - 32);
      if ( v62 && !*(_BYTE *)v62 )
        v61 = (*(_QWORD *)(v62 + 24) == *(_QWORD *)(v59 + 80)) + 1;
      v63 = *(unsigned __int8 **)(v59 + 48);
      if ( v63 )
        break;
      v82 = sub_B92180(a2);
      v83 = v82;
      if ( v82 )
      {
        v84 = *(_QWORD *)(v82 + 8);
        v85 = (__int64 *)(v84 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v84 & 4) != 0 )
          v85 = (__int64 *)*v85;
        v86 = sub_B01860(v85, 0, 0, v83, 0, 0, 0, 1);
        sub_B10CB0(&v126, (__int64)v86);
        v87 = *(_QWORD *)(v59 + 48);
        v88 = v59 + 48;
        if ( v87 )
        {
          sub_B91220(v59 + 48, v87);
          v88 = v59 + 48;
        }
        v89 = v126;
        *(_QWORD *)(v59 + 48) = v126;
        if ( !v89 )
          goto LABEL_116;
        sub_B976B0((__int64)&v126, v89, v88);
      }
      v63 = *(unsigned __int8 **)(v59 + 48);
      v124[0] = v63;
      if ( v63 )
      {
LABEL_74:
        sub_B96E90((__int64)v124, (__int64)v63, 1);
        if ( v124[0] )
        {
          v64 = sub_B10CD0((__int64)v124);
          v65 = qword_4F813A8[8];
          v66 = *(_BYTE *)(v64 - 16);
          if ( (v66 & 2) != 0 )
            v67 = *(__int64 **)(v64 - 32);
          else
            v67 = (__int64 *)(v64 - 16 - 8LL * ((v66 >> 2) & 0xF));
          v68 = *v67;
          if ( *(_BYTE *)v68 == 20 )
          {
            v69 = *(_DWORD *)(v68 + 4);
            if ( (v69 & 7) == 7 && (v69 & 0xFFFFFFF8) != 0 )
            {
              if ( (v69 & 0x10000000) != 0 )
              {
                v65 = 1;
                v70 = HIWORD(v69) & 7;
              }
              else
              {
                v99 = v69 >> 3;
                v70 = (unsigned __int16)v99;
                v65 = (v99 & 0xFFF8) == 0;
              }
              goto LABEL_104;
            }
            if ( LOBYTE(qword_4F813A8[8]) )
            {
              v70 = (unsigned __int8)v69;
              v65 = (v69 & 0xF8) == 0;
              goto LABEL_104;
            }
            if ( (v69 & 1) != 0 )
            {
              v73 = (v61 << 26) | (8 * v60) | 0x3200007;
              if ( v60 <= 0x1FFF )
                v73 |= 0x10000000u;
              goto LABEL_107;
            }
            v70 = (v69 >> 1) & 0x1F;
            if ( ((v69 >> 1) & 0x20) != 0 )
            {
              v70 |= (v69 >> 2) & 0xFE0;
              v65 = v70 <= 7;
LABEL_104:
              v73 = (v61 << 26) | (8 * v60) | 0x3200007;
              if ( v60 <= 0x1FFF && v65 )
                v73 |= (v70 << 16) | 0x10000000;
LABEL_107:
              v74 = sub_B10CD0((__int64)v124);
              v75 = sub_26BDBC0(v74, v73);
              sub_B10CB0(&v126, (__int64)v75);
              if ( v124[0] )
                sub_B91220((__int64)v124, (__int64)v124[0]);
              v124[0] = v126;
              if ( v126 )
              {
                v76 = (unsigned __int8 **)(v59 + 48);
                sub_B976B0((__int64)&v126, v126, (__int64)v124);
                v126 = v124[0];
                if ( v124[0] )
                {
                  sub_B96E90((__int64)&v126, (__int64)v124[0], 1);
                  if ( v76 == &v126 )
                  {
                    if ( v126 )
                      sub_B91220((__int64)&v126, (__int64)v126);
                    goto LABEL_114;
                  }
                  v80 = *(_QWORD *)(v59 + 48);
                  if ( !v80 )
                  {
LABEL_131:
                    v81 = v126;
                    *(_QWORD *)(v59 + 48) = v126;
                    if ( v81 )
                      sub_B976B0((__int64)&v126, v81, (__int64)v76);
                    goto LABEL_114;
                  }
LABEL_130:
                  sub_B91220((__int64)v76, v80);
                  goto LABEL_131;
                }
              }
              else
              {
                v76 = (unsigned __int8 **)(v59 + 48);
              }
              if ( v76 != &v126 )
              {
                v80 = *(_QWORD *)(v59 + 48);
                if ( v80 )
                  goto LABEL_130;
                *(_QWORD *)(v59 + 48) = v126;
              }
LABEL_114:
              if ( v124[0] )
                sub_B91220((__int64)v124, (__int64)v124[0]);
              goto LABEL_116;
            }
          }
          else
          {
            v70 = 0;
            if ( LOBYTE(qword_4F813A8[8]) )
              goto LABEL_104;
          }
          v65 = v70 <= 7;
          goto LABEL_104;
        }
LABEL_116:
        v58 = *(_QWORD *)v58;
        if ( !v58 )
          goto LABEL_117;
      }
      else
      {
        v58 = *(_QWORD *)v58;
        if ( !v58 )
          goto LABEL_117;
      }
    }
    v124[0] = *(unsigned __int8 **)(v59 + 48);
    goto LABEL_74;
  }
LABEL_117:
  v77 = sub_B8D270(&v120, v113, a1[5], (__int64)v109, v111);
  v78 = sub_BA8DC0((__int64)v112, (__int64)"llvm.pseudo_probe_desc", 22);
  sub_B979A0(v78, v77);
}
