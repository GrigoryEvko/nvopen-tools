// Function: sub_2F3D720
// Address: 0x2f3d720
//
__int64 __fastcall sub_2F3D720(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v7; // r14
  __int64 v8; // rsi
  _BYTE *v9; // rdi
  __int64 v10; // rdi
  __int64 (*v11)(); // rax
  __int64 v12; // rdi
  __int64 (*v13)(); // rax
  __int64 v14; // r14
  __int64 v15; // rbx
  _BYTE *v16; // rdi
  __int64 v17; // rdi
  __int64 (*v18)(); // rax
  __int64 v19; // rdi
  __int64 (*v20)(); // rax
  unsigned int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r14
  __int64 v25; // rbx
  _BYTE *v26; // rdi
  __int64 v27; // rdi
  __int64 (*v28)(); // rax
  __int64 v29; // rdi
  __int64 (*v30)(); // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r13
  __int64 v35; // r14
  _BYTE *v36; // rbx
  __int64 *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  unsigned __int64 v40; // r13
  __int64 v41; // rdx
  __int64 v42; // r13
  unsigned __int64 v43; // rbx
  signed __int64 v44; // r14
  __int64 **v45; // rax
  __int64 *v46; // rax
  __int64 *v47; // rdx
  __int64 *v48; // rbx
  __int64 v49; // rsi
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // r13
  unsigned __int8 *v53; // rax
  int v54; // ecx
  unsigned __int64 *v55; // rdx
  __int64 v56; // r13
  __int64 v57; // r14
  __int64 *v58; // rax
  unsigned __int64 v59; // rax
  __int64 v60; // rdx
  _QWORD *v61; // rdx
  _QWORD *v62; // rax
  __int64 v63; // rbx
  __int64 v64; // rdx
  unsigned __int8 *v65; // r13
  unsigned __int64 v66; // rdx
  __int64 v67; // r14
  __int64 v68; // rax
  unsigned __int8 *v69; // r14
  __int64 (__fastcall *v70)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v71; // r9
  unsigned __int64 v72; // rax
  __int64 v73; // rdi
  __int64 v74; // rdx
  int v75; // esi
  __int64 v76; // rax
  __int64 v77; // rsi
  int v78; // r14d
  _QWORD *v79; // rax
  __int64 v80; // rbx
  unsigned __int64 v81; // r14
  unsigned __int8 *v82; // r13
  __int64 v83; // rdx
  unsigned int v84; // esi
  __int64 v85; // r13
  __int64 v86; // rax
  __int64 *v87; // rsi
  unsigned __int64 v88; // r14
  __int64 v89; // rbx
  __int64 v90; // rdx
  unsigned int v91; // esi
  __int64 *v92; // rax
  int v93; // r13d
  unsigned __int64 v94; // rsi
  unsigned __int64 v95; // rbx
  __int64 v96; // [rsp+0h] [rbp-1F0h]
  __int64 v97; // [rsp+8h] [rbp-1E8h]
  __int64 v98; // [rsp+20h] [rbp-1D0h]
  __int64 v99; // [rsp+30h] [rbp-1C0h]
  __int64 v100; // [rsp+30h] [rbp-1C0h]
  char v101; // [rsp+38h] [rbp-1B8h]
  __int64 v102; // [rsp+40h] [rbp-1B0h]
  __int64 v103; // [rsp+40h] [rbp-1B0h]
  __int64 v104; // [rsp+48h] [rbp-1A8h]
  __int64 v105; // [rsp+48h] [rbp-1A8h]
  __int64 v106; // [rsp+48h] [rbp-1A8h]
  __int64 v107; // [rsp+48h] [rbp-1A8h]
  __int64 v108; // [rsp+48h] [rbp-1A8h]
  __int64 v109; // [rsp+48h] [rbp-1A8h]
  __int64 v110; // [rsp+58h] [rbp-198h]
  __int64 *v111; // [rsp+60h] [rbp-190h]
  _QWORD *v112; // [rsp+60h] [rbp-190h]
  char v113; // [rsp+60h] [rbp-190h]
  unsigned __int64 v114; // [rsp+68h] [rbp-188h]
  __int64 **v115; // [rsp+68h] [rbp-188h]
  unsigned int v116; // [rsp+70h] [rbp-180h]
  int v117; // [rsp+74h] [rbp-17Ch]
  unsigned __int64 v119; // [rsp+80h] [rbp-170h] BYREF
  __int64 v120; // [rsp+88h] [rbp-168h]
  __int64 v121; // [rsp+90h] [rbp-160h]
  __int64 v122[4]; // [rsp+A0h] [rbp-150h] BYREF
  __int16 v123; // [rsp+C0h] [rbp-130h]
  const char *v124; // [rsp+D0h] [rbp-120h] BYREF
  unsigned __int64 v125; // [rsp+D8h] [rbp-118h]
  __int64 v126; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v127; // [rsp+E8h] [rbp-108h] BYREF
  __int64 v128; // [rsp+F0h] [rbp-100h]
  unsigned __int8 *v129; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v130; // [rsp+138h] [rbp-B8h]
  _BYTE v131[32]; // [rsp+140h] [rbp-B0h] BYREF
  __int64 v132; // [rsp+160h] [rbp-90h]
  __int64 v133; // [rsp+168h] [rbp-88h]
  __int64 v134; // [rsp+170h] [rbp-80h]
  __int64 *v135; // [rsp+178h] [rbp-78h]
  void **v136; // [rsp+180h] [rbp-70h]
  void **v137; // [rsp+188h] [rbp-68h]
  __int64 v138; // [rsp+190h] [rbp-60h]
  int v139; // [rsp+198h] [rbp-58h]
  __int16 v140; // [rsp+19Ch] [rbp-54h]
  char v141; // [rsp+19Eh] [rbp-52h]
  __int64 v142; // [rsp+1A0h] [rbp-50h]
  __int64 v143; // [rsp+1A8h] [rbp-48h]
  void *v144; // [rsp+1B0h] [rbp-40h] BYREF
  void *v145; // [rsp+1B8h] [rbp-38h] BYREF

  v2 = 0;
  v117 = *(_DWORD *)(a2 + 36);
  v3 = *(_QWORD *)(a2 + 16);
  if ( v3 )
  {
    while ( 1 )
    {
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 8);
      v5 = *(_QWORD *)(v4 + 24);
      if ( v117 != 154 )
      {
        switch ( v117 )
        {
          case 238:
            v24 = sub_B43CB0(v5);
            v25 = ((__int64 (__fastcall *)(__int64, __int64))a1[1])(a1[2], v24);
            v26 = *(_BYTE **)(v5 + 32 * (2LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)));
            if ( *v26 == 17 && !(unsigned __int8)sub_2F3BFF0((__int64)v26, v25) )
              goto LABEL_6;
            if ( *((_BYTE *)a1 + 40) )
            {
              v27 = *a1;
              if ( !*a1 )
                goto LABEL_6;
              v28 = *(__int64 (**)())(*(_QWORD *)v27 + 16LL);
              if ( v28 == sub_23CE270 )
                BUG();
              v29 = ((__int64 (__fastcall *)(__int64, __int64))v28)(v27, v24);
              v30 = *(__int64 (**)())(*(_QWORD *)v29 + 144LL);
              if ( v30 == sub_2C8F680 )
                BUG();
              if ( *(_QWORD *)(((__int64 (__fastcall *)(__int64))v30)(v29) + 529032) )
                goto LABEL_6;
            }
            v23 = v25;
            goto LABEL_27;
          case 240:
            if ( **(_BYTE **)(v5 + 32 * (2LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF))) == 17 )
              goto LABEL_6;
            v22 = sub_B43CB0(v5);
            v23 = ((__int64 (__fastcall *)(__int64, __int64))a1[1])(a1[2], v22);
LABEL_27:
            sub_319DBE0(v5, v23, 0);
            sub_B43D60((_QWORD *)v5);
            goto LABEL_28;
          case 241:
            v14 = sub_B43CB0(v5);
            v15 = ((__int64 (__fastcall *)(__int64, __int64))a1[1])(a1[2], v14);
            v16 = *(_BYTE **)(v5 + 32 * (2LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)));
            if ( *v16 != 17 || (unsigned __int8)sub_2F3BFF0((__int64)v16, v15) )
            {
              if ( !*((_BYTE *)a1 + 40) )
                goto LABEL_23;
              v17 = *a1;
              if ( *a1 )
              {
                v18 = *(__int64 (**)())(*(_QWORD *)v17 + 16LL);
                if ( v18 == sub_23CE270 )
                  BUG();
                v19 = ((__int64 (__fastcall *)(__int64, __int64))v18)(v17, v14);
                v20 = *(__int64 (**)())(*(_QWORD *)v19 + 144LL);
                if ( v20 == sub_2C8F680 )
                  BUG();
                if ( !*(_QWORD *)(((__int64 (__fastcall *)(__int64))v20)(v19) + 529040) )
                {
LABEL_23:
                  v21 = sub_31A1B30(v5, v15);
                  if ( (_BYTE)v21 )
                  {
                    v2 = v21;
                    sub_B43D60((_QWORD *)v5);
                  }
                }
              }
            }
            goto LABEL_6;
          case 243:
            v7 = sub_B43CB0(v5);
            v8 = ((__int64 (__fastcall *)(__int64, __int64))a1[1])(a1[2], v7);
            v9 = *(_BYTE **)(v5 + 32 * (2LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)));
            if ( *v9 == 17 && !(unsigned __int8)sub_2F3BFF0((__int64)v9, v8) )
              goto LABEL_6;
            if ( *((_BYTE *)a1 + 40) )
            {
              v10 = *a1;
              if ( !*a1 )
                goto LABEL_6;
              v11 = *(__int64 (**)())(*(_QWORD *)v10 + 16LL);
              if ( v11 == sub_23CE270 )
                BUG();
              v12 = ((__int64 (__fastcall *)(__int64, __int64))v11)(v10, v7);
              v13 = *(__int64 (**)())(*(_QWORD *)v12 + 144LL);
              if ( v13 == sub_2C8F680 )
                BUG();
              if ( *(_QWORD *)(((__int64 (__fastcall *)(__int64))v13)(v12) + 529048) )
                goto LABEL_6;
            }
            goto LABEL_5;
          case 245:
            if ( **(_BYTE **)(v5 + 32 * (2LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF))) != 17 )
            {
LABEL_5:
              v2 = 1;
              sub_31A26A0(v5);
              sub_B43D60((_QWORD *)v5);
            }
            goto LABEL_6;
          default:
            BUG();
        }
      }
      v31 = sub_B43CB0(v5);
      v111 = (__int64 *)((__int64 (__fastcall *)(__int64, __int64))a1[3])(a1[4], v31);
      v32 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
      v33 = *(_QWORD *)(*(_QWORD *)(v5 - 32 * v32) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v33 + 8) - 17 <= 1 )
        v33 = **(_QWORD **)(v33 + 16);
      if ( !(*(_DWORD *)(v33 + 8) >> 8) )
      {
        v34 = *(_QWORD *)(v5 + 32 * (1 - v32));
        v104 = v34;
        v35 = *(_QWORD *)(v34 + 8);
        v36 = (_BYTE *)sub_B43CC0(v5);
        v37 = (__int64 *)sub_B43CA0(v5);
        if ( sub_11C99B0(v37, v111, 0x16Bu) && *(_BYTE *)v34 <= 0x15u && *(_BYTE *)v34 != 5 )
        {
          v129 = (unsigned __int8 *)sub_9208B0((__int64)v36, v35);
          v130 = v38;
          v114 = sub_CA1930(&v129);
          v129 = (unsigned __int8 *)sub_9208B0((__int64)v36, v35);
          v130 = v39;
          v40 = (unsigned __int64)(v129 + 7) & 0xFFFFFFFFFFFFFFF8LL;
          v101 = v39;
          v129 = (unsigned __int8 *)sub_9208B0((__int64)v36, v35);
          v130 = v41;
          if ( v129 == (unsigned __int8 *)v40 && (_BYTE)v130 == v101 )
          {
            if ( v114 )
            {
              v42 = v114 & (v114 - 1);
              if ( !v42 && !*v36 )
              {
                v43 = v114 >> 3;
                if ( v114 <= 0x87 )
                {
                  if ( v43 == 16 )
                    break;
                  v44 = 0x10 / v43;
                  v45 = (__int64 **)sub_BCD420(*(__int64 **)(v104 + 8), 0x10 / v43);
                  if ( v43 > 0x10 )
                  {
                    v104 = sub_AD1300(v45, 0, 0);
                  }
                  else
                  {
                    v115 = v45;
                    v46 = (__int64 *)sub_22077B0(8 * (0x10 / v43));
                    v47 = &v46[v44];
                    v48 = v46;
                    if ( v46 != &v46[v44] )
                    {
                      do
                        *v46++ = v104;
                      while ( v46 != v47 );
                      v42 = (v44 * 8) >> 3;
                    }
                    v104 = sub_AD1300(v115, v48, v42);
                    if ( v48 )
                      j_j___libc_free_0((unsigned __int64)v48);
                  }
                  if ( v104 )
                    break;
                }
              }
            }
          }
        }
      }
      sub_31A2750(v5);
      sub_B43D60((_QWORD *)v5);
LABEL_28:
      v2 = 1;
LABEL_6:
      if ( !v3 )
        return v2;
    }
    v135 = (__int64 *)sub_BD5C60(v5);
    v136 = &v144;
    v137 = &v145;
    v132 = 0;
    v129 = v131;
    v144 = &unk_49DA100;
    v133 = 0;
    v130 = 0x200000000LL;
    v138 = 0;
    v139 = 0;
    v140 = 512;
    v141 = 7;
    v142 = 0;
    v143 = 0;
    LOWORD(v134) = 0;
    v145 = &unk_49DA0B0;
    v132 = *(_QWORD *)(v5 + 40);
    v133 = v5 + 24;
    v49 = *(_QWORD *)sub_B46C60(v5);
    v124 = (const char *)v49;
    if ( v49 && (sub_B96E90((__int64)&v124, v49, 1), (v52 = (__int64)v124) != 0) )
    {
      v53 = v129;
      v54 = v130;
      v55 = (unsigned __int64 *)&v129[16 * (unsigned int)v130];
      if ( v129 != (unsigned __int8 *)v55 )
      {
        while ( 1 )
        {
          v51 = *(unsigned int *)v53;
          if ( !(_DWORD)v51 )
            break;
          v53 += 16;
          if ( v55 == (unsigned __int64 *)v53 )
            goto LABEL_93;
        }
        *((_QWORD *)v53 + 1) = v124;
        goto LABEL_65;
      }
LABEL_93:
      if ( (unsigned int)v130 >= (unsigned __int64)HIDWORD(v130) )
      {
        v94 = (unsigned int)v130 + 1LL;
        v95 = v96 & 0xFFFFFFFF00000000LL;
        v96 &= 0xFFFFFFFF00000000LL;
        if ( HIDWORD(v130) < v94 )
        {
          sub_C8D5F0((__int64)&v129, v131, v94, 0x10u, v50, v51);
          v55 = (unsigned __int64 *)&v129[16 * (unsigned int)v130];
        }
        *v55 = v95;
        v55[1] = v52;
        v52 = (__int64)v124;
        LODWORD(v130) = v130 + 1;
      }
      else
      {
        if ( v55 )
        {
          *(_DWORD *)v55 = 0;
          v55[1] = v52;
          v54 = v130;
          v52 = (__int64)v124;
        }
        LODWORD(v130) = v54 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v129, 0);
      v52 = (__int64)v124;
    }
    if ( !v52 )
    {
LABEL_66:
      v56 = sub_B43CA0(v5);
      v57 = sub_B43CC0(v5);
      v98 = *(_QWORD *)(*(_QWORD *)(v5 + 32 * (2LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF))) + 8LL);
      v99 = sub_BCE3C0(v135, 0);
      v102 = *(_QWORD *)(*(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)) + 8LL);
      v58 = (__int64 *)sub_BCB120(v135);
      v124 = (const char *)&v126;
      v126 = v102;
      v127 = v99;
      v128 = v98;
      v125 = 0x300000003LL;
      v59 = sub_BCF480(v58, &v126, 3, 0);
      v103 = sub_11C96C0(v56, v111, 0x16Bu, v59, 0);
      v100 = v60;
      if ( v124 != (const char *)&v126 )
        _libc_free((unsigned __int64)v124);
      sub_11C9500(v56, (__int64)"memset_pattern16", 0x10u, v111);
      v61 = *(_QWORD **)(v104 + 8);
      v124 = ".memset_pattern";
      v112 = v61;
      LOWORD(v128) = 259;
      BYTE4(v122[0]) = 0;
      v62 = sub_BD2C40(88, unk_3F0FAE8);
      v63 = (__int64)v62;
      if ( v62 )
        sub_B30000((__int64)v62, v56, v112, 1, 8, v104, (__int64)&v124, 0, 0, v122[0], 0);
      *(_BYTE *)(v63 + 32) = *(_BYTE *)(v63 + 32) & 0x3F | 0x80;
      sub_B2F770(v63, 4u);
      v123 = 257;
      v64 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
      v65 = *(unsigned __int8 **)(v5 + 32 * (2 - v64));
      v105 = *(_QWORD *)(*(_QWORD *)(v5 + 32 * (1 - v64)) + 8LL);
      v113 = sub_AE5020(v57, v105);
      v124 = (const char *)sub_9208B0(v57, v105);
      v125 = v66;
      v119 = (((unsigned __int64)(v124 + 7) >> 3) + (1LL << v113) - 1) >> v113 << v113;
      LOBYTE(v120) = v66;
      v67 = sub_CA1930(&v119);
      v68 = sub_BCB2E0(v135);
      v69 = (unsigned __int8 *)sub_ACD640(v68, v67, 0);
      v70 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v136 + 4);
      if ( v70 == sub_9201A0 )
      {
        if ( *v69 > 0x15u || *v65 > 0x15u )
        {
LABEL_97:
          LOWORD(v128) = 257;
          v108 = sub_B504D0(17, (__int64)v69, (__int64)v65, (__int64)&v124, 0, 0);
          (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v137 + 2))(
            v137,
            v108,
            v122,
            v133,
            v134);
          v88 = (unsigned __int64)v129;
          v71 = v108;
          v65 = &v129[16 * (unsigned int)v130];
          if ( v129 != v65 )
          {
            v109 = v63;
            v89 = v71;
            do
            {
              v90 = *(_QWORD *)(v88 + 8);
              v91 = *(_DWORD *)v88;
              v88 += 16LL;
              sub_B99FD0(v89, v91, v90);
            }
            while ( v65 != (unsigned __int8 *)v88 );
            v71 = v89;
            v63 = v109;
          }
LABEL_76:
          v123 = 257;
          v72 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
          v120 = v63;
          v121 = v71;
          v119 = v72;
          LOWORD(v128) = 257;
          v73 = v142 + 56 * v143;
          if ( v142 == v73 )
          {
            v78 = 4;
            v77 = 4;
          }
          else
          {
            v74 = v142;
            v75 = 0;
            do
            {
              v76 = *(_QWORD *)(v74 + 40) - *(_QWORD *)(v74 + 32);
              v74 += 56;
              v75 += v76 >> 3;
            }
            while ( v73 != v74 );
            v77 = (unsigned int)(v75 + 4);
            v78 = v77 & 0x7FFFFFF;
          }
          v97 = v142;
          v106 = v143;
          LOBYTE(v65) = 16 * (_DWORD)v143 != 0;
          v79 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v143) << 32) | v77);
          v80 = (__int64)v79;
          if ( v79 )
          {
            v110 = v106;
            v107 = (__int64)v79;
            v116 = v116 & 0xE0000000 | ((_DWORD)v65 << 28) | v78;
            sub_B44260((__int64)v79, **(_QWORD **)(v103 + 16), 56, v116, 0, 0);
            *(_QWORD *)(v80 + 72) = 0;
            sub_B4A290(v80, v103, v100, (__int64 *)&v119, 3, (__int64)&v124, v97, v110);
          }
          else
          {
            v107 = 0;
          }
          if ( (_BYTE)v140 )
          {
            v92 = (__int64 *)sub_BD5C60(v107);
            *(_QWORD *)(v80 + 72) = sub_A7A090((__int64 *)(v80 + 72), v92, -1, 72);
            if ( !(unsigned __int8)sub_920620(v107) )
              goto LABEL_84;
          }
          else if ( !(unsigned __int8)sub_920620(v107) )
          {
LABEL_84:
            (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v137 + 2))(
              v137,
              v80,
              v122,
              v133,
              v134);
            v81 = (unsigned __int64)v129;
            v82 = &v129[16 * (unsigned int)v130];
            if ( v129 != v82 )
            {
              do
              {
                v83 = *(_QWORD *)(v81 + 8);
                v84 = *(_DWORD *)v81;
                v81 += 16LL;
                sub_B99FD0(v80, v84, v83);
              }
              while ( v82 != (unsigned __int8 *)v81 );
            }
            sub_B91FC0((__int64 *)&v124, v5);
            sub_B9A100(v107, (__int64 *)&v124);
            v122[0] = *(_QWORD *)(v5 + 72);
            v85 = sub_A744E0(v122, 0);
            v86 = sub_BD5C60(v5);
            sub_A74940((__int64)&v124, v86, v85);
            v122[0] = *(_QWORD *)(v80 + 72);
            v87 = (__int64 *)sub_BD5C60(v5);
            *(_QWORD *)(v80 + 72) = sub_A7B2C0(v122, v87, 1, (__int64)&v124);
            sub_B43D60((_QWORD *)v5);
            if ( (__int64 *)v125 != &v127 )
              _libc_free(v125);
            nullsub_61();
            v144 = &unk_49DA100;
            nullsub_63();
            if ( v129 != v131 )
              _libc_free((unsigned __int64)v129);
            goto LABEL_28;
          }
          v93 = v139;
          if ( v138 )
            sub_B99FD0(v80, 3u, v138);
          sub_B45150(v80, v93);
          goto LABEL_84;
        }
        if ( (unsigned __int8)sub_AC47B0(17) )
          v71 = sub_AD5570(17, (__int64)v69, v65, 0, 0);
        else
          v71 = sub_AABE40(0x11u, v69, v65);
      }
      else
      {
        v71 = v70((__int64)v136, 17u, v69, v65, 0, 0);
      }
      if ( v71 )
        goto LABEL_76;
      goto LABEL_97;
    }
LABEL_65:
    sub_B91220((__int64)&v124, v52);
    goto LABEL_66;
  }
  return v2;
}
