// Function: sub_217C7E0
// Address: 0x217c7e0
//
__int64 __fastcall sub_217C7E0(__int64 a1, __int64 a2, unsigned int a3, __m128i a4, double a5, __m128i a6)
{
  __int16 v6; // ax
  __int64 *v10; // r15
  __int64 v11; // rcx
  __int64 v12; // r14
  __int16 v13; // ax
  __int64 v14; // rax
  char v15; // di
  __int64 v16; // rax
  unsigned int *v17; // rcx
  __int64 v18; // rax
  char v19; // di
  __int64 v20; // rax
  int v21; // eax
  unsigned int *v22; // rcx
  __int64 result; // rax
  __int16 v24; // ax
  _DWORD *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  _QWORD *v28; // r15
  __int64 v29; // rax
  int v30; // ecx
  __int64 v31; // rax
  _QWORD *v32; // r11
  _BOOL4 v33; // ebx
  __int64 *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 *v37; // r15
  __int64 v38; // rax
  __int64 v39; // rbx
  _DWORD *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rsi
  _QWORD *v43; // r15
  __int64 v44; // rax
  int v45; // ecx
  __int64 v46; // r15
  unsigned int *v47; // r14
  __int64 v48; // rax
  char v49; // di
  __int64 v50; // rax
  int v51; // eax
  __int64 v52; // rax
  __int64 v53; // rcx
  unsigned int v54; // r14d
  unsigned __int64 v55; // r15
  __int64 v56; // rsi
  __int64 v57; // r11
  __int64 v58; // rdx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rax
  __int64 *v62; // rax
  __int64 v63; // rsi
  __int64 v64; // r11
  __int64 v65; // rdx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rax
  __int64 *v69; // rax
  unsigned int v70; // eax
  __int64 v71; // rsi
  _QWORD *v72; // r9
  __int128 v73; // rcx
  __int64 v74; // r15
  __int64 v75; // rdx
  __int64 v76; // rcx
  int v77; // r8d
  int v78; // r9d
  __int64 v79; // r9
  _QWORD *v80; // rbx
  __int128 v81; // rax
  __int64 v82; // rax
  _QWORD *v83; // r11
  _BOOL4 v84; // ebx
  __int64 *v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 *v88; // r15
  __int64 v89; // [rsp-10h] [rbp-110h]
  unsigned int *v90; // [rsp+0h] [rbp-100h]
  __int64 v91; // [rsp+0h] [rbp-100h]
  __int64 v92; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v93; // [rsp+8h] [rbp-F8h]
  __int16 v94; // [rsp+8h] [rbp-F8h]
  unsigned int v95; // [rsp+10h] [rbp-F0h]
  __int64 v96; // [rsp+10h] [rbp-F0h]
  __int64 v97; // [rsp+10h] [rbp-F0h]
  __int64 v98; // [rsp+10h] [rbp-F0h]
  __int128 v99; // [rsp+10h] [rbp-F0h]
  __int64 v100; // [rsp+10h] [rbp-F0h]
  __int64 v101; // [rsp+10h] [rbp-F0h]
  __int64 v102; // [rsp+18h] [rbp-E8h]
  __int64 v103; // [rsp+20h] [rbp-E0h]
  _QWORD *v104; // [rsp+20h] [rbp-E0h]
  __int64 v105; // [rsp+20h] [rbp-E0h]
  __int64 v106; // [rsp+20h] [rbp-E0h]
  __int64 v107; // [rsp+20h] [rbp-E0h]
  _QWORD *v108; // [rsp+20h] [rbp-E0h]
  __int64 v109; // [rsp+20h] [rbp-E0h]
  __int128 v110; // [rsp+20h] [rbp-E0h]
  __int64 v111; // [rsp+20h] [rbp-E0h]
  _QWORD *v112; // [rsp+20h] [rbp-E0h]
  __int64 v113; // [rsp+28h] [rbp-D8h]
  __int64 v114; // [rsp+30h] [rbp-D0h] BYREF
  int v115; // [rsp+38h] [rbp-C8h]
  __int128 v116; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v117; // [rsp+50h] [rbp-B0h]
  __m128i v118; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v119; // [rsp+70h] [rbp-90h]
  __int64 *v120; // [rsp+80h] [rbp-80h] BYREF
  __int64 v121; // [rsp+88h] [rbp-78h]
  __int64 v122; // [rsp+90h] [rbp-70h] BYREF
  unsigned __int64 v123; // [rsp+98h] [rbp-68h]
  __int64 v124; // [rsp+A0h] [rbp-60h]
  unsigned __int64 v125; // [rsp+A8h] [rbp-58h]

  v10 = *(__int64 **)(a1 + 32);
  v11 = v10[5];
  v12 = *v10;
  v103 = v11;
  if ( v6 != -4450 )
  {
    if ( v6 != -4451 )
      return 0;
    v13 = *(_WORD *)(v12 + 24);
    if ( v13 < 0 && *(_WORD *)(v11 + 24) == 0xFD95 && v13 == -619 )
    {
      v40 = *(_DWORD **)(v12 + 32);
      v41 = *(_QWORD *)v40;
      if ( *(_WORD *)(*(_QWORD *)v40 + 24LL) == 659 )
      {
        v42 = *(_QWORD *)(v11 + 32);
        if ( v41 == *(_QWORD *)v42 && !v40[2] && *(_DWORD *)(v42 + 8) == 1 )
        {
          v43 = *(_QWORD **)(v41 + 104);
          v122 = v12;
          v121 = 0x200000002LL;
          v120 = &v122;
          v123 = v11;
          v97 = v41;
          if ( (unsigned __int8)sub_1D18CA0(&v122, 2, v41) )
          {
            v44 = *(_QWORD *)(v12 + 48);
            v45 = 0;
            if ( v44 )
            {
              if ( !*(_QWORD *)(v44 + 32) )
              {
                v82 = *(_QWORD *)(v103 + 48);
                if ( v82 )
                {
                  if ( !*(_QWORD *)(v82 + 32) )
                  {
                    v118 = 0u;
                    v83 = *(_QWORD **)(a2 + 16);
                    v119 = 0;
                    v84 = (*(_BYTE *)(v97 + 26) & 8) != 0;
                    if ( (*v43 & 4) != 0 )
                    {
                      v116 = 0u;
                      v117 = 0;
                    }
                    else
                    {
                      v85 = (__int64 *)(*v43 & 0xFFFFFFFFFFFFFFF8LL);
                      LOBYTE(v117) = 0;
                      v116 = (unsigned __int64)v85;
                      if ( v85 )
                      {
                        v86 = *v85;
                        if ( *(_BYTE *)(v86 + 8) == 16 )
                          v86 = **(_QWORD **)(v86 + 16);
                        v45 = *(_DWORD *)(v86 + 8) >> 8;
                      }
                    }
                    v87 = *(_QWORD *)(a1 + 72);
                    HIDWORD(v117) = v45;
                    v88 = *(__int64 **)(v97 + 32);
                    v114 = v87;
                    if ( v87 )
                    {
                      v112 = v83;
                      sub_21700C0(&v114);
                      v83 = v112;
                    }
                    v115 = *(_DWORD *)(a1 + 64);
                    v38 = sub_1D2B730(
                            v83,
                            6,
                            0,
                            (__int64)&v114,
                            *v88,
                            v88[1],
                            v88[5],
                            v88[6],
                            v116,
                            v117,
                            v84,
                            0,
                            (__int64)&v118,
                            0);
LABEL_33:
                    v39 = v38;
                    sub_17CD270(&v114);
                    result = v39;
                    goto LABEL_45;
                  }
                }
              }
            }
          }
          goto LABEL_44;
        }
      }
    }
LABEL_4:
    v92 = v10[1];
    v14 = *(_QWORD *)(v12 + 40) + 16LL * (unsigned int)v92;
    v15 = *(_BYTE *)v14;
    v16 = *(_QWORD *)(v14 + 8);
    LOBYTE(v120) = v15;
    v121 = v16;
    if ( v15 )
      v95 = sub_216FFF0(v15);
    else
      v95 = sub_1F58D40((__int64)&v120);
    if ( (unsigned __int8)sub_216FF90(v12) )
    {
      v17 = *(unsigned int **)(v12 + 32);
      v18 = *(_QWORD *)(*(_QWORD *)v17 + 40LL) + 16LL * v17[2];
      v19 = *(_BYTE *)v18;
      v20 = *(_QWORD *)(v18 + 8);
      LOBYTE(v120) = v19;
      v121 = v20;
      if ( v19 )
      {
        v21 = sub_216FFF0(v19);
      }
      else
      {
        v90 = v17;
        v21 = sub_1F58D40((__int64)&v120);
        v22 = v90;
      }
      if ( 2 * v95 == v21 )
      {
        v46 = v10[6];
        v91 = *(_QWORD *)v22;
        v93 = v22[2] | v92 & 0xFFFFFFFF00000000LL;
        if ( (unsigned __int8)sub_216FF90(v103) )
        {
          v47 = *(unsigned int **)(v103 + 32);
          v48 = *(_QWORD *)(*(_QWORD *)v47 + 40LL) + 16LL * v47[2];
          v49 = *(_BYTE *)v48;
          v50 = *(_QWORD *)(v48 + 8);
          LOBYTE(v120) = v49;
          v121 = v50;
          v51 = v49 ? sub_216FFF0(v49) : sub_1F58D40((__int64)&v120);
          if ( 2 * v95 == v51 )
          {
            v52 = v47[2];
            v53 = *(_QWORD *)v47;
            v54 = 0;
            v55 = v52 | v46 & 0xFFFFFFFF00000000LL;
            if ( v95 == 16 )
            {
              v122 = v53;
              LOBYTE(v54) = 5;
              v120 = &v122;
              v123 = v55;
              v124 = v91;
              v125 = v93;
              v121 = 0x400000002LL;
              if ( (unsigned __int8)sub_216FE40(0x10u, 0x10u, (int *)&v114) )
              {
                v74 = *(_QWORD *)(a2 + 16);
                *(_QWORD *)&v116 = *(_QWORD *)(a1 + 72);
                if ( (_QWORD)v116 )
                  sub_21700C0((__int64 *)&v116);
                DWORD2(v116) = *(_DWORD *)(a1 + 64);
                v118.m128i_i64[0] = sub_1D38BB0(v74, (unsigned int)v114, (__int64)&v116, 5, 0, 1, a4, a5, a6, 0);
                v118.m128i_i64[1] = v75;
                sub_1D23890((__int64)&v120, &v118, v75, v76, v77, v78);
                sub_17CD270((__int64 *)&v116);
                v79 = v89;
                v80 = *(_QWORD **)(a2 + 16);
                *(_QWORD *)&v81 = v120;
                *((_QWORD *)&v81 + 1) = (unsigned int)v121;
                v118.m128i_i64[0] = *(_QWORD *)(a1 + 72);
                if ( v118.m128i_i64[0] )
                {
                  *(_QWORD *)&v110 = v120;
                  *((_QWORD *)&v110 + 1) = (unsigned int)v121;
                  sub_21700C0(v118.m128i_i64);
                  v81 = v110;
                }
                v118.m128i_i32[2] = *(_DWORD *)(a1 + 64);
                v100 = sub_1D2CDB0(v80, 3243, (__int64)&v118, v54, 0, v79, v81);
                sub_17CD270(v118.m128i_i64);
                result = v100;
                goto LABEL_45;
              }
              v94 = 164;
              goto LABEL_56;
            }
            if ( v95 == 32 && byte_4FD2B00 && a3 > 0x31 )
            {
              v122 = v53;
              LOBYTE(v54) = 6;
              v120 = &v122;
              v124 = v91;
              v123 = v55;
              v125 = v93;
              v121 = 0x400000002LL;
              v94 = 165;
LABEL_56:
              v56 = *(_QWORD *)(a1 + 72);
              v57 = *(_QWORD *)(a2 + 16);
              v118.m128i_i64[0] = v56;
              if ( v56 )
              {
                v106 = v57;
                sub_1623A60((__int64)&v118, v56, 2);
                v57 = v106;
              }
              v118.m128i_i32[2] = *(_DWORD *)(a1 + 64);
              v107 = v95;
              v59 = sub_1D38BB0(v57, v95, (__int64)&v118, 5, 0, 1, a4, a5, a6, 0);
              v60 = v58;
              v61 = (unsigned int)v121;
              if ( (unsigned int)v121 >= HIDWORD(v121) )
              {
                v101 = v59;
                v102 = v58;
                sub_16CD150((__int64)&v120, &v122, 0, 16, v59, v58);
                v61 = (unsigned int)v121;
                v59 = v101;
                v60 = v102;
              }
              v62 = &v120[2 * v61];
              *v62 = v59;
              v62[1] = v60;
              LODWORD(v121) = v121 + 1;
              if ( v118.m128i_i64[0] )
                sub_161E7C0((__int64)&v118, v118.m128i_i64[0]);
              v63 = *(_QWORD *)(a1 + 72);
              v64 = *(_QWORD *)(a2 + 16);
              v118.m128i_i64[0] = v63;
              if ( v63 )
              {
                v98 = v64;
                sub_1623A60((__int64)&v118, v63, 2);
                v64 = v98;
              }
              v118.m128i_i32[2] = *(_DWORD *)(a1 + 64);
              v66 = sub_1D38BB0(v64, v107, (__int64)&v118, 5, 0, 1, a4, a5, a6, 0);
              v67 = v65;
              v68 = (unsigned int)v121;
              if ( (unsigned int)v121 >= HIDWORD(v121) )
              {
                v113 = v65;
                v111 = v66;
                sub_16CD150((__int64)&v120, &v122, 0, 16, v66, v65);
                v68 = (unsigned int)v121;
                v66 = v111;
                v67 = v113;
              }
              v69 = &v120[2 * v68];
              *v69 = v66;
              v69[1] = v67;
              v70 = v121 + 1;
              LODWORD(v121) = v121 + 1;
              if ( v118.m128i_i64[0] )
              {
                sub_161E7C0((__int64)&v118, v118.m128i_i64[0]);
                v70 = v121;
              }
              v71 = *(_QWORD *)(a1 + 72);
              v72 = *(_QWORD **)(a2 + 16);
              *((_QWORD *)&v73 + 1) = v70;
              *(_QWORD *)&v73 = v120;
              v118.m128i_i64[0] = v71;
              if ( v71 )
              {
                *(_QWORD *)&v99 = v120;
                *((_QWORD *)&v99 + 1) = v70;
                v108 = v72;
                sub_1623A60((__int64)&v118, v71, 2);
                v73 = v99;
                v72 = v108;
              }
              v118.m128i_i32[2] = *(_DWORD *)(a1 + 64);
              result = sub_1D2CDB0(v72, v94, (__int64)&v118, v54, 0, (__int64)v72, v73);
              if ( v118.m128i_i64[0] )
              {
                v109 = result;
                sub_161E7C0((__int64)&v118, v118.m128i_i64[0]);
                result = v109;
              }
              goto LABEL_45;
            }
          }
        }
      }
    }
    return 0;
  }
  v24 = *(_WORD *)(v12 + 24);
  if ( v24 >= 0 )
    goto LABEL_4;
  if ( *(_WORD *)(v11 + 24) != 0xFD99 )
    goto LABEL_4;
  if ( v24 != -615 )
    goto LABEL_4;
  v25 = *(_DWORD **)(v12 + 32);
  v26 = *(_QWORD *)v25;
  if ( *(_WORD *)(*(_QWORD *)v25 + 24LL) != 659 )
    goto LABEL_4;
  v27 = *(_QWORD *)(v11 + 32);
  if ( *(_QWORD *)v27 != v26 || v25[2] || *(_DWORD *)(v27 + 8) != 1 )
    goto LABEL_4;
  v28 = *(_QWORD **)(v26 + 104);
  v122 = v12;
  v121 = 0x200000002LL;
  v120 = &v122;
  v123 = v11;
  v96 = v26;
  if ( (unsigned __int8)sub_1D18CA0(&v122, 2, v26) )
  {
    v29 = *(_QWORD *)(v12 + 48);
    v30 = 0;
    if ( v29 )
    {
      if ( !*(_QWORD *)(v29 + 32) )
      {
        v31 = *(_QWORD *)(v103 + 48);
        if ( v31 )
        {
          if ( !*(_QWORD *)(v31 + 32) )
          {
            v118 = 0u;
            v32 = *(_QWORD **)(a2 + 16);
            v119 = 0;
            v33 = (*(_BYTE *)(v96 + 26) & 8) != 0;
            if ( (*v28 & 4) != 0 )
            {
              v116 = 0u;
              v117 = 0;
            }
            else
            {
              v34 = (__int64 *)(*v28 & 0xFFFFFFFFFFFFFFF8LL);
              LOBYTE(v117) = 0;
              v116 = (unsigned __int64)v34;
              if ( v34 )
              {
                v35 = *v34;
                if ( *(_BYTE *)(v35 + 8) == 16 )
                  v35 = **(_QWORD **)(v35 + 16);
                v30 = *(_DWORD *)(v35 + 8) >> 8;
              }
            }
            v36 = *(_QWORD *)(a1 + 72);
            HIDWORD(v117) = v30;
            v37 = *(__int64 **)(v96 + 32);
            v114 = v36;
            if ( v36 )
            {
              v104 = v32;
              sub_21700C0(&v114);
              v32 = v104;
            }
            v115 = *(_DWORD *)(a1 + 64);
            v38 = sub_1D2B730(
                    v32,
                    5,
                    0,
                    (__int64)&v114,
                    *v37,
                    v37[1],
                    v37[5],
                    v37[6],
                    v116,
                    v117,
                    v33,
                    0,
                    (__int64)&v118,
                    0);
            goto LABEL_33;
          }
        }
      }
    }
  }
LABEL_44:
  result = 0;
LABEL_45:
  if ( v120 != &v122 )
  {
    v105 = result;
    _libc_free((unsigned __int64)v120);
    return v105;
  }
  return result;
}
