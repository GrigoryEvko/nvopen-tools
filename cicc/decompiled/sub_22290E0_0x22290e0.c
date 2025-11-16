// Function: sub_22290E0
// Address: 0x22290e0
//
_QWORD *__fastcall sub_22290E0(
        __int64 a1,
        _QWORD *a2,
        unsigned __int64 a3,
        _QWORD *a4,
        _QWORD *a5,
        __int64 a6,
        _DWORD *a7,
        _DWORD *a8,
        wchar_t *s)
{
  int v9; // r15d
  unsigned int v10; // ebx
  __int64 v11; // rdi
  __int64 v13; // r14
  size_t i; // rbp
  bool v15; // r13
  char v16; // dl
  char v17; // r15
  int v18; // eax
  size_t v19; // rbx
  wchar_t *v20; // r15
  size_t v21; // r15
  char v22; // al
  int v23; // eax
  wchar_t v24; // r15d
  unsigned int v25; // eax
  int *v26; // rax
  int v27; // eax
  int *v28; // rax
  int v29; // eax
  bool v30; // zf
  _QWORD *v31; // rax
  size_t v32; // rcx
  _QWORD *v33; // rbp
  unsigned __int64 v35; // rax
  unsigned int *v36; // rax
  unsigned int v37; // edx
  int v38; // eax
  _QWORD *v39; // rax
  unsigned __int64 v40; // rbx
  _QWORD *v41; // rax
  unsigned int v42; // edx
  unsigned __int64 v43; // rbx
  __int64 v44; // rax
  unsigned int v45; // edx
  __int64 v46; // rcx
  unsigned __int64 v47; // rbx
  __int64 v48; // rsi
  unsigned __int64 v49; // rax
  __int64 v50; // rsi
  _QWORD *v51; // rax
  unsigned __int64 v52; // rbx
  _QWORD *v53; // rax
  unsigned int v54; // edx
  __int64 v55; // rdx
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rbx
  _QWORD *v58; // rax
  unsigned int v59; // edx
  unsigned int v60; // edx
  unsigned __int64 v61; // rbx
  _QWORD *v62; // rax
  unsigned int v63; // edx
  unsigned int v64; // edx
  _QWORD *v65; // rax
  _QWORD *v66; // rax
  unsigned int v67; // edx
  __int64 v68; // rdx
  _QWORD *v69; // rax
  __int64 v70; // rdx
  __int64 v71; // r8
  char v72; // r9
  unsigned __int64 v73; // rsi
  char v74; // cl
  int v75; // ecx
  int v76; // edx
  _QWORD *v77; // rsi
  unsigned int v78; // edx
  unsigned __int64 v79; // rbx
  unsigned int v80; // edx
  __int64 v81; // rcx
  unsigned __int64 v82; // rbx
  unsigned int *v83; // rax
  unsigned int v84; // eax
  unsigned int *v85; // rax
  unsigned int v86; // eax
  unsigned int *v87; // rax
  unsigned int v88; // eax
  unsigned int *v89; // rax
  unsigned int v90; // eax
  int *v91; // rax
  int v92; // eax
  bool v93; // zf
  _QWORD *v94; // rax
  int *v95; // rcx
  int v96; // eax
  int *v97; // rax
  int v98; // eax
  _DWORD *v99; // rax
  unsigned int v100; // [rsp+Ch] [rbp-25Ch]
  size_t v102; // [rsp+18h] [rbp-250h]
  unsigned __int64 v103; // [rsp+20h] [rbp-248h]
  __int64 v107; // [rsp+40h] [rbp-228h]
  unsigned __int64 v108; // [rsp+48h] [rbp-220h]
  char v109; // [rsp+48h] [rbp-220h]
  bool v110; // [rsp+50h] [rbp-218h]
  char v111; // [rsp+50h] [rbp-218h]
  __int64 v112; // [rsp+50h] [rbp-218h]
  __int64 v113; // [rsp+58h] [rbp-210h]
  __int64 v114; // [rsp+58h] [rbp-210h]
  char v115; // [rsp+58h] [rbp-210h]
  char v116; // [rsp+60h] [rbp-208h]
  char v117; // [rsp+60h] [rbp-208h]
  char v118; // [rsp+66h] [rbp-202h]
  __int64 v119; // [rsp+68h] [rbp-200h]
  int v120; // [rsp+1C8h] [rbp-A0h] BYREF
  int v121; // [rsp+1CCh] [rbp-9Ch] BYREF
  wchar_t v122[2]; // [rsp+1D0h] [rbp-98h] BYREF
  __int64 v123; // [rsp+1D8h] [rbp-90h]
  __int64 v124; // [rsp+1E0h] [rbp-88h]
  __int64 v125; // [rsp+1E8h] [rbp-80h]
  __int64 v126; // [rsp+1F0h] [rbp-78h]
  __int64 v127; // [rsp+1F8h] [rbp-70h]
  __int64 v128; // [rsp+200h] [rbp-68h]
  __int64 v129; // [rsp+208h] [rbp-60h]
  __int64 v130; // [rsp+210h] [rbp-58h]
  __int64 v131; // [rsp+218h] [rbp-50h]
  __int64 v132; // [rsp+220h] [rbp-48h]
  __int64 v133; // [rsp+228h] [rbp-40h]

  v9 = (int)a5;
  v10 = a3;
  v103 = a3;
  v11 = a6 + 208;
  v107 = sub_2244AF0(a6 + 208);
  v13 = sub_2243120(v11);
  v100 = v10;
  v102 = wcslen(s);
  v120 = 0;
  v118 = v9 == -1;
  for ( i = 0; ; i = v21 + 1 )
  {
    v15 = v100 == -1;
    v16 = v15 && a2 != 0;
    if ( v16 )
    {
      v26 = (int *)a2[2];
      if ( (unsigned __int64)v26 >= a2[3] )
      {
        v27 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
        v16 = v15 && a2 != 0;
      }
      else
      {
        v27 = *v26;
      }
      if ( v27 == -1 )
        a2 = 0;
      else
        v16 = 0;
    }
    else
    {
      v16 = v100 == -1;
    }
    v17 = v118 & (a4 != 0);
    if ( v17 )
      break;
    v18 = v120;
    if ( v16 == v118 )
      goto LABEL_33;
LABEL_6:
    if ( i >= v102 )
      goto LABEL_33;
    if ( v18 )
    {
      v33 = a2;
LABEL_35:
      *a7 |= 4u;
      return v33;
    }
    v19 = i;
    v20 = &s[i];
    if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v13 + 96LL))(
           v13,
           (unsigned int)*v20,
           0) == 37 )
    {
      v21 = i + 1;
      v22 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v13 + 96LL))(
              v13,
              (unsigned int)s[v19 + 1],
              0);
      v121 = 0;
      if ( v22 != 69 && v22 != 79 )
      {
        LOBYTE(v23) = v22 - 65;
LABEL_12:
        switch ( (char)v23 )
        {
          case 0:
            v65 = *(_QWORD **)(v107 + 16);
            *(_QWORD *)v122 = v65[11];
            v123 = v65[12];
            v124 = v65[13];
            v125 = v65[14];
            v126 = v65[15];
            v127 = v65[16];
            v128 = v65[17];
            v52 = v100 | v103 & 0xFFFFFFFF00000000LL;
            v53 = sub_2228060(a1, a2, v52, a4, a5, &v121, (__int64)v122, 7, a6, &v120);
            goto LABEL_69;
          case 1:
            v66 = *(_QWORD **)(v107 + 16);
            *(_QWORD *)v122 = v66[25];
            v123 = v66[26];
            v124 = v66[27];
            v125 = v66[28];
            v126 = v66[29];
            v127 = v66[30];
            v128 = v66[31];
            v129 = v66[32];
            v130 = v66[33];
            v131 = v66[34];
            v132 = v66[35];
            v133 = v66[36];
            v40 = v100 | v103 & 0xFFFFFFFF00000000LL;
            v41 = sub_2228060(a1, a2, v40, a4, a5, &v121, (__int64)v122, 12, a6, &v120);
            goto LABEL_52;
          case 2:
          case 24:
          case 56:
            a2 = sub_2227C20(a1, a2, v100, a4, (int)a5, &v121, 0, 9999, 4u, a6, &v120);
            v103 = v37 | v103 & 0xFFFFFFFF00000000LL;
            v100 = v37;
            if ( !v120 )
            {
              v38 = v121 - 1900;
              if ( v121 < 0 )
                v38 = v121 + 100;
              a8[5] = v38;
            }
            continue;
          case 3:
            (*(void (__fastcall **)(__int64, char *, const char *, wchar_t *))(*(_QWORD *)v13 + 88LL))(
              v13,
              &aHM[-9],
              "%H:%M",
              v122);
            v43 = v100 | v103 & 0xFFFFFFFF00000000LL;
            v44 = sub_22290E0(a1, (int)a2, v100, (int)a4, (int)a5, a6, (__int64)&v120, (__int64)a8, v122);
            goto LABEL_55;
          case 7:
            v61 = v100 | v103 & 0xFFFFFFFF00000000LL;
            v62 = sub_2227C20(a1, a2, v100, a4, (int)a5, &v121, 0, 23, 2u, a6, &v120);
            goto LABEL_82;
          case 8:
            v61 = v100 | v103 & 0xFFFFFFFF00000000LL;
            v62 = sub_2227C20(a1, a2, v100, a4, (int)a5, &v121, 1, 12, 2u, a6, &v120);
LABEL_82:
            v100 = v63;
            a2 = v62;
            v103 = v63 | v61 & 0xFFFFFFFF00000000LL;
            if ( !v120 )
              a8[2] = v121;
            continue;
          case 12:
            a2 = sub_2227C20(a1, a2, v100, a4, (int)a5, &v121, 0, 59, 2u, a6, &v120);
            v103 = v64 | v103 & 0xFFFFFFFF00000000LL;
            v100 = v64;
            if ( !v120 )
              a8[1] = v121;
            continue;
          case 17:
            (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, wchar_t *))(*(_QWORD *)v13 + 88LL))(
              v13,
              &byte_4360B49[-6],
              byte_4360B49,
              v122);
            v43 = v100 | v103 & 0xFFFFFFFF00000000LL;
            v44 = sub_22290E0(a1, (int)a2, v100, (int)a4, (int)a5, a6, (__int64)&v120, (__int64)a8, v122);
            goto LABEL_55;
          case 18:
            a2 = sub_2227C20(a1, a2, v100, a4, (int)a5, &v121, 0, 60, 2u, a6, &v120);
            v103 = v67 | v103 & 0xFFFFFFFF00000000LL;
            v100 = v67;
            if ( !v120 )
              *a8 = v121;
            continue;
          case 19:
            (*(void (**)(__int64, char *, const char *, ...))(*(_QWORD *)v13 + 88LL))(v13, &a9lu[-9], "%.9lu", v122);
            v44 = sub_22290E0(a1, (int)a2, v100, (int)a4, (int)a5, a6, (__int64)&v120, (__int64)a8, v122);
            v46 = v45;
            v47 = v103 & 0xFFFFFFFF00000000LL;
            goto LABEL_56;
          case 23:
            v44 = sub_22290E0(
                    a1,
                    (int)a2,
                    v100,
                    (int)a4,
                    (int)a5,
                    a6,
                    (__int64)&v120,
                    (__int64)a8,
                    *(wchar_t **)(*(_QWORD *)(v107 + 16) + 32LL));
            v46 = v45;
            v47 = v103 & 0xFFFFFFFF00000000LL;
            goto LABEL_56;
          case 25:
            if ( a2 && v100 == -1 )
            {
              v89 = (unsigned int *)a2[2];
              if ( (unsigned __int64)v89 >= a2[3] )
              {
                v90 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
                v68 = v90;
              }
              else
              {
                v68 = *v89;
                v90 = *v89;
              }
              if ( v90 == -1 )
                a2 = 0;
            }
            else
            {
              v68 = v100;
            }
            if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v13 + 16LL))(v13, 256, v68) )
              goto LABEL_20;
            v69 = sub_2228060(
                    a1,
                    a2,
                    v100 | v103 & 0xFFFFFFFF00000000LL,
                    a4,
                    a5,
                    v122,
                    (__int64)&off_4CDFAE0,
                    14,
                    a6,
                    &v120);
            v73 = 0xFFFFFFFF00000000LL;
            a2 = v69;
            v108 = (unsigned int)v70 | v103 & 0xFFFFFFFF00000000LL;
            v103 = v108;
            v100 = v70;
            LOBYTE(v71) = (_DWORD)v70 == -1;
            v72 = v71;
            LOBYTE(v73) = v71 & (v69 != 0);
            if ( (_BYTE)v73 )
            {
              v95 = (int *)v69[2];
              if ( (unsigned __int64)v95 >= v69[3] )
              {
                v114 = v70;
                v110 = (_DWORD)v70 == -1;
                v96 = (*(__int64 (__fastcall **)(_QWORD *, unsigned __int64, __int64, int *, __int64, _QWORD))(*v69 + 72LL))(
                        v69,
                        v73,
                        v70,
                        v95,
                        v71,
                        (unsigned int)v71);
                v70 = v114;
                LOBYTE(v71) = v110;
                v73 = (unsigned __int8)v73;
              }
              else
              {
                v96 = *v95;
              }
              v72 = 0;
              if ( v96 == -1 )
              {
                v72 = v73;
                a2 = 0;
                v73 = 0;
              }
            }
            v74 = v118 & (a4 != 0);
            if ( v74 )
            {
              v91 = (int *)a4[2];
              if ( (unsigned __int64)v91 >= a4[3] )
              {
                v119 = v70;
                v115 = v72;
                v111 = v71;
                v92 = (*(__int64 (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4);
                v70 = v119;
                v74 = v118 & (a4 != 0);
                v72 = v115;
                LOBYTE(v71) = v111;
                v73 = (unsigned __int8)v73;
              }
              else
              {
                v92 = *v91;
              }
              v93 = v92 == -1;
              v94 = 0;
              if ( !v93 )
                v94 = a4;
              a4 = v94;
              if ( !v93 )
                v74 = 0;
            }
            else
            {
              v74 = v118;
            }
            if ( v72 != v74 && !(v122[0] | v120) )
            {
              v75 = v70;
              if ( (_BYTE)v73 )
              {
                v97 = (int *)a2[2];
                if ( (unsigned __int64)v97 >= a2[3] )
                {
                  v112 = v70;
                  v117 = v71;
                  v98 = (*(__int64 (__fastcall **)(_QWORD *, unsigned __int64, __int64, _QWORD))(*a2 + 72LL))(
                          a2,
                          v73,
                          v70,
                          (unsigned int)v70);
                  v70 = v112;
                  LOBYTE(v71) = v117;
                  v75 = v98;
                }
                else
                {
                  v75 = *v97;
                }
                if ( v75 == -1 )
                  a2 = 0;
              }
              v113 = v70;
              v116 = v71;
              if ( v75 == (*(unsigned int (__fastcall **)(__int64, __int64))(*(_QWORD *)v13 + 80LL))(v13, 45) )
                goto LABEL_108;
              v76 = v113;
              if ( a2 && v116 )
              {
                v99 = (_DWORD *)a2[2];
                v76 = (unsigned __int64)v99 >= a2[3]
                    ? (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 72LL))(a2, 45, v113)
                    : *v99;
                if ( v76 == -1 )
                  a2 = 0;
              }
              if ( v76 == (*(unsigned int (__fastcall **)(__int64, __int64))(*(_QWORD *)v13 + 80LL))(v13, 43) )
              {
LABEL_108:
                v77 = sub_2227C20(a1, a2, v108, a4, (int)a5, v122, 0, 23, 2u, a6, &v120);
                v79 = v78 | v108 & 0xFFFFFFFF00000000LL;
                a2 = sub_2227C20(a1, v77, v78, a4, (int)a5, v122, 0, 59, 2u, a6, &v120);
                v100 = v80;
                v103 = v80 | v79 & 0xFFFFFFFF00000000LL;
              }
            }
            continue;
          case 32:
            v51 = *(_QWORD **)(v107 + 16);
            *(_QWORD *)v122 = v51[18];
            v123 = v51[19];
            v124 = v51[20];
            v125 = v51[21];
            v126 = v51[22];
            v127 = v51[23];
            v128 = v51[24];
            v52 = v100 | v103 & 0xFFFFFFFF00000000LL;
            v53 = sub_2228060(a1, a2, v52, a4, a5, &v121, (__int64)v122, 7, a6, &v120);
LABEL_69:
            v100 = v54;
            a2 = v53;
            v103 = v54 | v52 & 0xFFFFFFFF00000000LL;
            if ( !v120 )
              a8[6] = v121;
            continue;
          case 33:
          case 39:
            v39 = *(_QWORD **)(v107 + 16);
            *(_QWORD *)v122 = v39[37];
            v123 = v39[38];
            v124 = v39[39];
            v125 = v39[40];
            v126 = v39[41];
            v127 = v39[42];
            v128 = v39[43];
            v129 = v39[44];
            v130 = v39[45];
            v131 = v39[46];
            v132 = v39[47];
            v133 = v39[48];
            v40 = v100 | v103 & 0xFFFFFFFF00000000LL;
            v41 = sub_2228060(a1, a2, v40, a4, a5, &v121, (__int64)v122, 12, a6, &v120);
LABEL_52:
            v100 = v42;
            a2 = v41;
            v103 = v42 | v40 & 0xFFFFFFFF00000000LL;
            if ( !v120 )
              a8[4] = v121;
            continue;
          case 34:
            v43 = v100 | v103 & 0xFFFFFFFF00000000LL;
            v44 = sub_22290E0(
                    a1,
                    (int)a2,
                    v100,
                    (int)a4,
                    (int)a5,
                    a6,
                    (__int64)&v120,
                    (__int64)a8,
                    *(wchar_t **)(*(_QWORD *)(v107 + 16) + 48LL));
            goto LABEL_55;
          case 35:
            v58 = sub_2227C20(a1, a2, v100, a4, (int)a5, &v121, 1, 31, 2u, a6, &v120);
            v81 = v59;
            v82 = v103 & 0xFFFFFFFF00000000LL;
            goto LABEL_111;
          case 36:
            if ( a2 && v100 == -1 )
            {
              v87 = (unsigned int *)a2[2];
              if ( (unsigned __int64)v87 >= a2[3] )
              {
                v88 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
                v55 = v88;
              }
              else
              {
                v55 = *v87;
                v88 = *v87;
              }
              if ( v88 == -1 )
                a2 = 0;
            }
            else
            {
              v55 = v100;
            }
            if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v13 + 16LL))(
                   v13,
                   0x2000,
                   v55) )
            {
              v56 = a2[2];
              if ( v56 >= a2[3] )
                (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
              else
                a2[2] = v56 + 4;
              v57 = v103 | 0xFFFFFFFF;
              v58 = sub_2227C20(a1, a2, 0xFFFFFFFF, a4, (int)a5, &v121, 1, 9, 1u, a6, &v120);
            }
            else
            {
              v57 = v100 | v103 & 0xFFFFFFFF00000000LL;
              v58 = sub_2227C20(a1, a2, v100, a4, (int)a5, &v121, 10, 31, 2u, a6, &v120);
            }
            v81 = v59;
            v82 = v57 & 0xFFFFFFFF00000000LL;
LABEL_111:
            v100 = v59;
            a2 = v58;
            v103 = v81 | v82;
            if ( !v120 )
              a8[3] = v121;
            continue;
          case 44:
            a2 = sub_2227C20(a1, a2, v100, a4, (int)a5, &v121, 1, 12, 2u, a6, &v120);
            v103 = v60 | v103 & 0xFFFFFFFF00000000LL;
            v100 = v60;
            if ( !v120 )
              a8[4] = v121 - 1;
            continue;
          case 45:
            if ( a2 && v100 == -1 )
            {
              v85 = (unsigned int *)a2[2];
              if ( (unsigned __int64)v85 >= a2[3] )
              {
                v86 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
                v48 = v86;
              }
              else
              {
                v48 = *v85;
                v86 = *v85;
              }
              if ( v86 == -1 )
                a2 = 0;
            }
            else
            {
              v48 = v100;
            }
            if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v13 + 96LL))(v13, v48, 0) != 10 )
              goto LABEL_20;
            goto LABEL_61;
          case 51:
            if ( a2 && v100 == -1 )
            {
              v83 = (unsigned int *)a2[2];
              if ( (unsigned __int64)v83 >= a2[3] )
              {
                v84 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
                v50 = v84;
              }
              else
              {
                v50 = *v83;
                v84 = *v83;
              }
              if ( v84 == -1 )
                a2 = 0;
            }
            else
            {
              v50 = v100;
            }
            if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v13 + 96LL))(v13, v50, 0) != 9 )
              goto LABEL_20;
LABEL_61:
            v49 = a2[2];
            if ( v49 >= a2[3] )
              (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
            else
              a2[2] = v49 + 4;
            goto LABEL_40;
          case 55:
            v43 = v100 | v103 & 0xFFFFFFFF00000000LL;
            v44 = sub_22290E0(
                    a1,
                    (int)a2,
                    v100,
                    (int)a4,
                    (int)a5,
                    a6,
                    (__int64)&v120,
                    (__int64)a8,
                    *(wchar_t **)(*(_QWORD *)(v107 + 16) + 16LL));
LABEL_55:
            v46 = v45;
            v47 = v43 & 0xFFFFFFFF00000000LL;
LABEL_56:
            v100 = v45;
            a2 = (_QWORD *)v44;
            v103 = v46 | v47;
            continue;
          default:
            goto LABEL_20;
        }
      }
      v21 = i + 2;
      v23 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v13 + 96LL))(
              v13,
              (unsigned int)s[v19 + 2],
              0)
          - 65;
      if ( (unsigned __int8)v23 <= 0x38u )
        goto LABEL_12;
LABEL_20:
      v120 |= 4u;
    }
    else
    {
      v24 = *v20;
      if ( a2 && v100 == -1 )
      {
        v36 = (unsigned int *)a2[2];
        if ( (unsigned __int64)v36 >= a2[3] )
          v25 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
        else
          v25 = *v36;
        if ( v25 == -1 )
          a2 = 0;
      }
      else
      {
        v25 = v100;
      }
      if ( v24 == v25 )
      {
        v35 = a2[2];
        if ( v35 >= a2[3] )
          (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
        else
          a2[2] = v35 + 4;
        v21 = i;
LABEL_40:
        v100 = -1;
      }
      else
      {
        v120 |= 4u;
        v21 = i;
      }
    }
  }
  v28 = (int *)a4[2];
  if ( (unsigned __int64)v28 >= a4[3] )
  {
    v109 = v16;
    v29 = (*(__int64 (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4);
    v16 = v109;
  }
  else
  {
    v29 = *v28;
  }
  v30 = v29 == -1;
  v31 = 0;
  if ( !v30 )
    v31 = a4;
  a4 = v31;
  if ( !v30 )
    v17 = 0;
  v18 = v120;
  if ( v16 != v17 )
    goto LABEL_6;
LABEL_33:
  v32 = i;
  v33 = a2;
  if ( v18 || v32 != v102 )
    goto LABEL_35;
  return v33;
}
