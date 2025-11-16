// Function: sub_766570
// Address: 0x766570
//
char __fastcall sub_766570(
        _QWORD *a1,
        char a2,
        __int64 (__fastcall *a3)(_QWORD, _QWORD),
        __int64 (__fastcall *a4)(_QWORD, _QWORD),
        __int64 a5)
{
  unsigned __int64 v5; // rax
  __int64 (__fastcall *v6)(_QWORD, _QWORD); // r12
  __int64 (__fastcall *v7)(_QWORD, _QWORD); // r13
  int v8; // r14d
  __int64 (__fastcall *v10)(_QWORD, __int64); // r9
  __int64 (__fastcall *v11)(_QWORD, _QWORD); // r10
  __int64 v12; // rax
  int v13; // r15d
  __int64 (__fastcall *v14)(_QWORD, __int64); // rax
  __int64 v15; // rax
  int v16; // r11d
  __int64 (__fastcall *v17)(_QWORD, __int64); // rax
  __int64 (__fastcall *v18)(_QWORD, __int64); // rax
  __int64 (__fastcall *v19)(_QWORD, __int64); // rax
  __int64 v20; // rax
  bool v21; // zf
  __int64 (__fastcall *v22)(_QWORD, __int64); // rax
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // r10d
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rdi
  __int64 v29; // rax
  int v30; // esi
  __int64 (__fastcall *v31)(_QWORD, __int64); // rax
  __int64 v32; // rax
  int v33; // ecx
  __int64 (__fastcall *v34)(_QWORD, __int64); // rax
  int v35; // r15d
  __int64 v36; // rcx
  __int64 v37; // rax
  char v38; // al
  __int64 v39; // rax
  __int64 v40; // rax
  int v41; // ecx
  __int64 (__fastcall *v42)(_QWORD, __int64); // rax
  __int64 v43; // rax
  int v44; // edx
  __int64 (__fastcall *v45)(_QWORD, __int64); // rax
  unsigned __int8 v46; // al
  __int64 v47; // rax
  __int64 v48; // rax
  int v49; // r8d
  __int64 v50; // rax
  int v51; // edi
  __int64 v52; // rax
  __int64 v53; // rax
  bool v54; // sf
  __int64 v55; // rax
  int v56; // edx
  __int64 (__fastcall *v57)(_QWORD, __int64); // rax
  __int64 v58; // rax
  int v59; // edx
  __int64 (__fastcall *v60)(_QWORD, __int64); // rax
  __int64 v61; // rax
  __int64 v62; // rax
  int v63; // r9d
  __int64 (__fastcall *v64)(_QWORD, __int64); // rax
  __int64 v65; // rax
  int v66; // r8d
  __int64 (__fastcall *v67)(_QWORD, __int64); // rax
  __int64 v68; // rax
  __int64 v69; // rax
  int v70; // r11d
  __int64 (__fastcall *v71)(_QWORD, __int64); // rax
  __int64 v72; // rax
  int v73; // r10d
  __int64 (__fastcall *v74)(_QWORD, __int64); // rax
  char v75; // r15
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 (__fastcall *v78)(_QWORD, __int64); // r9
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  int v83; // ecx
  __int64 (__fastcall *v84)(_QWORD, __int64); // rax
  __int64 v85; // rax
  int v86; // edx
  __int64 (__fastcall *v87)(_QWORD, __int64); // rax
  __int64 v88; // rax
  int v89; // r8d
  __int64 (__fastcall *v90)(_QWORD, __int64); // rax
  __int64 v91; // rax
  int v92; // edi
  __int64 (__fastcall *v93)(_QWORD, __int64); // rax
  char v94; // al
  __int64 v95; // rax
  int v96; // edx
  __int64 (__fastcall *v97)(_QWORD, __int64); // rax
  __int64 v98; // rax
  int v99; // r10d
  __int64 (__fastcall *v100)(_QWORD, __int64); // rax
  __int64 (__fastcall *v101)(_QWORD, __int64); // r9
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 (__fastcall *v104)(_QWORD, __int64); // r9
  __int64 v105; // rax
  __int64 (__fastcall *v106)(_QWORD, __int64); // r9
  __int64 v107; // rax
  __int64 v108; // rax
  __int64 (__fastcall *v109)(_QWORD, __int64); // rdx
  __int64 v110; // rdi
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // rax
  __int64 v116; // rax
  __int64 v117; // rax
  __int64 (__fastcall *v118)(_QWORD, __int64); // r9
  __int64 v119; // rax
  __int64 v120; // rax
  int v121; // eax
  __int64 v122; // rax
  int v123; // r11d
  __int64 (__fastcall *v124)(_QWORD, __int64); // rax
  __int64 v125; // rax
  __int64 v126; // rdi
  __int64 (__fastcall *v127)(_QWORD, __int64); // r9
  __int64 v128; // rdi
  char v129; // al
  __int64 v130; // rax
  __int64 v131; // rax
  __int64 v132; // rax
  __int64 v133; // rax
  __int64 v134; // rax
  char v135; // al
  __int64 v136; // rax
  __int64 v137; // rdi
  __int64 v138; // rax
  __int64 v139; // rcx
  __int64 v140; // rax
  char v142; // [rsp+Fh] [rbp-31h]

  v6 = qword_4F08028;
  v7 = qword_4F08020;
  qword_4F08028 = a3;
  v8 = dword_4F08018;
  qword_4F08020 = a4;
  dword_4F08018 = a5;
  v10 = a3;
  v11 = a4;
  switch ( a2 )
  {
    case 1:
      if ( a3 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                26,
                a3,
                a4,
                a5,
                a3);
        if ( qword_4F08028 )
        {
          a1[1] = qword_4F08028(a1[1], 26);
          if ( qword_4F08028 )
            a1[2] = qword_4F08028(a1[2], 26);
        }
        v11 = qword_4F08020;
      }
      if ( v11 )
      {
        a1[5] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))v11)(
                  a1[5],
                  1,
                  a3,
                  a4,
                  a5,
                  v10);
        v5 = (unsigned __int64)qword_4F08028;
        if ( !qword_4F08028 )
        {
          v5 = (unsigned __int64)qword_4F08020;
          if ( !qword_4F08020 )
            goto LABEL_35;
          goto LABEL_559;
        }
      }
      else
      {
        v5 = (unsigned __int64)qword_4F08028;
        if ( !qword_4F08028 )
          goto LABEL_35;
      }
      a1[6] = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(a1[6], 1);
      v5 = (unsigned __int64)qword_4F08020;
      if ( !qword_4F08020 )
        goto LABEL_560;
LABEL_559:
      a1[7] = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(a1[7], 1);
LABEL_560:
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        v5 = qword_4F08028(a1[8], 85);
        a1[8] = v5;
      }
      goto LABEL_35;
    case 2:
      if ( a4 )
      {
        v102 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                 a1[15],
                 2,
                 a3,
                 a4,
                 a5,
                 a3);
        a3 = qword_4F08028;
        a1[15] = v102;
      }
      if ( a3 )
      {
        v103 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                 a1[16],
                 6,
                 a3,
                 a4,
                 a5,
                 v10);
        v104 = qword_4F08028;
        a1[16] = v103;
        if ( v104 )
        {
          v105 = v104(a1[17], 6);
          v106 = qword_4F08028;
          a1[17] = v105;
          if ( v106 )
            a1[18] = v106(a1[18], 13);
        }
      }
      if ( dword_4F08018 )
      {
        a1[19] = 0;
        a1[20] = 0;
      }
      switch ( *((_BYTE *)a1 + 173) )
      {
        case 0:
        case 1:
        case 3:
        case 5:
        case 0xE:
          goto LABEL_693;
        case 2:
          v118 = qword_4F08028;
          if ( !qword_4F08028 )
            goto LABEL_697;
          a1[23] = qword_4F08028(a1[23], 25);
          v118 = qword_4F08028;
          goto LABEL_694;
        case 4:
          v118 = qword_4F08028;
          if ( !qword_4F08028 )
            goto LABEL_697;
          a1[22] = qword_4F08028(a1[22], 27);
          v118 = qword_4F08028;
          goto LABEL_694;
        case 6:
          switch ( *((_BYTE *)a1 + 176) )
          {
            case 0:
              if ( qword_4F08028 )
                a1[23] = qword_4F08028(a1[23], 11);
              goto LABEL_1021;
            case 1:
              if ( qword_4F08028 )
                a1[23] = qword_4F08028(a1[23], 7);
              goto LABEL_1021;
            case 2:
            case 3:
              if ( qword_4F08028 )
                a1[23] = qword_4F08028(a1[23], 2);
              goto LABEL_1021;
            case 4:
            case 5:
              if ( qword_4F08028 )
                a1[23] = qword_4F08028(a1[23], 6);
              goto LABEL_1021;
            case 6:
              if ( qword_4F08028 )
                a1[23] = qword_4F08028(a1[23], 12);
LABEL_1021:
              if ( !qword_4F08020 )
                goto LABEL_693;
              a1[25] = qword_4F08020(a1[25], 83);
              v118 = qword_4F08028;
              break;
            default:
              goto LABEL_37;
          }
          goto LABEL_694;
        case 7:
          v118 = qword_4F08028;
          if ( qword_4F08028 )
          {
            v133 = qword_4F08028(a1[22], 37);
            v118 = qword_4F08028;
            a1[22] = v133;
            if ( v118 )
            {
              v134 = v118(a1[23], 65);
              v118 = qword_4F08028;
              a1[23] = v134;
            }
          }
          if ( (a1[24] & 2) == 0 )
            goto LABEL_827;
          if ( !v118 )
            goto LABEL_697;
          a1[25] = v118(a1[25], 11);
          v118 = qword_4F08028;
          goto LABEL_694;
        case 8:
          v118 = qword_4F08028;
          if ( qword_4F08028 )
          {
            v132 = qword_4F08028(a1[22], 2);
            v118 = qword_4F08028;
            a1[22] = v132;
            if ( v118 )
              goto LABEL_835;
          }
          goto LABEL_697;
        case 9:
          v118 = qword_4F08028;
          if ( !qword_4F08028 )
            goto LABEL_697;
          v131 = qword_4F08028(a1[22], 30);
          v54 = *((char *)a1 + 171) < 0;
          v118 = qword_4F08028;
          a1[22] = v131;
          if ( !v54 )
            goto LABEL_831;
          if ( !v118 )
            goto LABEL_697;
          a1[23] = v118(a1[23], 37);
          v118 = qword_4F08028;
          goto LABEL_694;
        case 0xA:
          if ( qword_4F08020 )
            a1[22] = qword_4F08020(a1[22], 2);
          v118 = qword_4F08028;
          if ( !qword_4F08028 )
            goto LABEL_697;
          v130 = qword_4F08028(a1[23], 2);
          v54 = *((char *)a1 + 171) < 0;
          v118 = qword_4F08028;
          a1[23] = v130;
          if ( v54 )
          {
            if ( !v118 )
              goto LABEL_697;
            a1[25] = v118(a1[25], 37);
            v118 = qword_4F08028;
          }
          else
          {
LABEL_827:
            if ( !v118 )
              goto LABEL_697;
            a1[25] = v118(a1[25], 8);
            v118 = qword_4F08028;
          }
          goto LABEL_694;
        case 0xB:
          v118 = qword_4F08028;
          if ( !qword_4F08028 )
            goto LABEL_697;
          a1[22] = qword_4F08028(a1[22], 2);
          v118 = qword_4F08028;
          goto LABEL_694;
        case 0xC:
          v118 = qword_4F08028;
          switch ( *((_BYTE *)a1 + 176) )
          {
            case 0:
            case 2:
              goto LABEL_694;
            case 1:
              if ( !qword_4F08028 )
                goto LABEL_697;
              a1[23] = qword_4F08028(a1[23], 13);
              v118 = qword_4F08028;
              break;
            case 3:
              if ( qword_4F08028 )
              {
                v138 = qword_4F08028(a1[23], 6);
                v118 = qword_4F08028;
                a1[23] = v138;
              }
              if ( dword_4F08018 )
                a1[24] = 0;
              goto LABEL_694;
            case 4:
            case 0xC:
              goto LABEL_1026;
            case 5:
            case 6:
            case 7:
            case 8:
            case 9:
            case 0xA:
              if ( !qword_4F08028 )
                goto LABEL_697;
              v136 = qword_4F08028(a1[23], 6);
              v118 = qword_4F08028;
              a1[23] = v136;
              if ( !v118 )
                goto LABEL_697;
              a1[24] = v118(a1[24], 13);
              v118 = qword_4F08028;
              break;
            case 0xB:
              if ( qword_4F08028 )
                a1[23] = qword_4F08028(a1[23], 2);
              if ( !qword_4F08020 )
                goto LABEL_693;
              a1[24] = qword_4F08020(a1[24], 48);
              v118 = qword_4F08028;
              break;
            case 0xD:
              if ( !qword_4F08028 )
                goto LABEL_697;
              a1[23] = qword_4F08028(a1[23], 6);
              v118 = qword_4F08028;
              break;
            default:
              goto LABEL_37;
          }
          goto LABEL_694;
        case 0xD:
          v118 = qword_4F08028;
          v129 = a1[22] & 2;
          if ( (a1[22] & 1) != 0 )
          {
            if ( v129 )
            {
              if ( !qword_4F08028 )
                goto LABEL_697;
              a1[23] = qword_4F08028(a1[23], 24);
              v118 = qword_4F08028;
            }
            else
            {
LABEL_831:
              if ( !v118 )
                goto LABEL_697;
              a1[23] = v118(a1[23], 8);
              v118 = qword_4F08028;
            }
          }
          else if ( v129 )
          {
LABEL_1026:
            if ( !v118 )
              goto LABEL_697;
LABEL_835:
            a1[23] = v118(a1[23], 2);
            v118 = qword_4F08028;
          }
LABEL_694:
          if ( v118 )
          {
            v119 = v118(a1[1], 24);
            v118 = qword_4F08028;
            a1[1] = v119;
            if ( v118 )
            {
              v120 = v118(a1[3], 24);
              v118 = qword_4F08028;
              a1[3] = v120;
            }
          }
LABEL_697:
          v121 = dword_4F08018;
          if ( dword_4F08018 )
            a1[4] = 0;
          if ( !v118 )
            goto LABEL_1208;
          a1[5] = v118(a1[5], 23);
          if ( qword_4F08028 )
          {
            a1[6] = qword_4F08028(a1[6], 11);
            if ( qword_4F08028 )
            {
              v122 = qword_4F08028(a1[12], 52);
              v123 = dword_4F08018;
              a1[12] = v122;
              v124 = qword_4F08028;
              if ( v123 )
                *a1 = 0;
              if ( v124 )
              {
                a1[9] = v124(a1[9], 61);
                if ( qword_4F08028 )
                  a1[10] = qword_4F08028(a1[10], 65);
              }
              goto LABEL_707;
            }
            if ( !dword_4F08018 )
            {
LABEL_707:
              LOBYTE(v5) = (_BYTE)qword_4F08020;
              if ( qword_4F08020 )
              {
                v5 = qword_4F08020(a1[13], 75);
                a1[13] = v5;
              }
              goto LABEL_35;
            }
          }
          else
          {
            v121 = dword_4F08018;
LABEL_1208:
            if ( !v121 )
              goto LABEL_707;
          }
          *a1 = 0;
          goto LABEL_707;
        case 0xF:
          v118 = qword_4F08028;
          if ( !qword_4F08028 )
            goto LABEL_697;
          a1[23] = qword_4F08028(a1[23], *((unsigned __int8 *)a1 + 176));
LABEL_693:
          v118 = qword_4F08028;
          goto LABEL_694;
        default:
          goto LABEL_37;
      }
    case 3:
      if ( a4 )
      {
        v107 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                 *a1,
                 3,
                 a3,
                 a4,
                 a5,
                 a3);
        v10 = qword_4F08028;
        *a1 = v107;
      }
      if ( v10 )
      {
        a1[1] = v10(a1[1], 6);
        if ( qword_4F08028 )
        {
          a1[2] = qword_4F08028(a1[2], 6);
          if ( qword_4F08028 )
          {
            a1[3] = qword_4F08028(a1[3], 24);
            if ( qword_4F08028 )
              a1[5] = qword_4F08028(a1[5], 13);
          }
        }
      }
      LODWORD(v5) = dword_4F08018;
      if ( dword_4F08018 )
        a1[6] = 0;
      if ( qword_4F08020 )
      {
        v108 = qword_4F08020(a1[7], 72);
        v109 = qword_4F08028;
        a1[7] = v108;
        if ( !v109 )
          goto LABEL_574;
      }
      else
      {
        v109 = qword_4F08028;
        if ( !qword_4F08028 )
          goto LABEL_577;
      }
      a1[9] = v109(a1[9], 61);
LABEL_574:
      if ( qword_4F08020 )
        a1[8] = qword_4F08020(a1[8], 75);
      LODWORD(v5) = dword_4F08018;
LABEL_577:
      if ( (_DWORD)v5 )
        a1[10] = 0;
      goto LABEL_35;
    case 4:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               3,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        a1[5] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                  a1[5],
                  6,
                  a3,
                  a4,
                  a5,
                  v10);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[6] = qword_4F08028(a1[6], 23);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[7] = qword_4F08028(a1[7], 9);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              v5 = qword_4F08028(a1[1], 11);
              a1[1] = v5;
            }
          }
        }
      }
      goto LABEL_35;
    case 5:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               5,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
        goto LABEL_645;
      goto LABEL_35;
    case 6:
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                  a1[1],
                  24,
                  a3,
                  a4,
                  a5,
                  a3);
        if ( qword_4F08028 )
        {
          v95 = qword_4F08028(a1[3], 24);
          v96 = dword_4F08018;
          a1[3] = v95;
          v97 = qword_4F08028;
          if ( v96 )
            a1[4] = 0;
          if ( v97 )
          {
            a1[5] = v97(a1[5], 23);
            if ( qword_4F08028 )
            {
              a1[6] = qword_4F08028(a1[6], 11);
              if ( qword_4F08028 )
              {
                v98 = qword_4F08028(a1[12], 52);
                v99 = dword_4F08018;
                a1[12] = v98;
                v100 = qword_4F08028;
                if ( v99 )
                  *a1 = 0;
                if ( v100 )
                {
                  a1[9] = v100(a1[9], 61);
                  if ( qword_4F08028 )
                    a1[10] = qword_4F08028(a1[10], 65);
                }
                goto LABEL_528;
              }
              if ( dword_4F08018 )
              {
                *a1 = 0;
                v11 = qword_4F08020;
LABEL_529:
                if ( v11 )
                {
                  a1[13] = v11(a1[13], 75);
                  if ( qword_4F08020 )
                  {
                    a1[14] = qword_4F08020(a1[14], 6);
                    if ( qword_4F08020 )
                      a1[15] = qword_4F08020(a1[15], 5);
                  }
                }
                v101 = qword_4F08028;
                if ( qword_4F08028 )
                {
                  a1[19] = qword_4F08028(a1[19], 7);
                  LOBYTE(v5) = *((_BYTE *)a1 + 140);
                  switch ( (char)v5 )
                  {
                    case 0:
                    case 1:
                    case 3:
                    case 4:
                    case 5:
                    case 17:
                    case 18:
                    case 19:
                    case 20:
                    case 21:
                      goto LABEL_35;
                    case 2:
                      goto LABEL_910;
                    case 6:
                      LOBYTE(v5) = (_BYTE)qword_4F08028;
                      if ( qword_4F08028 )
                        goto LABEL_930;
                      goto LABEL_35;
                    case 7:
                      if ( qword_4F08028 )
                      {
                        a1[20] = qword_4F08028(a1[20], 6);
                        LOBYTE(v5) = (_BYTE)qword_4F08028;
                        if ( qword_4F08028 )
                        {
                          v5 = qword_4F08028(a1[21], 4);
                          a1[21] = v5;
                        }
                      }
                      goto LABEL_35;
                    case 8:
                      v101 = qword_4F08028;
                      goto LABEL_926;
                    case 9:
                    case 10:
                    case 11:
                      goto LABEL_797;
                    case 12:
                      if ( qword_4F08028 )
                      {
                        a1[20] = qword_4F08028(a1[20], 6);
                        LOBYTE(v5) = (_BYTE)qword_4F08028;
                        if ( qword_4F08028 )
                        {
                          v5 = qword_4F08028(a1[21], 78);
                          a1[21] = v5;
                        }
                      }
                      goto LABEL_923;
                    case 13:
                      if ( qword_4F08028 )
                      {
                        a1[20] = qword_4F08028(a1[20], 6);
                        LOBYTE(v5) = (_BYTE)qword_4F08028;
                        if ( qword_4F08028 )
                        {
                          v5 = qword_4F08028(a1[21], 6);
                          a1[21] = v5;
                        }
                      }
                      goto LABEL_35;
                    case 14:
                      if ( qword_4F08028 )
                      {
                        v5 = qword_4F08028(a1[21], 41);
                        a1[21] = v5;
                      }
                      goto LABEL_35;
                    case 15:
                      if ( qword_4F08028 )
                      {
                        a1[20] = qword_4F08028(a1[20], 6);
                        LOBYTE(v5) = (_BYTE)qword_4F08028;
                        if ( qword_4F08028 )
                        {
                          v5 = qword_4F08028(a1[21], 2);
                          a1[21] = v5;
                        }
                      }
                      goto LABEL_35;
                    case 16:
                      if ( qword_4F08028 )
                        goto LABEL_930;
                      goto LABEL_35;
                    default:
                      goto LABEL_37;
                  }
                }
                LOBYTE(v5) = *((_BYTE *)a1 + 140);
                switch ( (char)v5 )
                {
                  case 0:
                  case 1:
                  case 3:
                  case 4:
                  case 5:
                  case 6:
                  case 7:
                  case 13:
                  case 14:
                  case 15:
                  case 16:
                  case 17:
                  case 18:
                  case 19:
                  case 20:
                  case 21:
                    goto LABEL_35;
                  case 2:
LABEL_910:
                    v135 = *((_BYTE *)a1 + 161);
                    if ( (v135 & 8) != 0 )
                    {
                      if ( (v135 & 0x10) != 0 )
                      {
                        LOBYTE(v5) = (_BYTE)qword_4F08028;
                        if ( !qword_4F08028 )
                          goto LABEL_35;
                        a1[21] = qword_4F08028(a1[21], 23);
                      }
                      else if ( qword_4F08020 )
                      {
                        a1[21] = qword_4F08020(a1[21], 2);
                      }
                    }
                    else
                    {
                      LOBYTE(v5) = (_BYTE)qword_4F08028;
                      if ( !qword_4F08028 )
                        goto LABEL_35;
                      a1[21] = qword_4F08028(a1[21], 6);
                    }
                    LOBYTE(v5) = (_BYTE)qword_4F08028;
                    if ( qword_4F08028 )
                    {
                      v5 = qword_4F08028(a1[22], 79);
                      a1[22] = v5;
                    }
                    goto LABEL_35;
                  case 8:
LABEL_926:
                    LOBYTE(v5) = *((_BYTE *)a1 + 169) & 3;
                    if ( (_BYTE)v5 == 1 )
                    {
                      if ( !v101 )
                        goto LABEL_35;
                      v5 = v101(a1[22], 13);
                      a1[22] = v5;
                      v101 = qword_4F08028;
                    }
                    else if ( *((char *)a1 + 168) < 0 )
                    {
                      if ( !v101 )
                        goto LABEL_35;
                      v5 = v101(a1[22], 2);
                      a1[22] = v5;
                      v101 = qword_4F08028;
                    }
                    if ( v101 )
                    {
                      a1[23] = v101(a1[23], 2);
                      LOBYTE(v5) = (_BYTE)qword_4F08028;
                      if ( qword_4F08028 )
                      {
LABEL_930:
                        v5 = qword_4F08028(a1[20], 6);
                        a1[20] = v5;
                      }
                    }
                    break;
                  case 9:
                  case 10:
                  case 11:
LABEL_797:
                    if ( qword_4F08020 )
                      a1[20] = qword_4F08020(a1[20], 8);
                    LOBYTE(v5) = (_BYTE)qword_4F08028;
                    if ( qword_4F08028 )
                    {
                      v5 = qword_4F08028(a1[21], 40);
                      a1[21] = v5;
                    }
                    goto LABEL_35;
                  case 12:
LABEL_923:
                    if ( dword_4F08018 )
                      a1[22] = 0;
                    goto LABEL_35;
                  default:
                    goto LABEL_37;
                }
                goto LABEL_35;
              }
LABEL_528:
              v11 = qword_4F08020;
              goto LABEL_529;
            }
            v96 = dword_4F08018;
          }
          v11 = qword_4F08020;
          if ( !v96 )
            goto LABEL_529;
        }
        else
        {
          if ( !dword_4F08018 )
            goto LABEL_528;
          a1[4] = 0;
          v11 = qword_4F08020;
        }
      }
      else
      {
        if ( !(_DWORD)a5 )
          goto LABEL_529;
        a1[4] = 0;
      }
      *a1 = 0;
      goto LABEL_529;
    case 7:
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                  a1[1],
                  24,
                  a3,
                  a4,
                  a5,
                  a3);
        if ( qword_4F08028 )
        {
          v88 = qword_4F08028(a1[3], 24);
          v89 = dword_4F08018;
          a1[3] = v88;
          v90 = qword_4F08028;
          if ( !v89 )
          {
            if ( !qword_4F08028 )
              goto LABEL_506;
LABEL_499:
            a1[5] = v90(a1[5], 23);
            if ( qword_4F08028 )
            {
              a1[6] = qword_4F08028(a1[6], 11);
              if ( qword_4F08028 )
              {
                v91 = qword_4F08028(a1[12], 52);
                v92 = dword_4F08018;
                a1[12] = v91;
                v93 = qword_4F08028;
                if ( v92 )
                  *a1 = 0;
                if ( v93 )
                {
                  a1[9] = v93(a1[9], 61);
                  if ( qword_4F08028 )
                    a1[10] = qword_4F08028(a1[10], 65);
                }
                goto LABEL_506;
              }
              if ( dword_4F08018 )
              {
                *a1 = 0;
                v11 = qword_4F08020;
                goto LABEL_507;
              }
LABEL_506:
              v11 = qword_4F08020;
              goto LABEL_507;
            }
            v11 = qword_4F08020;
            if ( dword_4F08018 )
              goto LABEL_1182;
LABEL_507:
            if ( v11 )
            {
              a1[13] = v11(a1[13], 75);
              if ( qword_4F08020 )
                a1[14] = qword_4F08020(a1[14], 7);
            }
            if ( qword_4F08028 )
              a1[15] = qword_4F08028(a1[15], 6);
            v94 = *((_BYTE *)a1 + 170);
            if ( (v94 & 1) != 0 )
            {
              if ( qword_4F08028 )
              {
                a1[16] = qword_4F08028(a1[16], 7);
                goto LABEL_515;
              }
            }
            else
            {
              if ( (v94 & 2) != 0 )
              {
                if ( !qword_4F08020 )
                {
                  switch ( *((_BYTE *)a1 + 177) )
                  {
                    case 0:
LABEL_790:
                      if ( (*((_BYTE *)a1 + 172) & 4) == 0 || !qword_4F08028 )
                        goto LABEL_729;
                      goto LABEL_792;
                    case 1:
LABEL_795:
                      if ( !qword_4F08028 )
                        goto LABEL_729;
LABEL_792:
                      a1[23] = qword_4F08028(a1[23], 2);
                      goto LABEL_729;
                    case 2:
                    case 6:
LABEL_727:
                      if ( qword_4F08028 )
                        a1[23] = qword_4F08028(a1[23], 30);
                      goto LABEL_729;
                    case 5:
LABEL_793:
                      if ( qword_4F08028 )
                        a1[23] = qword_4F08028(a1[23], 13);
                      goto LABEL_729;
                    default:
                      goto LABEL_731;
                  }
                }
                a1[16] = qword_4F08020(a1[16], 72);
                goto LABEL_515;
              }
              if ( qword_4F08028 )
              {
                a1[16] = qword_4F08028(a1[16], 3);
LABEL_515:
                switch ( *((_BYTE *)a1 + 177) )
                {
                  case 0:
                    goto LABEL_790;
                  case 1:
                    goto LABEL_795;
                  case 2:
                  case 6:
                    goto LABEL_727;
                  case 5:
                    goto LABEL_793;
                  default:
                    break;
                }
LABEL_729:
                if ( qword_4F08020 )
                  a1[26] = qword_4F08020(a1[26], 72);
LABEL_731:
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( !qword_4F08028 )
                  goto LABEL_744;
                a1[28] = qword_4F08028(a1[28], 26);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( !qword_4F08028 )
                  goto LABEL_744;
                v125 = qword_4F08028(a1[27], 82);
                v126 = a1[20];
                a1[27] = v125;
                v5 = (unsigned __int64)qword_4F08028;
                if ( v126 )
                {
                  if ( !qword_4F08028 )
                    goto LABEL_744;
                  a1[20] = qword_4F08028(v126, 11);
                  v5 = (unsigned __int64)qword_4F08028;
                }
                if ( (*((_BYTE *)a1 + 169) & 8) == 0 )
                {
LABEL_739:
                  if ( v5 )
                  {
                    a1[29] = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(a1[29], 7);
                    LOBYTE(v5) = (_BYTE)qword_4F08028;
                    if ( qword_4F08028 )
                    {
                      a1[30] = qword_4F08028(a1[30], 24);
                      LOBYTE(v5) = (_BYTE)qword_4F08028;
                      if ( qword_4F08028 )
                      {
                        a1[31] = qword_4F08028(a1[31], 7);
                        LOBYTE(v5) = (_BYTE)qword_4F08028;
                        if ( qword_4F08028 )
                        {
                          v5 = qword_4F08028(a1[32], 6);
                          a1[32] = v5;
                        }
                      }
                    }
                  }
                  goto LABEL_744;
                }
                if ( v5 )
                {
                  a1[18] = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(a1[18], 26);
                  v5 = (unsigned __int64)qword_4F08028;
                  goto LABEL_739;
                }
LABEL_744:
                if ( dword_4F08018 )
                  a1[33] = 0;
                goto LABEL_35;
              }
            }
            switch ( *((_BYTE *)a1 + 177) )
            {
              case 0:
                goto LABEL_790;
              default:
                goto LABEL_729;
            }
            goto LABEL_729;
          }
          a1[4] = 0;
          if ( v90 )
            goto LABEL_499;
        }
        else
        {
          if ( !dword_4F08018 )
            goto LABEL_506;
          a1[4] = 0;
        }
        v11 = qword_4F08020;
      }
      else
      {
        if ( !(_DWORD)a5 )
          goto LABEL_507;
        a1[4] = 0;
      }
LABEL_1182:
      *a1 = 0;
      goto LABEL_507;
    case 8:
      if ( !a3 )
      {
        if ( !(_DWORD)a5 )
          goto LABEL_484;
        a1[4] = 0;
        goto LABEL_1138;
      }
      a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                a1[1],
                24,
                a3,
                a4,
                a5,
                a3);
      if ( qword_4F08028 )
      {
        v82 = qword_4F08028(a1[3], 24);
        v83 = dword_4F08018;
        a1[3] = v82;
        v84 = qword_4F08028;
        if ( !v83 )
        {
          if ( !qword_4F08028 )
            goto LABEL_483;
          goto LABEL_476;
        }
        a1[4] = 0;
        if ( v84 )
        {
LABEL_476:
          a1[5] = v84(a1[5], 23);
          if ( qword_4F08028 )
          {
            a1[6] = qword_4F08028(a1[6], 11);
            if ( qword_4F08028 )
            {
              v85 = qword_4F08028(a1[12], 52);
              v86 = dword_4F08018;
              a1[12] = v85;
              v87 = qword_4F08028;
              if ( v86 )
                *a1 = 0;
              if ( v87 )
              {
                a1[9] = v87(a1[9], 61);
                if ( qword_4F08028 )
                  a1[10] = qword_4F08028(a1[10], 65);
              }
              goto LABEL_483;
            }
            if ( dword_4F08018 )
            {
              *a1 = 0;
              v11 = qword_4F08020;
              goto LABEL_484;
            }
LABEL_483:
            v11 = qword_4F08020;
            goto LABEL_484;
          }
          v11 = qword_4F08020;
          if ( !dword_4F08018 )
          {
LABEL_484:
            if ( v11 )
            {
              a1[13] = v11(a1[13], 75);
              if ( qword_4F08020 )
                a1[14] = qword_4F08020(a1[14], 8);
            }
            if ( qword_4F08028 )
            {
              a1[15] = qword_4F08028(a1[15], 6);
              if ( qword_4F08028 )
                a1[19] = qword_4F08028(a1[19], 30);
            }
            if ( qword_4F08020 )
              a1[20] = qword_4F08020(a1[20], 72);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              a1[21] = qword_4F08028(a1[21], 2);
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[23], 6);
                a1[23] = v5;
              }
            }
            goto LABEL_35;
          }
LABEL_1138:
          *a1 = 0;
          goto LABEL_484;
        }
      }
      else
      {
        if ( !dword_4F08018 )
          goto LABEL_483;
        a1[4] = 0;
      }
      v11 = qword_4F08020;
      goto LABEL_1138;
    case 9:
      LOBYTE(v5) = *(_BYTE *)a1;
      if ( (*(_BYTE *)a1 & 0x20) != 0 )
      {
        if ( (_DWORD)a5 )
          a1[1] = 0;
      }
      else if ( (v5 & 0x40) != 0 )
      {
        if ( (_DWORD)a5 )
          a1[1] = 0;
      }
      else if ( (v5 & 1) != 0 )
      {
        if ( a3 )
        {
          v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                 a1[1],
                 2,
                 a3,
                 a4,
                 a5,
                 a3);
          a1[1] = v5;
        }
      }
      else if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               a1[1],
               10,
               a3,
               a4,
               a5,
               a3);
        a1[1] = v5;
      }
      goto LABEL_35;
    case 10:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               10,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
        goto LABEL_645;
      goto LABEL_35;
    case 11:
      if ( !a3 )
      {
        if ( !(_DWORD)a5 )
          goto LABEL_14;
        a1[4] = 0;
        goto LABEL_1194;
      }
      a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                a1[1],
                24,
                a3,
                a4,
                a5,
                a3);
      if ( !qword_4F08028 )
      {
        if ( !dword_4F08018 )
          goto LABEL_13;
        a1[4] = 0;
        goto LABEL_1299;
      }
      v12 = qword_4F08028(a1[3], 24);
      v13 = dword_4F08018;
      a1[3] = v12;
      v14 = qword_4F08028;
      if ( v13 )
      {
        a1[4] = 0;
        if ( v14 )
          goto LABEL_6;
LABEL_1299:
        v11 = qword_4F08020;
LABEL_1194:
        *a1 = 0;
        goto LABEL_14;
      }
      if ( !qword_4F08028 )
        goto LABEL_13;
LABEL_6:
      a1[5] = v14(a1[5], 23);
      if ( qword_4F08028 )
      {
        a1[6] = qword_4F08028(a1[6], 11);
        if ( qword_4F08028 )
        {
          v15 = qword_4F08028(a1[12], 52);
          v16 = dword_4F08018;
          a1[12] = v15;
          v17 = qword_4F08028;
          if ( v16 )
            *a1 = 0;
          if ( v17 )
          {
            a1[9] = v17(a1[9], 61);
            if ( qword_4F08028 )
              a1[10] = qword_4F08028(a1[10], 65);
          }
          goto LABEL_13;
        }
        if ( dword_4F08018 )
        {
          *a1 = 0;
          v11 = qword_4F08020;
          goto LABEL_14;
        }
LABEL_13:
        v11 = qword_4F08020;
        goto LABEL_14;
      }
      v11 = qword_4F08020;
      if ( dword_4F08018 )
        goto LABEL_1194;
LABEL_14:
      if ( v11 && (a1[13] = v11(a1[13], 75), qword_4F08020) )
      {
        a1[14] = qword_4F08020(a1[14], 11);
        v18 = qword_4F08028;
        if ( !qword_4F08028 )
          goto LABEL_18;
      }
      else
      {
        v18 = qword_4F08028;
        if ( !qword_4F08028 )
        {
LABEL_1157:
          if ( (unsigned __int8)(*((_BYTE *)a1 + 174) - 1) <= 1u && dword_4F08018 )
            a1[22] = 0;
          goto LABEL_26;
        }
      }
      a1[19] = v18(a1[19], 6);
LABEL_18:
      if ( qword_4F08020 )
        a1[30] = qword_4F08020(a1[30], 48);
      if ( !qword_4F08028 )
        goto LABEL_1157;
      a1[31] = qword_4F08028(a1[31], 59);
      v19 = qword_4F08028;
      if ( (unsigned __int8)(*((_BYTE *)a1 + 174) - 1) <= 1u && dword_4F08018 )
        a1[22] = 0;
      if ( !v19 )
        goto LABEL_26;
      v20 = v19(a1[40], 11);
      v21 = *((_BYTE *)a1 + 174) == 6;
      a1[40] = v20;
      v22 = qword_4F08028;
      if ( !v21 )
        goto LABEL_24;
      if ( qword_4F08028 )
      {
        a1[22] = qword_4F08028(a1[22], 11);
        v22 = qword_4F08028;
LABEL_24:
        if ( v22 )
          a1[27] = v22(a1[27], 63);
      }
LABEL_26:
      if ( (*((_BYTE *)a1 + 194) & 0x40) != 0
        || qword_4F08020 && (a1[29] = qword_4F08020(a1[29], 38), (*((_BYTE *)a1 + 194) & 0x40) != 0) )
      {
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( !qword_4F08028 )
          goto LABEL_35;
        a1[29] = qword_4F08028(a1[29], 11);
      }
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        a1[33] = qword_4F08028(a1[33], 6);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[34] = qword_4F08028(a1[34], 11);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[35] = qword_4F08028(a1[35], 11);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              a1[32] = qword_4F08028(a1[32], 80);
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[45], 29);
                a1[45] = v5;
              }
            }
          }
        }
      }
LABEL_35:
      qword_4F08028 = v6;
      qword_4F08020 = v7;
      dword_4F08018 = v8;
      return v5;
    case 12:
      if ( !a3 )
      {
        if ( !(_DWORD)a5 )
          goto LABEL_394;
        a1[4] = 0;
        goto LABEL_1169;
      }
      a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                a1[1],
                24,
                a3,
                a4,
                a5,
                a3);
      if ( qword_4F08028 )
      {
        v62 = qword_4F08028(a1[3], 24);
        v63 = dword_4F08018;
        a1[3] = v62;
        v64 = qword_4F08028;
        if ( !v63 )
        {
          if ( !qword_4F08028 )
            goto LABEL_393;
          goto LABEL_386;
        }
        a1[4] = 0;
        if ( v64 )
        {
LABEL_386:
          a1[5] = v64(a1[5], 23);
          if ( qword_4F08028 )
          {
            a1[6] = qword_4F08028(a1[6], 11);
            if ( qword_4F08028 )
            {
              v65 = qword_4F08028(a1[12], 52);
              v66 = dword_4F08018;
              a1[12] = v65;
              v67 = qword_4F08028;
              if ( v66 )
                *a1 = 0;
              if ( v67 )
              {
                a1[9] = v67(a1[9], 61);
                if ( qword_4F08028 )
                  a1[10] = qword_4F08028(a1[10], 65);
              }
              goto LABEL_393;
            }
            if ( dword_4F08018 )
            {
              *a1 = 0;
              v11 = qword_4F08020;
              goto LABEL_394;
            }
LABEL_393:
            v11 = qword_4F08020;
            goto LABEL_394;
          }
          v11 = qword_4F08020;
          if ( !dword_4F08018 )
          {
LABEL_394:
            if ( v11 )
            {
              a1[13] = v11(a1[13], 75);
              if ( qword_4F08020 )
                a1[14] = qword_4F08020(a1[14], 12);
            }
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              v5 = qword_4F08028(a1[16], 21);
              a1[16] = v5;
            }
            goto LABEL_35;
          }
LABEL_1169:
          *a1 = 0;
          goto LABEL_394;
        }
      }
      else
      {
        if ( !dword_4F08018 )
          goto LABEL_393;
        a1[4] = 0;
      }
      v11 = qword_4F08020;
      goto LABEL_1169;
    case 13:
      if ( a3 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                6,
                a3,
                a4,
                a5,
                a3);
        if ( qword_4F08028 )
          a1[1] = qword_4F08028(a1[1], 6);
        a4 = qword_4F08020;
      }
      if ( a4 )
        a1[2] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a4)(
                  a1[2],
                  13,
                  a3,
                  a4,
                  a5,
                  v10);
      if ( dword_4F08018 )
        a1[10] = 0;
      LOBYTE(v5) = *((_BYTE *)a1 + 24);
      switch ( (char)v5 )
      {
        case 0:
        case 16:
        case 19:
        case 24:
          goto LABEL_35;
        case 1:
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
          {
            v5 = qword_4F08020(a1[9], 13);
            a1[9] = v5;
          }
          goto LABEL_35;
        case 2:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[7] = qword_4F08028(a1[7], 2);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
              goto LABEL_1012;
          }
          goto LABEL_35;
        case 3:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[7] = qword_4F08028(a1[7], 7);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
              goto LABEL_1012;
          }
          goto LABEL_35;
        case 4:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[7] = qword_4F08028(a1[7], 8);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
              goto LABEL_1012;
          }
          goto LABEL_35;
        case 5:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[7] = qword_4F08028(a1[7], 30);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
              goto LABEL_1279;
          }
          goto LABEL_35;
        case 6:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[8] = qword_4F08028(a1[8], 73);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
              goto LABEL_984;
          }
          goto LABEL_35;
        case 7:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[7], 49);
            a1[7] = v5;
          }
          goto LABEL_35;
        case 8:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[7], 50);
            a1[7] = v5;
          }
          goto LABEL_35;
        case 9:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[7], 51);
            a1[7] = v5;
          }
          goto LABEL_35;
        case 10:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[7] = qword_4F08028(a1[7], 13);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              v5 = qword_4F08028(a1[8], 22);
              a1[8] = v5;
            }
          }
          goto LABEL_35;
        case 11:
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
            goto LABEL_1018;
          goto LABEL_35;
        case 12:
        case 14:
        case 15:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( *((_BYTE *)a1 + 56) )
          {
            if ( !qword_4F08028 )
              goto LABEL_35;
            goto LABEL_1279;
          }
          if ( !qword_4F08028 )
            goto LABEL_35;
          goto LABEL_1281;
        case 13:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( *((_BYTE *)a1 + 57) )
          {
            if ( qword_4F08028 )
            {
              v5 = qword_4F08028(a1[8], 59);
              a1[8] = v5;
            }
            goto LABEL_35;
          }
          if ( *((_BYTE *)a1 + 56) )
          {
            if ( qword_4F08028 )
            {
LABEL_1279:
              v5 = qword_4F08028(a1[8], 6);
              a1[8] = v5;
            }
            goto LABEL_35;
          }
          if ( !qword_4F08028 )
            goto LABEL_35;
          break;
        case 17:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[7], 21);
            a1[7] = v5;
          }
          goto LABEL_35;
        case 18:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
            goto LABEL_984;
          goto LABEL_35;
        case 20:
          v137 = a1[7];
          if ( !v137 )
            goto LABEL_1011;
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[7] = qword_4F08028(v137, 11);
LABEL_1011:
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
LABEL_1012:
              v5 = qword_4F08028(a1[8], 65);
              a1[8] = v5;
            }
          }
          goto LABEL_35;
        case 21:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[7], 7);
            a1[7] = v5;
          }
          goto LABEL_35;
        case 22:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[7] = qword_4F08028(a1[7], 6);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
              goto LABEL_1012;
          }
          goto LABEL_35;
        case 23:
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
            goto LABEL_850;
          goto LABEL_35;
        case 25:
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
            goto LABEL_1018;
          goto LABEL_35;
        case 26:
          if ( qword_4F08020 )
            a1[7] = qword_4F08020(a1[7], 13);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
            goto LABEL_1281;
          goto LABEL_35;
        case 27:
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
            goto LABEL_1018;
          goto LABEL_35;
        case 28:
        case 29:
          if ( qword_4F08028 )
            a1[7] = qword_4F08028(a1[7], 13);
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
          {
LABEL_850:
            v5 = qword_4F08020(a1[8], 13);
            a1[8] = v5;
          }
          goto LABEL_35;
        case 30:
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
            goto LABEL_1018;
          goto LABEL_35;
        case 31:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
LABEL_984:
            v5 = qword_4F08028(a1[7], 30);
            a1[7] = v5;
          }
          goto LABEL_35;
        case 32:
          if ( qword_4F08028 )
            a1[7] = qword_4F08028(a1[7], 59);
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
          {
            v5 = qword_4F08020(a1[8], 48);
            a1[8] = v5;
          }
          goto LABEL_35;
        case 33:
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
          {
            a1[7] = qword_4F08020(a1[7], 13);
            LOBYTE(v5) = (_BYTE)qword_4F08020;
            if ( qword_4F08020 )
            {
              v5 = qword_4F08020(a1[8], 3);
              a1[8] = v5;
            }
          }
          goto LABEL_35;
        case 34:
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
          {
LABEL_1018:
            v5 = qword_4F08020(a1[7], 13);
            a1[7] = v5;
          }
          goto LABEL_35;
        case 35:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
            goto LABEL_946;
          goto LABEL_35;
        case 36:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
LABEL_946:
            v5 = qword_4F08028(a1[7], 13);
            a1[7] = v5;
          }
          goto LABEL_35;
        case 37:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[7], 59);
            a1[7] = v5;
          }
          goto LABEL_35;
        default:
          goto LABEL_37;
      }
      goto LABEL_1281;
    case 14:
      if ( a3 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                21,
                a3,
                a4,
                a5,
                a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[1] = qword_4F08028(a1[1], 13);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[2], 23);
            a1[2] = v5;
          }
        }
      }
      goto LABEL_35;
    case 15:
      if ( a3 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                21,
                a3,
                a4,
                a5,
                a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[1] = qword_4F08028(a1[1], 7);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[2] = qword_4F08028(a1[2], 7);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              a1[3] = qword_4F08028(a1[3], 23);
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[4] = qword_4F08028(a1[4], 23);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                {
                  a1[5] = qword_4F08028(a1[5], 7);
                  LOBYTE(v5) = (_BYTE)qword_4F08028;
                  if ( qword_4F08028 )
                  {
                    a1[6] = qword_4F08028(a1[6], 7);
                    LOBYTE(v5) = (_BYTE)qword_4F08028;
                    if ( qword_4F08028 )
                    {
                      a1[7] = qword_4F08028(a1[7], 13);
                      LOBYTE(v5) = (_BYTE)qword_4F08028;
                      if ( qword_4F08028 )
                      {
LABEL_1281:
                        v5 = qword_4F08028(a1[8], 13);
                        a1[8] = v5;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      goto LABEL_35;
    case 16:
      if ( !a3 )
        goto LABEL_1124;
      *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
              *a1,
              21,
              a3,
              a4,
              a5,
              a3);
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 && (a1[1] = qword_4F08028(a1[1], 2), LOBYTE(v5) = (_BYTE)qword_4F08028, qword_4F08028) )
      {
        v61 = qword_4F08028(a1[2], 2);
        a4 = qword_4F08020;
        a1[2] = v61;
        if ( !a4 )
          goto LABEL_370;
      }
      else
      {
        a4 = qword_4F08020;
LABEL_1124:
        if ( !a4 )
          goto LABEL_35;
      }
      a1[3] = a4(a1[3], 16);
LABEL_370:
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        v5 = qword_4F08028(a1[4], 16);
        a1[4] = v5;
      }
      goto LABEL_35;
    case 17:
      if ( a4 )
      {
        v115 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                 *a1,
                 16,
                 a3,
                 a4,
                 a5,
                 a3);
        a3 = qword_4F08028;
        *a1 = v115;
        if ( !a3 )
          goto LABEL_664;
      }
      else if ( !a3 )
      {
        goto LABEL_35;
      }
      a1[1] = a3(a1[1], 16);
LABEL_664:
      LOBYTE(v5) = (_BYTE)qword_4F08020;
      if ( qword_4F08020 )
      {
        v5 = qword_4F08020(a1[2], 16);
        a1[2] = v5;
      }
      goto LABEL_35;
    case 18:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               18,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        a1[2] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                  a1[2],
                  7,
                  a3,
                  a4,
                  a5,
                  v10);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[3] = qword_4F08028(a1[3], 21);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[4], 30);
            a1[4] = v5;
          }
        }
      }
      goto LABEL_35;
    case 19:
      if ( a3 )
      {
        v114 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                 a1[1],
                 21,
                 a3,
                 a4,
                 a5,
                 a3);
        a4 = qword_4F08020;
        a1[1] = v114;
        if ( !a4 )
          goto LABEL_653;
      }
      else if ( !a4 )
      {
        goto LABEL_35;
      }
      a1[2] = a4(a1[2], 18);
LABEL_653:
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        v5 = qword_4F08028(a1[3], 22);
        a1[3] = v5;
      }
      goto LABEL_35;
    case 20:
      if ( (a1[3] & 2) != 0 )
      {
        if ( !a3 )
          goto LABEL_35;
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                  a1[1],
                  23,
                  a3,
                  a4,
                  a5,
                  a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( !qword_4F08028 )
          goto LABEL_35;
      }
      else
      {
        if ( !a3 )
          goto LABEL_35;
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                  a1[1],
                  23,
                  a3,
                  a4,
                  a5,
                  a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( !qword_4F08028 )
          goto LABEL_35;
      }
      v5 = qword_4F08028(a1[2], 22);
      a1[2] = v5;
      goto LABEL_35;
    case 21:
      if ( a4 )
      {
        v117 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                 a1[2],
                 21,
                 a3,
                 a4,
                 a5,
                 a3);
        a3 = qword_4F08028;
        a1[2] = v117;
        if ( !a3 )
        {
LABEL_678:
          if ( qword_4F08020 )
            a1[4] = qword_4F08020(a1[4], 75);
          if ( qword_4F08028 )
          {
            a1[7] = qword_4F08028(a1[7], 52);
            if ( qword_4F08028 )
              a1[6] = qword_4F08028(a1[6], 13);
          }
LABEL_683:
          LOBYTE(v5) = *((_BYTE *)a1 + 40);
          switch ( (char)v5 )
          {
            case 0:
            case 10:
            case 23:
            case 24:
              goto LABEL_35;
            case 1:
            case 3:
            case 4:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[9] = qword_4F08028(a1[9], 21);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                {
                  v5 = qword_4F08028(a1[10], 21);
                  a1[10] = v5;
                }
              }
              goto LABEL_35;
            case 2:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[9], 84);
                a1[9] = v5;
              }
              goto LABEL_35;
            case 5:
            case 12:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
                goto LABEL_896;
              goto LABEL_35;
            case 6:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[9] = qword_4F08028(a1[9], 12);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                  goto LABEL_875;
              }
              goto LABEL_35;
            case 7:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[9] = qword_4F08028(a1[9], 12);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                {
LABEL_875:
                  v5 = qword_4F08028(a1[10], 22);
                  a1[10] = v5;
                }
              }
              goto LABEL_35;
            case 8:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
                goto LABEL_884;
              goto LABEL_35;
            case 9:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[9], 81);
                a1[9] = v5;
              }
              goto LABEL_35;
            case 11:
              if ( qword_4F08028 )
                a1[10] = qword_4F08028(a1[10], 20);
              LOBYTE(v5) = (_BYTE)qword_4F08020;
              if ( qword_4F08020 )
              {
                v5 = qword_4F08020(a1[9], 21);
                a1[9] = v5;
              }
              goto LABEL_35;
            case 13:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[10] = qword_4F08028(a1[10], 14);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                  goto LABEL_896;
              }
              goto LABEL_35;
            case 14:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[10] = qword_4F08028(a1[10], 15);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                {
LABEL_896:
                  v5 = qword_4F08028(a1[9], 21);
                  a1[9] = v5;
                }
              }
              goto LABEL_35;
            case 15:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[9] = qword_4F08028(a1[9], 21);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                {
                  v5 = qword_4F08028(a1[10], 16);
                  a1[10] = v5;
                }
              }
              goto LABEL_35;
            case 16:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[9] = qword_4F08028(a1[9], 21);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                {
                  v5 = qword_4F08028(a1[10], 17);
                  a1[10] = v5;
                }
              }
              goto LABEL_35;
            case 17:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
                goto LABEL_884;
              goto LABEL_35;
            case 18:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[9], 43);
                a1[9] = v5;
              }
              goto LABEL_35;
            case 19:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[9], 19);
                a1[9] = v5;
              }
              goto LABEL_35;
            case 20:
              LOBYTE(v5) = (_BYTE)qword_4F08020;
              if ( qword_4F08020 )
              {
                v5 = qword_4F08020(a1[9], 72);
                a1[9] = v5;
              }
              goto LABEL_35;
            case 21:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[9], 32);
                a1[9] = v5;
              }
              goto LABEL_35;
            case 22:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( *((_BYTE *)a1 + 72) )
              {
                if ( qword_4F08028 )
                {
                  v5 = qword_4F08028(a1[10], 6);
                  a1[10] = v5;
                }
              }
              else if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[10], 7);
                a1[10] = v5;
              }
              goto LABEL_35;
            case 25:
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
LABEL_884:
                v5 = qword_4F08028(a1[9], 30);
                a1[9] = v5;
              }
              goto LABEL_35;
            default:
              goto LABEL_37;
          }
        }
      }
      else if ( !a3 )
      {
        goto LABEL_683;
      }
      a1[3] = a3(a1[3], 21);
      goto LABEL_678;
    case 22:
      if ( a3 )
      {
        v116 = ((__int64 (__fastcall *)(_QWORD, _QWORD, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                 a1[2],
                 *((unsigned __int8 *)a1 + 8),
                 a3,
                 a4,
                 a5,
                 a3);
        v11 = qword_4F08020;
        a1[2] = v116;
        if ( !v11 )
          goto LABEL_669;
      }
      else if ( !a4 )
      {
        goto LABEL_35;
      }
      a1[3] = v11(a1[3], 30);
LABEL_669:
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        a1[4] = qword_4F08028(a1[4], 22);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          v5 = qword_4F08028(a1[5], 30);
          a1[5] = v5;
        }
      }
      if ( qword_4F08020 )
      {
        a1[6] = qword_4F08020(a1[6], 22);
        LOBYTE(v5) = (_BYTE)qword_4F08020;
        if ( qword_4F08020 )
        {
          v5 = qword_4F08020(a1[7], 22);
          a1[7] = v5;
        }
      }
      goto LABEL_35;
    case 23:
      v75 = *((_BYTE *)a1 + 28);
      if ( a4 )
      {
        v76 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                23,
                a3,
                a4,
                a5,
                a3);
        v10 = qword_4F08028;
        *a1 = v76;
        a3 = v10;
      }
      if ( a3 )
      {
        v77 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                a1[1],
                23,
                a3,
                a4,
                a5,
                v10);
        v78 = qword_4F08028;
        a1[1] = v77;
        if ( v78 )
        {
          a1[2] = v78(a1[2], 23);
          switch ( v75 )
          {
            case 0:
            case 8:
              goto LABEL_760;
            case 1:
            case 6:
            case 16:
              v10 = qword_4F08028;
              goto LABEL_758;
            case 2:
              v10 = qword_4F08028;
              goto LABEL_905;
            case 3:
              v10 = qword_4F08028;
              goto LABEL_902;
            case 15:
              v10 = qword_4F08028;
              goto LABEL_908;
            case 17:
              goto LABEL_851;
            default:
              goto LABEL_37;
          }
        }
        switch ( v75 )
        {
          case 0:
          case 1:
          case 2:
          case 3:
          case 6:
          case 8:
          case 15:
          case 16:
            goto LABEL_762;
          case 17:
            goto LABEL_851;
          default:
            goto LABEL_37;
        }
      }
      switch ( v75 )
      {
        case 0:
        case 8:
          goto LABEL_762;
        case 1:
        case 6:
        case 16:
LABEL_758:
          if ( !v10 )
            goto LABEL_762;
          a1[4] = v10(a1[4], 6);
          break;
        case 2:
LABEL_905:
          if ( !v10 )
            goto LABEL_762;
          a1[4] = v10(a1[4], 18);
          break;
        case 3:
LABEL_902:
          if ( !v10 )
            goto LABEL_762;
          a1[4] = v10(a1[4], 28);
          break;
        case 15:
LABEL_908:
          if ( !v10 )
            goto LABEL_762;
          a1[4] = v10(a1[4], 21);
          break;
        case 17:
LABEL_851:
          if ( qword_4F08020 )
          {
            a1[5] = qword_4F08020(a1[5], 7);
            if ( qword_4F08020 )
              a1[6] = qword_4F08020(a1[6], 42);
          }
          if ( !qword_4F08028 )
            goto LABEL_762;
          a1[7] = qword_4F08028(a1[7], 22);
          if ( !qword_4F08028 )
            goto LABEL_762;
          a1[8] = qword_4F08028(a1[8], 7);
          if ( !qword_4F08028 )
            goto LABEL_762;
          a1[9] = qword_4F08028(a1[9], 7);
          break;
        default:
          goto LABEL_37;
      }
LABEL_760:
      if ( qword_4F08028 )
        a1[11] = qword_4F08028(a1[11], 22);
LABEL_762:
      if ( qword_4F08020 )
      {
        a1[12] = qword_4F08020(a1[12], 2);
        if ( qword_4F08020 )
        {
          a1[13] = qword_4F08020(a1[13], 6);
          if ( qword_4F08020 )
          {
            a1[14] = qword_4F08020(a1[14], 7);
            if ( qword_4F08020 )
            {
              a1[18] = qword_4F08020(a1[18], 11);
              if ( qword_4F08020 )
              {
                a1[15] = qword_4F08020(a1[15], 7);
                if ( qword_4F08020 )
                {
                  a1[17] = qword_4F08020(a1[17], 12);
                  if ( qword_4F08020 )
                  {
                    a1[20] = qword_4F08020(a1[20], 23);
                    if ( qword_4F08020 )
                    {
                      a1[21] = qword_4F08020(a1[21], 28);
                      if ( qword_4F08020 )
                      {
                        a1[22] = qword_4F08020(a1[22], 29);
                        if ( qword_4F08020 )
                        {
                          a1[23] = qword_4F08020(a1[23], 29);
                          if ( qword_4F08020 )
                          {
                            a1[19] = qword_4F08020(a1[19], 43);
                            if ( qword_4F08020 )
                            {
                              a1[24] = qword_4F08020(a1[24], 30);
                              if ( qword_4F08020 )
                              {
                                a1[25] = qword_4F08020(a1[25], 31);
                                if ( qword_4F08020 )
                                {
                                  a1[26] = qword_4F08020(a1[26], 32);
                                  if ( qword_4F08020 )
                                  {
                                    a1[27] = qword_4F08020(a1[27], 68);
                                    if ( qword_4F08020 )
                                    {
                                      a1[28] = qword_4F08020(a1[28], 71);
                                      if ( qword_4F08020 )
                                      {
                                        a1[29] = qword_4F08020(a1[29], 58);
                                        if ( qword_4F08020 )
                                          a1[34] = qword_4F08020(a1[34], 59);
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      v127 = qword_4F08028;
      if ( v75 != 17 )
        goto LABEL_782;
      if ( qword_4F08028 )
      {
        v140 = qword_4F08028(a1[4], 11);
        v127 = qword_4F08028;
        a1[4] = v140;
LABEL_782:
        if ( v127 )
          a1[10] = v127(a1[10], 21);
      }
      LOBYTE(v5) = (_BYTE)qword_4F08020;
      if ( qword_4F08020 )
      {
        a1[32] = qword_4F08020(a1[32], 52);
        LOBYTE(v5) = (_BYTE)qword_4F08020;
        if ( qword_4F08020 )
        {
          v5 = qword_4F08020(a1[33], 55);
          a1[33] = v5;
        }
      }
      goto LABEL_35;
    case 27:
    case 77:
      goto LABEL_35;
    case 28:
      if ( !a3 )
      {
        if ( !(_DWORD)a5 )
          goto LABEL_424;
        a1[4] = 0;
        goto LABEL_1155;
      }
      a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                a1[1],
                24,
                a3,
                a4,
                a5,
                a3);
      if ( qword_4F08028 )
      {
        v69 = qword_4F08028(a1[3], 24);
        v70 = dword_4F08018;
        a1[3] = v69;
        v71 = qword_4F08028;
        if ( !v70 )
        {
          if ( !qword_4F08028 )
            goto LABEL_423;
          goto LABEL_416;
        }
        a1[4] = 0;
        if ( v71 )
        {
LABEL_416:
          a1[5] = v71(a1[5], 23);
          if ( qword_4F08028 )
          {
            a1[6] = qword_4F08028(a1[6], 11);
            if ( qword_4F08028 )
            {
              v72 = qword_4F08028(a1[12], 52);
              v73 = dword_4F08018;
              a1[12] = v72;
              v74 = qword_4F08028;
              if ( v73 )
                *a1 = 0;
              if ( v74 )
              {
                a1[9] = v74(a1[9], 61);
                if ( qword_4F08028 )
                  a1[10] = qword_4F08028(a1[10], 65);
              }
              goto LABEL_423;
            }
            if ( dword_4F08018 )
            {
              *a1 = 0;
              v11 = qword_4F08020;
              goto LABEL_424;
            }
LABEL_423:
            v11 = qword_4F08020;
            goto LABEL_424;
          }
          v11 = qword_4F08020;
          if ( !dword_4F08018 )
          {
LABEL_424:
            if ( v11 )
            {
              a1[13] = v11(a1[13], 75);
              if ( qword_4F08020 )
                a1[14] = qword_4F08020(a1[14], 28);
            }
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( (*((_BYTE *)a1 + 124) & 1) != 0 )
            {
              if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[16], 28);
                a1[16] = v5;
              }
            }
            else if ( qword_4F08028 )
            {
              v5 = qword_4F08028(a1[16], 23);
              a1[16] = v5;
            }
            goto LABEL_35;
          }
LABEL_1155:
          *a1 = 0;
          goto LABEL_424;
        }
      }
      else
      {
        if ( !dword_4F08018 )
          goto LABEL_423;
        a1[4] = 0;
      }
      v11 = qword_4F08020;
      goto LABEL_1155;
    case 29:
      if ( a4 )
      {
        v113 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                 *a1,
                 29,
                 a3,
                 a4,
                 a5,
                 a3);
        a3 = qword_4F08028;
        *a1 = v113;
        if ( !a3 )
          goto LABEL_614;
      }
      else if ( !a3 )
      {
        goto LABEL_35;
      }
      a1[3] = a3(a1[3], *((unsigned __int8 *)a1 + 16));
LABEL_614:
      if ( qword_4F08020 )
        a1[4] = qword_4F08020(a1[4], 75);
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( (a1[5] & 2) != 0 )
      {
        if ( !qword_4F08028 )
          goto LABEL_35;
        a1[6] = qword_4F08028(a1[6], 6);
      }
      else
      {
        if ( !qword_4F08028 )
          goto LABEL_35;
        a1[6] = qword_4F08028(a1[6], 28);
      }
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        a1[8] = qword_4F08028(a1[8], 52);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          v5 = qword_4F08028(a1[9], 29);
          a1[9] = v5;
        }
      }
      goto LABEL_35;
    case 30:
      if ( a4 )
      {
        v112 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                 *a1,
                 30,
                 a3,
                 a4,
                 a5,
                 a3);
        a3 = qword_4F08028;
        *a1 = v112;
      }
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                  a1[1],
                  7,
                  a3,
                  a4,
                  a5,
                  v10);
        if ( qword_4F08028 )
        {
          a1[2] = qword_4F08028(a1[2], 11);
          if ( qword_4F08028 )
            a1[3] = qword_4F08028(a1[3], 22);
        }
      }
      if ( qword_4F08020 )
        a1[4] = qword_4F08020(a1[4], 30);
      if ( !qword_4F08028 )
      {
        LOBYTE(v5) = *((_BYTE *)a1 + 48);
        switch ( (char)v5 )
        {
          case 0:
          case 1:
          case 2:
          case 3:
          case 4:
          case 6:
          case 7:
          case 8:
          case 9:
LABEL_1144:
            if ( !dword_4F08018 )
              goto LABEL_35;
            a1[10] = 0;
            a1[11] = 0;
            a1[12] = 0;
            break;
          case 5:
LABEL_804:
            v128 = a1[7];
            if ( v128 && qword_4F08028 )
              a1[7] = qword_4F08028(v128, 11);
            if ( qword_4F08020 )
            {
              a1[8] = qword_4F08020(a1[8], 13);
              v5 = (unsigned __int64)qword_4F08028;
            }
            else
            {
LABEL_752:
              v5 = (unsigned __int64)qword_4F08028;
            }
            goto LABEL_711;
          default:
            goto LABEL_37;
        }
        goto LABEL_716;
      }
      a1[5] = qword_4F08028(a1[5], 22);
      LOBYTE(v5) = *((_BYTE *)a1 + 48);
      switch ( (char)v5 )
      {
        case 0:
        case 1:
          goto LABEL_752;
        case 2:
        case 6:
        case 9:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( !qword_4F08028 )
            goto LABEL_1144;
          a1[7] = qword_4F08028(a1[7], 2);
          v5 = (unsigned __int64)qword_4F08028;
          break;
        case 3:
        case 4:
        case 7:
          if ( !qword_4F08028 )
            goto LABEL_1144;
          a1[7] = qword_4F08028(a1[7], 13);
          v5 = (unsigned __int64)qword_4F08028;
          break;
        case 5:
          goto LABEL_804;
        case 8:
          if ( !qword_4F08028 )
            goto LABEL_1144;
          a1[7] = qword_4F08028(a1[7], 2);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( !qword_4F08028 )
            goto LABEL_1144;
          a1[8] = qword_4F08028(a1[8], 73);
          v5 = (unsigned __int64)qword_4F08028;
          break;
        default:
          goto LABEL_37;
      }
LABEL_711:
      if ( dword_4F08018 )
      {
        a1[10] = 0;
        a1[11] = 0;
        a1[12] = 0;
        if ( !v5 )
          goto LABEL_716;
      }
      else if ( !v5 )
      {
        goto LABEL_35;
      }
      a1[13] = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(a1[13], 22);
      if ( qword_4F08028 )
        a1[14] = qword_4F08028(a1[14], 30);
      LOBYTE(v5) = dword_4F08018;
      if ( !dword_4F08018 )
        goto LABEL_35;
LABEL_716:
      a1[15] = 0;
      goto LABEL_35;
    case 31:
      if ( a4 )
      {
        v111 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                 *a1,
                 31,
                 a3,
                 a4,
                 a5,
                 a3);
        v10 = qword_4F08028;
        *a1 = v111;
        a3 = v10;
      }
      if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
               a1[1],
               7,
               a3,
               a4,
               a5,
               v10);
        v10 = qword_4F08028;
        a1[1] = v5;
        LOBYTE(v5) = *((_BYTE *)a1 + 16);
        if ( (_BYTE)v5 != 5 )
        {
          if ( (unsigned __int8)v5 > 5u )
          {
            if ( (_BYTE)v5 != 6 )
              goto LABEL_599;
          }
          else
          {
            v10 = qword_4F08028;
            if ( (_BYTE)v5 == 1 )
            {
LABEL_597:
              if ( !v10 )
                goto LABEL_35;
              a1[3] = v10(a1[3], 2);
              goto LABEL_599;
            }
            if ( (_BYTE)v5 != 2 )
              goto LABEL_599;
          }
          v10 = qword_4F08028;
          goto LABEL_1055;
        }
      }
      else
      {
        LOBYTE(v5) = *((_BYTE *)a1 + 16);
        if ( (_BYTE)v5 != 5 )
        {
          if ( (unsigned __int8)v5 > 5u )
          {
            if ( (_BYTE)v5 != 6 )
              goto LABEL_35;
          }
          else
          {
            if ( (_BYTE)v5 == 1 )
              goto LABEL_597;
            if ( (_BYTE)v5 != 2 )
              goto LABEL_35;
          }
LABEL_1055:
          if ( !v10 )
            goto LABEL_35;
          a1[3] = v10(a1[3], 30);
LABEL_599:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[4], 22);
            a1[4] = v5;
          }
          goto LABEL_35;
        }
      }
      if ( !v10 )
        goto LABEL_35;
      a1[3] = v10(a1[3], 13);
      goto LABEL_599;
    case 32:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               32,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( !a3 )
        goto LABEL_35;
      v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
             a1[1],
             6,
             a3,
             a4,
             a5,
             v10);
      v110 = a1[2];
      a1[1] = v5;
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( v110 )
      {
        if ( !qword_4F08028 )
          goto LABEL_35;
        a1[2] = qword_4F08028(v110, 13);
      }
      else
      {
        if ( !qword_4F08028 )
          goto LABEL_35;
        a1[3] = qword_4F08028(a1[3], 32);
      }
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        v5 = qword_4F08028(a1[6], 7);
        a1[6] = v5;
      }
      goto LABEL_35;
    case 33:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               33,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                  a1[1],
                  11,
                  a3,
                  a4,
                  a5,
                  v10);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
          goto LABEL_688;
      }
      goto LABEL_35;
    case 34:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               34,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                  a1[1],
                  11,
                  a3,
                  a4,
                  a5,
                  v10);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[2] = qword_4F08028(a1[2], 11);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[3] = qword_4F08028(a1[3], 37);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              v5 = qword_4F08028(a1[4], 37);
              a1[4] = v5;
            }
          }
        }
      }
      goto LABEL_35;
    case 35:
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                  a1[1],
                  35,
                  a3,
                  a4,
                  a5,
                  a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          *a1 = qword_4F08028(*a1, 35);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
            goto LABEL_688;
        }
      }
      goto LABEL_35;
    case 36:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               36,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        a1[2] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                  a1[2],
                  35,
                  a3,
                  a4,
                  a5,
                  v10);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          v5 = qword_4F08028(a1[1], 35);
          a1[1] = v5;
        }
      }
      goto LABEL_35;
    case 37:
      if ( a4 )
      {
        v52 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                37,
                a3,
                a4,
                a5,
                a3);
        a3 = qword_4F08028;
        *a1 = v52;
      }
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                  a1[1],
                  37,
                  a3,
                  a4,
                  a5,
                  v10);
        if ( qword_4F08028 )
        {
          a1[2] = qword_4F08028(a1[2], 37);
          if ( qword_4F08028 )
            a1[3] = qword_4F08028(a1[3], 37);
        }
      }
      if ( qword_4F08020 )
        a1[4] = qword_4F08020(a1[4], 75);
      if ( qword_4F08028 )
      {
        a1[5] = qword_4F08028(a1[5], 6);
        if ( qword_4F08028 )
        {
          a1[6] = qword_4F08028(a1[6], 6);
          if ( qword_4F08028 )
            a1[7] = qword_4F08028(a1[7], 6);
        }
      }
      LODWORD(v5) = dword_4F08018;
      if ( dword_4F08018 )
        a1[8] = 0;
      if ( qword_4F08020 )
      {
        v53 = qword_4F08020(a1[14], 36);
        v54 = *((char *)a1 + 96) < 0;
        a1[14] = v53;
        if ( !v54 )
        {
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
          {
            v5 = qword_4F08020(a1[15], 34);
            a1[15] = v5;
          }
          goto LABEL_35;
        }
        LODWORD(v5) = dword_4F08018;
      }
      else if ( *((char *)a1 + 96) >= 0 )
      {
        goto LABEL_35;
      }
      if ( (_DWORD)v5 )
        a1[15] = 0;
      goto LABEL_35;
    case 38:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               38,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
LABEL_645:
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
               a1[1],
               6,
               a3,
               a4,
               a5,
               v10);
        a1[1] = v5;
      }
      goto LABEL_35;
    case 39:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               39,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
               a1[1],
               11,
               a3,
               a4,
               a5,
               v10);
        a1[1] = v5;
      }
      goto LABEL_35;
    case 40:
      if ( a3 )
      {
        v47 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                a1[20],
                59,
                a3,
                a4,
                a5,
                a3);
        a4 = qword_4F08020;
        a1[20] = v47;
      }
      if ( a4 )
      {
        a1[21] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a4)(
                   a1[21],
                   48,
                   a3,
                   a4,
                   a5,
                   v10);
        if ( qword_4F08020 )
        {
          a1[22] = qword_4F08020(a1[22], 48);
          if ( qword_4F08020 )
          {
            a1[16] = qword_4F08020(a1[16], 38);
            if ( qword_4F08020 )
              *a1 = qword_4F08020(*a1, 37);
          }
        }
      }
      if ( qword_4F08028 )
      {
        a1[1] = qword_4F08028(a1[1], 37);
        if ( qword_4F08028 )
        {
          a1[2] = qword_4F08028(a1[2], 37);
          if ( qword_4F08028 )
          {
            a1[3] = qword_4F08028(a1[3], 37);
            if ( qword_4F08028 )
            {
              a1[15] = qword_4F08028(a1[15], 8);
              if ( qword_4F08028 )
              {
                a1[19] = qword_4F08028(a1[19], 23);
                if ( qword_4F08028 )
                  a1[10] = qword_4F08028(a1[10], 37);
              }
            }
          }
        }
      }
      if ( qword_4F08020 )
      {
        a1[8] = qword_4F08020(a1[8], 33);
        if ( qword_4F08020 )
        {
          a1[17] = qword_4F08020(a1[17], 39);
          if ( qword_4F08020 )
            a1[18] = qword_4F08020(a1[18], 38);
        }
      }
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        v48 = qword_4F08028(a1[23], 11);
        v49 = dword_4F08018;
        a1[23] = v48;
        v5 = (unsigned __int64)qword_4F08028;
        if ( v49 )
        {
          a1[24] = 0;
          a1[25] = 0;
          if ( !v5 )
          {
            a1[27] = 0;
            goto LABEL_35;
          }
        }
        else if ( !qword_4F08028 )
        {
          goto LABEL_35;
        }
        v50 = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(a1[26], 6);
        v51 = dword_4F08018;
        a1[26] = v50;
        v5 = (unsigned __int64)qword_4F08028;
        if ( v51 )
          a1[27] = 0;
        if ( (*((_BYTE *)a1 + 110) & 8) != 0 )
        {
          if ( !v5 )
            goto LABEL_35;
          a1[30] = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(a1[30], 7);
        }
        else if ( (*((_BYTE *)a1 + 110) & 0x10) != 0 )
        {
          if ( !v5 )
            goto LABEL_35;
          a1[30] = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(a1[30], 8);
        }
        else
        {
          if ( !v5 )
            goto LABEL_35;
          a1[30] = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(a1[30], 11);
        }
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          v5 = qword_4F08028(a1[32], 6);
          a1[32] = v5;
        }
      }
      else if ( dword_4F08018 )
      {
        a1[24] = 0;
        a1[25] = 0;
        a1[27] = 0;
      }
      goto LABEL_35;
    case 41:
      if ( a3 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                6,
                a3,
                a4,
                a5,
                a3);
        if ( qword_4F08028 )
          a1[1] = qword_4F08028(a1[1], 6);
        LODWORD(a5) = dword_4F08018;
      }
      LODWORD(v5) = *((_DWORD *)a1 + 7);
      if ( (_DWORD)a5 )
      {
        a1[2] = 0;
        if ( (_DWORD)v5 == -2 )
        {
          a1[4] = 0;
          goto LABEL_35;
        }
      }
      else if ( (_DWORD)v5 == -2 )
      {
        goto LABEL_35;
      }
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        v5 = qword_4F08028(a1[4], 13);
        a1[4] = v5;
      }
      goto LABEL_35;
    case 42:
      if ( a4 )
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                42,
                a3,
                a4,
                a5,
                a3);
      v46 = *((_BYTE *)a1 + 8);
      if ( v46 == 2 )
      {
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( !qword_4F08028 )
          goto LABEL_35;
        a1[2] = qword_4F08028(a1[2], 8);
      }
      else
      {
        if ( v46 <= 2u )
        {
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( !qword_4F08028 )
            goto LABEL_35;
          a1[2] = qword_4F08028(a1[2], 37);
          goto LABEL_279;
        }
        if ( v46 != 3 )
LABEL_37:
          sub_721090();
      }
LABEL_279:
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        a1[3] = qword_4F08028(a1[3], 30);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[4] = qword_4F08028(a1[4], 13);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[7], 6);
            a1[7] = v5;
          }
        }
      }
      goto LABEL_35;
    case 43:
      if ( !a3 )
      {
        if ( !(_DWORD)a5 )
          goto LABEL_264;
        a1[4] = 0;
        goto LABEL_1148;
      }
      a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                a1[1],
                24,
                a3,
                a4,
                a5,
                a3);
      if ( qword_4F08028 )
      {
        v40 = qword_4F08028(a1[3], 24);
        v41 = dword_4F08018;
        a1[3] = v40;
        v42 = qword_4F08028;
        if ( !v41 )
        {
          if ( !qword_4F08028 )
            goto LABEL_263;
          goto LABEL_256;
        }
        a1[4] = 0;
        if ( v42 )
        {
LABEL_256:
          a1[5] = v42(a1[5], 23);
          if ( qword_4F08028 )
          {
            a1[6] = qword_4F08028(a1[6], 11);
            if ( qword_4F08028 )
            {
              v43 = qword_4F08028(a1[12], 52);
              v44 = dword_4F08018;
              a1[12] = v43;
              v45 = qword_4F08028;
              if ( v44 )
                *a1 = 0;
              if ( v45 )
              {
                a1[9] = v45(a1[9], 61);
                if ( qword_4F08028 )
                  a1[10] = qword_4F08028(a1[10], 65);
              }
              goto LABEL_263;
            }
            if ( dword_4F08018 )
            {
              *a1 = 0;
              v11 = qword_4F08020;
              goto LABEL_264;
            }
LABEL_263:
            v11 = qword_4F08020;
            goto LABEL_264;
          }
          v11 = qword_4F08020;
          if ( !dword_4F08018 )
          {
LABEL_264:
            if ( v11 )
            {
              a1[13] = v11(a1[13], 75);
              if ( qword_4F08020 )
                a1[14] = qword_4F08020(a1[14], 43);
            }
            if ( qword_4F08028 )
              a1[15] = qword_4F08028(a1[15], 2);
            LOBYTE(v5) = (_BYTE)qword_4F08020;
            if ( qword_4F08020 )
            {
              a1[17] = qword_4F08020(a1[17], 44);
              LOBYTE(v5) = (_BYTE)qword_4F08020;
              if ( qword_4F08020 )
              {
                a1[18] = qword_4F08020(a1[18], 46);
                LOBYTE(v5) = (_BYTE)qword_4F08020;
                if ( qword_4F08020 )
                {
                  v5 = qword_4F08020(a1[19], 47);
                  a1[19] = v5;
                }
              }
            }
            goto LABEL_35;
          }
LABEL_1148:
          *a1 = 0;
          goto LABEL_264;
        }
      }
      else
      {
        if ( !dword_4F08018 )
          goto LABEL_263;
        a1[4] = 0;
      }
      v11 = qword_4F08020;
      goto LABEL_1148;
    case 44:
      if ( a4 )
      {
        v39 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                44,
                a3,
                a4,
                a5,
                a3);
        a3 = qword_4F08028;
        *a1 = v39;
        if ( !a3 )
        {
          v5 = (unsigned __int64)qword_4F08020;
          if ( !qword_4F08020 )
            goto LABEL_35;
          goto LABEL_249;
        }
      }
      else if ( !a3 )
      {
        goto LABEL_35;
      }
      a1[1] = a3(a1[1], 26);
      v5 = (unsigned __int64)qword_4F08020;
      if ( !qword_4F08020 )
        goto LABEL_250;
LABEL_249:
      a1[2] = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(a1[2], 45);
LABEL_250:
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
LABEL_251:
        v5 = qword_4F08028(a1[5], 13);
        a1[5] = v5;
      }
      goto LABEL_35;
    case 45:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               a1[2],
               45,
               a3,
               a4,
               a5,
               a3);
        a1[2] = v5;
      }
      goto LABEL_35;
    case 46:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               46,
               a3,
               a4,
               a5,
               a3);
        *a1 = v5;
      }
      goto LABEL_35;
    case 47:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               47,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
               a1[1],
               12,
               a3,
               a4,
               a5,
               v10);
        a1[1] = v5;
      }
      goto LABEL_35;
    case 48:
      if ( a4 )
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                48,
                a3,
                a4,
                a5,
                a3);
      v38 = *((_BYTE *)a1 + 8);
      if ( v38 )
      {
        switch ( v38 )
        {
          case 1:
            if ( (a1[3] & 1) == 0 && qword_4F08028 )
              a1[4] = qword_4F08028(a1[4], 2);
            break;
          case 2:
            if ( qword_4F08028 )
              a1[4] = qword_4F08028(a1[4], 59);
            break;
          case 3:
            break;
          default:
            goto LABEL_37;
        }
      }
      else if ( qword_4F08028 )
      {
        a1[4] = qword_4F08028(a1[4], 6);
      }
      LOBYTE(v5) = dword_4F08018;
      if ( dword_4F08018 )
      {
        a1[6] = 0;
        a1[2] = 0;
      }
      goto LABEL_35;
    case 49:
      if ( !a3 )
        goto LABEL_1171;
      a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                a1[1],
                6,
                a3,
                a4,
                a5,
                a3);
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        v37 = qword_4F08028(a1[2], 11);
        a4 = qword_4F08020;
        a1[2] = v37;
        if ( !a4 )
        {
LABEL_227:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[4] = qword_4F08028(a1[4], 30);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              a1[5] = qword_4F08028(a1[5], 30);
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[6], 13);
                a1[6] = v5;
              }
            }
          }
          goto LABEL_35;
        }
      }
      else
      {
        a4 = qword_4F08020;
LABEL_1171:
        if ( !a4 )
          goto LABEL_35;
      }
      a1[3] = a4(a1[3], 13);
      goto LABEL_227;
    case 50:
      if ( a3 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                6,
                a3,
                a4,
                a5,
                a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[1] = qword_4F08028(a1[1], 30);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
            goto LABEL_222;
        }
      }
      goto LABEL_35;
    case 51:
      if ( a3 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                23,
                a3,
                a4,
                a5,
                a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[1] = qword_4F08028(a1[1], 30);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[2] = qword_4F08028(a1[2], 13);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              v5 = qword_4F08028(a1[3], 21);
              a1[3] = v5;
            }
          }
        }
      }
      goto LABEL_35;
    case 52:
      v35 = *((unsigned __int8 *)a1 + 16);
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               52,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      LOBYTE(v5) = (_BYTE)v35 == 6;
      LOBYTE(a4) = (_BYTE)v35 == 53;
      v36 = (unsigned int)v5 | (unsigned int)a4;
      if ( a3 )
      {
        v142 = v36;
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
               a1[1],
               52,
               a3,
               v36,
               a5,
               v10);
        a3 = qword_4F08028;
        a1[1] = v5;
        if ( v142 )
          goto LABEL_1051;
      }
      else if ( (_BYTE)v36 )
      {
        goto LABEL_35;
      }
      v5 = (unsigned int)(v35 - 54);
      if ( (unsigned __int8)(v35 - 54) > 0x20u || (v139 = 0x100018005LL, !_bittest64(&v139, v5)) )
      {
        if ( !a3 )
          goto LABEL_35;
        goto LABEL_213;
      }
LABEL_1051:
      if ( !a3 )
        goto LABEL_35;
LABEL_213:
      v5 = a3(a1[3], (unsigned __int8)v35);
      a1[3] = v5;
      goto LABEL_35;
    case 53:
      LOBYTE(v5) = *((_BYTE *)a1 + 16);
      if ( !a3 )
        goto LABEL_1132;
      a1[3] = ((__int64 (__fastcall *)(_QWORD, _QWORD, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                a1[3],
                (unsigned __int8)v5,
                a3,
                a4,
                a5,
                a3);
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 && (a1[4] = qword_4F08028(a1[4], 6), LOBYTE(v5) = (_BYTE)qword_4F08028, qword_4F08028) )
      {
        v80 = qword_4F08028(a1[5], 65);
        a4 = qword_4F08020;
        a1[5] = v80;
        if ( !a4 )
          goto LABEL_454;
      }
      else
      {
        a4 = qword_4F08020;
LABEL_1132:
        if ( !a4 )
          goto LABEL_35;
      }
      a1[6] = a4(a1[6], 75);
LABEL_454:
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        v5 = qword_4F08028(a1[1], 61);
        a1[1] = v5;
      }
      goto LABEL_35;
    case 54:
      LOBYTE(v5) = *((_BYTE *)a1 + 8);
      if ( (_BYTE)v5 == 6 )
      {
        if ( a3 )
        {
          v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                 a1[2],
                 6,
                 a3,
                 a4,
                 a5,
                 a3);
          a1[2] = v5;
        }
      }
      else if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, _QWORD, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
               a1[2],
               (unsigned __int8)v5,
               a3,
               a4,
               a5,
               a3);
        a1[2] = v5;
      }
      goto LABEL_35;
    case 55:
      if ( a4 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                55,
                a3,
                a4,
                a5,
                a3);
        LOBYTE(v5) = (_BYTE)qword_4F08020;
        if ( qword_4F08020 )
        {
          v5 = qword_4F08020(a1[1], 52);
          a1[1] = v5;
        }
        a3 = qword_4F08028;
      }
      if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
               a1[2],
               52,
               a3,
               a4,
               a5,
               v10);
        a1[2] = v5;
      }
      goto LABEL_35;
    case 56:
      if ( a3 )
      {
        v79 = ((__int64 (__fastcall *)(_QWORD, _QWORD, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                a1[2],
                *((unsigned __int8 *)a1 + 8),
                a3,
                a4,
                a5,
                a3);
        a4 = qword_4F08020;
        a1[2] = v79;
        if ( !a4 )
          goto LABEL_438;
      }
      else if ( !a4 )
      {
        goto LABEL_35;
      }
      a1[4] = a4(a1[4], 75);
LABEL_438:
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        v5 = qword_4F08028(a1[5], 61);
        a1[5] = v5;
      }
      goto LABEL_35;
    case 57:
      if ( a4 )
      {
        v81 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                57,
                a3,
                a4,
                a5,
                a3);
        a3 = qword_4F08028;
        *a1 = v81;
        if ( !a3 )
          goto LABEL_467;
      }
      else if ( !a3 )
      {
        goto LABEL_35;
      }
      a1[1] = a3(a1[1], 11);
LABEL_467:
      LOBYTE(v5) = (_BYTE)qword_4F08020;
      if ( qword_4F08020 )
      {
        a1[3] = qword_4F08020(a1[3], 6);
        LOBYTE(v5) = (_BYTE)qword_4F08020;
        if ( qword_4F08020 )
        {
          a1[4] = qword_4F08020(a1[4], 7);
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
          {
            a1[5] = qword_4F08020(a1[5], 28);
            LOBYTE(v5) = (_BYTE)qword_4F08020;
            if ( qword_4F08020 )
            {
              v5 = qword_4F08020(a1[6], 55);
              a1[6] = v5;
            }
          }
        }
      }
      goto LABEL_35;
    case 58:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               58,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        a1[3] = ((__int64 (__fastcall *)(_QWORD, _QWORD, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                  a1[3],
                  *((unsigned __int8 *)a1 + 16),
                  a3,
                  a4,
                  a5,
                  v10);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[5] = qword_4F08028(a1[5], 52);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[6], 26);
            v21 = *((_BYTE *)a1 + 8) == 21;
            a1[6] = v5;
            if ( v21 )
            {
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[7], 2);
                a1[7] = v5;
              }
            }
          }
        }
      }
      goto LABEL_35;
    case 59:
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                  a1[1],
                  24,
                  a3,
                  a4,
                  a5,
                  a3);
        if ( qword_4F08028 )
        {
          v55 = qword_4F08028(a1[3], 24);
          v56 = dword_4F08018;
          a1[3] = v55;
          v57 = qword_4F08028;
          if ( v56 )
            a1[4] = 0;
          if ( v57 )
          {
            a1[5] = v57(a1[5], 23);
            if ( qword_4F08028 )
            {
              a1[6] = qword_4F08028(a1[6], 11);
              if ( qword_4F08028 )
              {
                v58 = qword_4F08028(a1[12], 52);
                v59 = dword_4F08018;
                a1[12] = v58;
                v60 = qword_4F08028;
                if ( v59 )
                  *a1 = 0;
                if ( v60 )
                {
                  a1[9] = v60(a1[9], 61);
                  if ( qword_4F08028 )
                    a1[10] = qword_4F08028(a1[10], 65);
                }
                goto LABEL_359;
              }
              if ( dword_4F08018 )
              {
                *a1 = 0;
                v11 = qword_4F08020;
                goto LABEL_360;
              }
LABEL_359:
              v11 = qword_4F08020;
              goto LABEL_360;
            }
            v56 = dword_4F08018;
          }
          v11 = qword_4F08020;
          if ( v56 )
            goto LABEL_1128;
LABEL_360:
          if ( v11 )
          {
            a1[13] = v11(a1[13], 75);
            if ( qword_4F08020 )
              a1[14] = qword_4F08020(a1[14], 59);
          }
          if ( qword_4F08028 )
          {
            a1[22] = qword_4F08028(a1[22], 62);
            switch ( *((_BYTE *)a1 + 120) )
            {
              case 0:
              case 8:
                break;
              case 1:
              case 6:
              case 7:
                if ( !qword_4F08028 )
                  goto LABEL_723;
                a1[24] = qword_4F08028(a1[24], 6);
                break;
              case 2:
              case 4:
                if ( !qword_4F08028 )
                  goto LABEL_723;
                a1[24] = qword_4F08028(a1[24], 11);
                break;
              case 3:
              case 5:
                if ( !qword_4F08028 )
                  goto LABEL_723;
                a1[24] = qword_4F08028(a1[24], 7);
                break;
              case 9:
                if ( !qword_4F08028 )
                  goto LABEL_723;
                a1[24] = qword_4F08028(a1[24], 13);
                break;
              default:
                goto LABEL_37;
            }
            if ( qword_4F08028 )
            {
              a1[25] = qword_4F08028(a1[25], 59);
              if ( qword_4F08028 )
              {
                a1[26] = qword_4F08028(a1[26], 59);
                if ( qword_4F08028 )
                  a1[27] = qword_4F08028(a1[27], 59);
              }
            }
          }
          else if ( *((_BYTE *)a1 + 120) > 9u )
          {
            goto LABEL_37;
          }
LABEL_723:
          LOBYTE(v5) = dword_4F08018;
          if ( dword_4F08018 )
            a1[21] = 0;
          goto LABEL_35;
        }
        if ( !dword_4F08018 )
          goto LABEL_359;
        a1[4] = 0;
        v11 = qword_4F08020;
      }
      else
      {
        if ( !(_DWORD)a5 )
          goto LABEL_360;
        a1[4] = 0;
      }
LABEL_1128:
      *a1 = 0;
      goto LABEL_360;
    case 60:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               60,
               a3,
               a4,
               a5,
               a3);
        *a1 = v5;
      }
      goto LABEL_35;
    case 61:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               a1[6],
               60,
               a3,
               a4,
               a5,
               a3);
        a1[6] = v5;
      }
      goto LABEL_35;
    case 62:
      if ( a3 )
      {
        v68 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                62,
                a3,
                a4,
                a5,
                a3);
        a4 = qword_4F08020;
        *a1 = v68;
        if ( !a4 )
          goto LABEL_409;
      }
      else if ( !a4 )
      {
        goto LABEL_35;
      }
      a1[1] = a4(a1[1], 64);
LABEL_409:
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        a1[2] = qword_4F08028(a1[2], 63);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          v5 = qword_4F08028(a1[3], 23);
          a1[3] = v5;
        }
      }
      goto LABEL_35;
    case 63:
      if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
               *a1,
               13,
               a3,
               a4,
               a5,
               a3);
        *a1 = v5;
      }
      goto LABEL_35;
    case 64:
      if ( !a3 )
      {
        if ( !(_DWORD)a5 )
          goto LABEL_189;
        a1[4] = 0;
        goto LABEL_1189;
      }
      a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                a1[1],
                24,
                a3,
                a4,
                a5,
                a3);
      if ( qword_4F08028 )
      {
        v29 = qword_4F08028(a1[3], 24);
        v30 = dword_4F08018;
        a1[3] = v29;
        v31 = qword_4F08028;
        if ( !v30 )
        {
          if ( !qword_4F08028 )
            goto LABEL_188;
          goto LABEL_181;
        }
        a1[4] = 0;
        if ( v31 )
        {
LABEL_181:
          a1[5] = v31(a1[5], 23);
          if ( qword_4F08028 )
          {
            a1[6] = qword_4F08028(a1[6], 11);
            if ( qword_4F08028 )
            {
              v32 = qword_4F08028(a1[12], 52);
              v33 = dword_4F08018;
              a1[12] = v32;
              v34 = qword_4F08028;
              if ( v33 )
                *a1 = 0;
              if ( v34 )
              {
                a1[9] = v34(a1[9], 61);
                if ( qword_4F08028 )
                  a1[10] = qword_4F08028(a1[10], 65);
              }
              goto LABEL_188;
            }
            if ( dword_4F08018 )
            {
              *a1 = 0;
              v11 = qword_4F08020;
              goto LABEL_189;
            }
LABEL_188:
            v11 = qword_4F08020;
            goto LABEL_189;
          }
          v11 = qword_4F08020;
          if ( !dword_4F08018 )
          {
LABEL_189:
            if ( v11 )
            {
              a1[13] = v11(a1[13], 75);
              if ( qword_4F08020 )
                a1[14] = qword_4F08020(a1[14], 64);
            }
            LOBYTE(v5) = *((_BYTE *)a1 + 120);
            if ( (_BYTE)v5 == 2 )
            {
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[16] = qword_4F08028(a1[16], 2);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                {
                  v5 = qword_4F08028(a1[17], 2);
                  a1[17] = v5;
                }
              }
              goto LABEL_35;
            }
            if ( (unsigned __int8)v5 <= 2u )
            {
              if ( (_BYTE)v5 )
              {
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                {
                  a1[16] = qword_4F08028(a1[16], 6);
                  LOBYTE(v5) = (_BYTE)qword_4F08028;
                  if ( qword_4F08028 )
                  {
                    v5 = qword_4F08028(a1[17], 6);
                    a1[17] = v5;
                  }
                }
              }
              goto LABEL_35;
            }
            if ( (_BYTE)v5 == 3 )
            {
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[16] = qword_4F08028(a1[16], 59);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                {
                  v5 = qword_4F08028(a1[17], 59);
                  a1[17] = v5;
                }
              }
              goto LABEL_35;
            }
            goto LABEL_37;
          }
LABEL_1189:
          *a1 = 0;
          goto LABEL_189;
        }
      }
      else
      {
        if ( !dword_4F08018 )
          goto LABEL_188;
        a1[4] = 0;
      }
      v11 = qword_4F08020;
      goto LABEL_1189;
    case 65:
      if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
               *a1,
               65,
               a3,
               a4,
               a5,
               a3);
        *a1 = v5;
      }
      v27 = a1[1];
      if ( v27 )
      {
        if ( dword_4F07590 || (*((_BYTE *)a1 + 33) & 0x20) == 0 )
        {
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(v27, 66);
            a1[1] = v5;
          }
        }
        else
        {
          LOBYTE(v5) = dword_4F08018;
          if ( dword_4F08018 )
            a1[1] = 0;
        }
      }
      if ( !*((_BYTE *)a1 + 32) )
      {
        v28 = a1[2];
        if ( v28 )
        {
          LOBYTE(v5) = dword_4F07590;
          if ( dword_4F07590 || (*((_BYTE *)a1 + 33) & 0x20) == 0 )
          {
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              v5 = qword_4F08028(v28, 6);
              a1[2] = v5;
            }
          }
          else if ( dword_4F08018 )
          {
            a1[2] = 0;
          }
        }
      }
      goto LABEL_35;
    case 66:
      if ( (_DWORD)a5 )
        *a1 = 0;
      if ( (a1[4] & 1) != 0 )
      {
        if ( !a3 )
          goto LABEL_35;
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                  a1[1],
                  6,
                  a3,
                  a4,
                  a5,
                  a3);
        v5 = (unsigned __int64)qword_4F08028;
      }
      else
      {
        if ( !a3 )
          goto LABEL_35;
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                  a1[1],
                  28,
                  a3,
                  a4,
                  a5,
                  a3);
        v5 = (unsigned __int64)qword_4F08028;
      }
      v26 = a1[3];
      if ( v26 )
      {
        if ( !v5 )
          goto LABEL_35;
        a1[3] = ((__int64 (__fastcall *)(__int64, __int64))v5)(v26, 26);
        v5 = (unsigned __int64)qword_4F08028;
      }
      if ( v5 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(a1[2], 66);
        a1[2] = v5;
      }
      goto LABEL_35;
    case 67:
      if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
               a1[3],
               1,
               a3,
               a4,
               a5,
               a3);
        a4 = qword_4F08020;
        a1[3] = v5;
      }
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a4)(
               *a1,
               67,
               a3,
               a4,
               a5,
               v10);
        *a1 = v5;
      }
      goto LABEL_35;
    case 68:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               68,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                  a1[1],
                  13,
                  a3,
                  a4,
                  a5,
                  v10);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          v5 = qword_4F08028(a1[4], *((unsigned __int8 *)a1 + 24));
          a1[4] = v5;
        }
      }
      goto LABEL_35;
    case 69:
      if ( a3 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                2,
                a3,
                a4,
                a5,
                a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          v5 = qword_4F08028(a1[1], 2);
          a1[1] = v5;
        }
      }
      goto LABEL_35;
    case 70:
      if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
               *a1,
               2,
               a3,
               a4,
               a5,
               a3);
        *a1 = v5;
      }
      goto LABEL_35;
    case 71:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               71,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                  a1[1],
                  23,
                  a3,
                  a4,
                  a5,
                  v10);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          v5 = qword_4F08028(a1[3], *((unsigned __int8 *)a1 + 16));
          a1[3] = v5;
        }
      }
      goto LABEL_35;
    case 72:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               72,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, _QWORD, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
               a1[2],
               *((unsigned __int8 *)a1 + 8),
               a3,
               a4,
               a5,
               v10);
        a1[2] = v5;
      }
      goto LABEL_35;
    case 73:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               74,
               a3,
               a4,
               a5,
               a3);
        v10 = qword_4F08028;
        *a1 = v5;
      }
      if ( v10 )
      {
        a1[1] = v10(a1[1], 6);
        if ( (a1[3] & 1) == 0 || (v5 = (unsigned __int64)&dword_4F07590, dword_4F07590) )
        {
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
LABEL_222:
            v5 = qword_4F08028(a1[2], 11);
            a1[2] = v5;
          }
          goto LABEL_35;
        }
      }
      else
      {
        if ( (a1[3] & 1) == 0 )
          goto LABEL_35;
        LOBYTE(v5) = dword_4F07590;
        if ( dword_4F07590 )
          goto LABEL_35;
      }
      if ( dword_4F08018 )
        a1[2] = 0;
      goto LABEL_35;
    case 74:
      if ( a4 )
      {
        v24 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                74,
                a3,
                a4,
                a5,
                a3);
        v10 = qword_4F08028;
        *a1 = v24;
        a3 = v10;
      }
      LOBYTE(v5) = *((_BYTE *)a1 + 32);
      if ( (v5 & 1) != 0 )
      {
        if ( a3 )
        {
          v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                 a1[1],
                 30,
                 a3,
                 a4,
                 a5,
                 v10);
          v25 = dword_4F08018;
          v10 = qword_4F08028;
          a1[1] = v5;
          if ( !v25 )
            goto LABEL_122;
        }
        else if ( !dword_4F08018 )
        {
          goto LABEL_35;
        }
        a1[2] = 0;
      }
      else
      {
        if ( (v5 & 2) != 0 )
        {
          if ( !a3 )
            goto LABEL_35;
          a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                    a1[1],
                    8,
                    a3,
                    a4,
                    a5,
                    v10);
        }
        else
        {
          if ( !a3 )
            goto LABEL_35;
          a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                    a1[1],
                    7,
                    a3,
                    a4,
                    a5,
                    v10);
        }
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( !qword_4F08028 )
          goto LABEL_35;
        v5 = qword_4F08028(a1[2], 8);
        v10 = qword_4F08028;
        a1[2] = v5;
      }
LABEL_122:
      if ( v10 )
      {
        v5 = v10(a1[3], 8);
        a1[3] = v5;
      }
      goto LABEL_35;
    case 75:
      if ( a4 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
               *a1,
               75,
               a3,
               a4,
               a5,
               a3);
        a3 = qword_4F08028;
        *a1 = v5;
      }
      if ( a3 )
      {
        a1[2] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
                  a1[2],
                  24,
                  a3,
                  a4,
                  a5,
                  v10);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[3] = qword_4F08028(a1[3], 24);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[4] = qword_4F08028(a1[4], 76);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              v5 = qword_4F08028(a1[5], 77);
              a1[5] = v5;
            }
          }
        }
      }
      if ( dword_4F08018 )
      {
        a1[6] = 0;
        a1[9] = 0;
      }
      goto LABEL_35;
    case 76:
      if ( a3 )
      {
        v23 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                76,
                a3,
                a4,
                a5,
                a3);
        LODWORD(a5) = dword_4F08018;
        *a1 = v23;
      }
      if ( (_DWORD)a5 )
        a1[2] = 0;
      LOBYTE(v5) = *((_BYTE *)a1 + 10);
      switch ( (char)v5 )
      {
        case 0:
          goto LABEL_35;
        case 1:
        case 2:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[5], 24);
            a1[5] = v5;
          }
          goto LABEL_35;
        case 3:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[5], 2);
            a1[5] = v5;
          }
          goto LABEL_35;
        case 4:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
            goto LABEL_101;
          goto LABEL_35;
        case 5:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
            goto LABEL_251;
          goto LABEL_35;
        default:
          goto LABEL_37;
      }
    case 78:
      if ( a4 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                48,
                a3,
                a4,
                a5,
                a3);
        v5 = (unsigned __int64)&dword_4F07590;
        if ( dword_4F07590 )
        {
          LOBYTE(v5) = (_BYTE)qword_4F08020;
          if ( qword_4F08020 )
          {
            v5 = qword_4F08020(a1[1], 48);
            a1[1] = v5;
          }
          a3 = qword_4F08028;
LABEL_97:
          if ( a3 )
          {
            a1[2] = a3(a1[2], 59);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              a1[3] = qword_4F08028(a1[3], 13);
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[4] = qword_4F08028(a1[4], 6);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                {
LABEL_101:
                  v5 = qword_4F08028(a1[5], 6);
                  a1[5] = v5;
                }
              }
            }
          }
          goto LABEL_35;
        }
        LODWORD(a5) = dword_4F08018;
        a3 = qword_4F08028;
      }
      else
      {
        v5 = (unsigned __int64)&dword_4F07590;
        if ( dword_4F07590 )
          goto LABEL_97;
      }
      if ( (_DWORD)a5 )
        a1[1] = 0;
      goto LABEL_97;
    case 79:
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                  a1[1],
                  6,
                  a3,
                  a4,
                  a5,
                  a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          v5 = qword_4F08028(a1[3], 59);
          a1[3] = v5;
        }
      }
      goto LABEL_35;
    case 80:
      if ( a3 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                26,
                a3,
                a4,
                a5,
                a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[1] = qword_4F08028(a1[1], 11);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[2] = qword_4F08028(a1[2], 7);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              a1[3] = qword_4F08028(a1[3], 11);
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                v5 = qword_4F08028(a1[5], 26);
                a1[5] = v5;
              }
            }
          }
        }
      }
      goto LABEL_35;
    case 81:
      if ( !a3 )
        goto LABEL_72;
      *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
              *a1,
              6,
              a3,
              a4,
              a5,
              a3);
      if ( !qword_4F08028 )
        goto LABEL_71;
      a1[1] = qword_4F08028(a1[1], 7);
      if ( !qword_4F08028 )
        goto LABEL_71;
      a1[2] = qword_4F08028(a1[2], 7);
      if ( !qword_4F08028 )
        goto LABEL_71;
      a1[3] = qword_4F08028(a1[3], 7);
      LOBYTE(v5) = (_BYTE)qword_4F08028;
      if ( qword_4F08028 )
      {
        a1[4] = qword_4F08028(a1[4], 7);
LABEL_71:
        a4 = qword_4F08020;
LABEL_72:
        if ( !a4 )
        {
LABEL_74:
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            a1[6] = qword_4F08028(a1[6], 12);
            LOBYTE(v5) = (_BYTE)qword_4F08028;
            if ( qword_4F08028 )
            {
              a1[7] = qword_4F08028(a1[7], 13);
              LOBYTE(v5) = (_BYTE)qword_4F08028;
              if ( qword_4F08028 )
              {
                a1[8] = qword_4F08028(a1[8], 13);
                LOBYTE(v5) = (_BYTE)qword_4F08028;
                if ( qword_4F08028 )
                {
                  a1[9] = qword_4F08028(a1[9], 13);
                  LOBYTE(v5) = (_BYTE)qword_4F08028;
                  if ( qword_4F08028 )
                  {
                    a1[10] = qword_4F08028(a1[10], 13);
                    LOBYTE(v5) = (_BYTE)qword_4F08028;
                    if ( qword_4F08028 )
                    {
                      a1[11] = qword_4F08028(a1[11], 13);
                      LOBYTE(v5) = (_BYTE)qword_4F08028;
                      if ( qword_4F08028 )
                      {
                        a1[12] = qword_4F08028(a1[12], 11);
                        LOBYTE(v5) = (_BYTE)qword_4F08028;
                        if ( qword_4F08028 )
                        {
                          v5 = qword_4F08028(a1[13], 11);
                          a1[13] = v5;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          goto LABEL_35;
        }
      }
      else
      {
        a4 = qword_4F08020;
        if ( !qword_4F08020 )
          goto LABEL_35;
      }
      a1[5] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a4)(
                a1[5],
                7,
                a3,
                a4,
                a5,
                v10);
      goto LABEL_74;
    case 82:
      if ( a4 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                48,
                a3,
                a4,
                a5,
                a3);
        LOBYTE(v5) = (_BYTE)qword_4F08020;
        if ( qword_4F08020 )
        {
          v5 = qword_4F08020(a1[1], 48);
          a1[1] = v5;
        }
        a3 = qword_4F08028;
      }
      if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
               a1[2],
               59,
               a3,
               a4,
               a5,
               v10);
        a1[2] = v5;
      }
      goto LABEL_35;
    case 83:
      if ( a4 )
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                83,
                a3,
                a4,
                a5,
                a3);
      LOBYTE(v5) = *((_BYTE *)a1 + 8);
      if ( (v5 & 2) != 0 )
      {
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
LABEL_688:
          v5 = qword_4F08028(a1[2], 37);
          a1[2] = v5;
        }
      }
      else if ( (v5 & 1) == 0 )
      {
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          v5 = qword_4F08028(a1[2], 8);
          a1[2] = v5;
        }
      }
      goto LABEL_35;
    case 84:
      if ( a3 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                *a1,
                21,
                a3,
                a4,
                a5,
                a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          v5 = qword_4F08028(a1[1], 21);
          a1[1] = v5;
        }
      }
      goto LABEL_35;
    case 85:
      if ( a3 )
      {
        a1[1] = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a3)(
                  a1[1],
                  26,
                  a3,
                  a4,
                  a5,
                  a3);
        LOBYTE(v5) = (_BYTE)qword_4F08028;
        if ( qword_4F08028 )
        {
          a1[2] = qword_4F08028(a1[2], 26);
          LOBYTE(v5) = (_BYTE)qword_4F08028;
          if ( qword_4F08028 )
          {
            v5 = qword_4F08028(a1[3], 26);
            a1[3] = v5;
          }
        }
        LODWORD(a5) = dword_4F08018;
      }
      if ( (_DWORD)a5 )
        a1[4] = 0;
      goto LABEL_35;
    case 86:
      if ( a4 )
      {
        *a1 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, _QWORD)))a4)(
                *a1,
                86,
                a3,
                a4,
                a5,
                a3);
        LOBYTE(v5) = (_BYTE)qword_4F08020;
        if ( qword_4F08020 )
        {
          v5 = qword_4F08020(a1[3], 75);
          a1[3] = v5;
        }
        a3 = qword_4F08028;
      }
      if ( a3 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), __int64 (__fastcall *)(_QWORD, _QWORD), __int64, __int64 (__fastcall *)(_QWORD, __int64)))a3)(
               a1[4],
               85,
               a3,
               a4,
               a5,
               v10);
        a1[4] = v5;
      }
      goto LABEL_35;
    default:
      goto LABEL_37;
  }
}
