// Function: sub_2224080
// Address: 0x2224080
//
_QWORD *__fastcall sub_2224080(
        __int64 a1,
        _QWORD *a2,
        unsigned int a3,
        _QWORD *a4,
        int a5,
        __int64 a6,
        _DWORD *a7,
        __int64 a8)
{
  __int64 v8; // rbp
  __int64 v9; // r15
  __int64 v13; // r13
  __int64 *v14; // r14
  __int64 v15; // rsi
  unsigned __int64 v16; // rdx
  wchar_t *v17; // rax
  unsigned __int64 i; // r15
  char v19; // r13
  char v20; // bp
  char v21; // al
  char v22; // r14
  unsigned int v23; // eax
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  char v27; // bp
  char v28; // al
  char v29; // r15
  unsigned int *v30; // rax
  unsigned int v31; // eax
  char v32; // bp
  char v33; // al
  char v34; // r14
  unsigned int v35; // ebp
  wchar_t *v36; // rax
  unsigned __int64 v37; // rbx
  unsigned __int64 v38; // rbp
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rax
  unsigned int *v41; // rax
  unsigned int v42; // eax
  unsigned __int64 v43; // r15
  unsigned __int64 v44; // rax
  char v45; // bp
  char v46; // al
  char v47; // r13
  __int64 v48; // rdx
  unsigned __int64 v49; // rax
  __int64 v50; // rbx
  unsigned __int64 v51; // rax
  int *v52; // rax
  int v53; // eax
  int *v54; // rax
  int v55; // eax
  char v56; // al
  __int64 v57; // rax
  char v58; // bp
  __int64 v59; // rdx
  char v61; // al
  int *v62; // rax
  int v63; // eax
  bool v64; // zf
  int *v65; // rax
  int v66; // eax
  unsigned int *v67; // rax
  char v68; // r13
  char v69; // al
  char v70; // r14
  int *v71; // rax
  int v72; // eax
  int *v73; // rax
  int v74; // eax
  bool v75; // zf
  __int64 v76; // rbp
  char v77; // r14
  unsigned __int64 v78; // rdx
  unsigned __int64 v79; // rax
  int *v80; // rax
  int v81; // eax
  int *v82; // rax
  int v83; // eax
  int *v84; // rax
  int v85; // eax
  int *v86; // rax
  int v87; // eax
  _QWORD *v88; // rax
  char v89; // r13
  char v90; // al
  char v91; // r14
  unsigned __int64 v92; // rax
  unsigned __int64 v93; // rdx
  int *v94; // rax
  int v95; // eax
  int *v96; // rax
  int v97; // eax
  bool v98; // zf
  int *v99; // rax
  int v100; // eax
  unsigned int *v101; // rax
  unsigned int v102; // eax
  unsigned __int64 v103; // rax
  unsigned int *v104; // rax
  unsigned int v105; // eax
  unsigned __int64 v106; // rax
  int *v107; // rax
  int v108; // eax
  unsigned int *v109; // rax
  unsigned int v110; // eax
  int *v111; // rax
  int v112; // eax
  int *v113; // rax
  int v114; // eax
  bool v115; // zf
  int *v116; // rax
  int v117; // eax
  bool v118; // zf
  __int64 v120; // [rsp+8h] [rbp-110h]
  unsigned int v122; // [rsp+14h] [rbp-104h]
  wchar_t *s; // [rsp+18h] [rbp-100h]
  __int64 v124; // [rsp+20h] [rbp-F8h]
  unsigned __int64 v125; // [rsp+38h] [rbp-E0h]
  __int64 v126; // [rsp+48h] [rbp-D0h]
  unsigned __int64 v127; // [rsp+58h] [rbp-C0h]
  char v130; // [rsp+88h] [rbp-90h]
  unsigned __int8 v131; // [rsp+8Dh] [rbp-8Bh]
  char v132; // [rsp+8Eh] [rbp-8Ah]
  bool v133; // [rsp+8Fh] [rbp-89h]
  int v134; // [rsp+9Ch] [rbp-7Ch]
  _QWORD *v135; // [rsp+A0h] [rbp-78h] BYREF
  __int64 v136; // [rsp+A8h] [rbp-70h]
  _QWORD v137[2]; // [rsp+B0h] [rbp-68h] BYREF
  _QWORD *v138; // [rsp+C0h] [rbp-58h] BYREF
  unsigned __int64 v139; // [rsp+C8h] [rbp-50h]
  _QWORD v140[9]; // [rsp+D0h] [rbp-48h] BYREF

  v9 = a6 + 208;
  v126 = sub_2243120(a6 + 208);
  v13 = sub_22091A0(&qword_4FD6900);
  v14 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a6 + 208) + 24LL) + 8 * v13);
  v120 = *v14;
  if ( !*v14 )
  {
    v8 = sub_22077B0(0xA0u);
    *(_DWORD *)(v8 + 8) = 0;
    *(_QWORD *)(v8 + 16) = 0;
    *(_QWORD *)(v8 + 24) = 0;
    *(_QWORD *)v8 = off_4A048A0;
    *(_BYTE *)(v8 + 32) = 0;
    *(_QWORD *)(v8 + 36) = 0;
    *(_QWORD *)(v8 + 48) = 0;
    *(_QWORD *)(v8 + 56) = 0;
    *(_QWORD *)(v8 + 64) = 0;
    *(_QWORD *)(v8 + 72) = 0;
    *(_QWORD *)(v8 + 80) = 0;
    *(_QWORD *)(v8 + 88) = 0;
    *(_QWORD *)(v8 + 96) = 0;
    *(_DWORD *)(v8 + 104) = 0;
    *(_BYTE *)(v8 + 152) = 0;
    sub_2243C60(v8, v9);
    sub_2209690(*(_QWORD *)(a6 + 208), (volatile signed __int32 *)v8, v13);
    v120 = *v14;
  }
  if ( *(_QWORD *)(v120 + 72) )
    v133 = *(_QWORD *)(v120 + 88) != 0;
  else
    v133 = 0;
  LOBYTE(v137[0]) = 0;
  v135 = v137;
  v136 = 0;
  if ( *(_BYTE *)(v120 + 32) )
    sub_2240E30(&v135, 32);
  v15 = 32;
  v139 = 0;
  v138 = v140;
  LOBYTE(v140[0]) = 0;
  sub_2240E30(&v138, 32);
  v131 = 0;
  v124 = 0;
  v122 = 0;
  v134 = *(_DWORD *)(v120 + 104);
  v130 = 0;
  v125 = 0;
  v132 = 0;
  while ( 2 )
  {
    switch ( *((_BYTE *)&v134 + v124) )
    {
      case 0:
        LODWORD(v13) = 1;
        goto LABEL_10;
      case 1:
        v45 = a3 == -1 && a2 != 0;
        if ( v45 )
        {
          v99 = (int *)a2[2];
          if ( (unsigned __int64)v99 >= a2[3] )
            v100 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
          else
            v100 = *v99;
          if ( v100 == -1 )
            a2 = 0;
          else
            v45 = 0;
        }
        else
        {
          v45 = a3 == -1;
        }
        v46 = a5 == -1;
        v47 = v46 & (a4 != 0);
        if ( v47 )
        {
          v15 = (__int64)a4;
          v96 = (int *)a4[2];
          if ( (unsigned __int64)v96 >= a4[3] )
            v97 = (*(__int64 (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4);
          else
            v97 = *v96;
          v98 = v97 == -1;
          v16 = 0;
          if ( v97 != -1 )
            v16 = (unsigned __int64)a4;
          v46 = 0;
          if ( v98 )
            v46 = v47;
          a4 = (_QWORD *)v16;
        }
        LODWORD(v13) = 0;
        if ( v46 != v45 )
        {
          if ( a2 && a3 == -1 )
          {
            v109 = (unsigned int *)a2[2];
            if ( (unsigned __int64)v109 >= a2[3] )
            {
              v110 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
              v48 = v110;
            }
            else
            {
              v48 = *v109;
              v110 = *v109;
            }
            if ( v110 == -1 )
              a2 = 0;
          }
          else
          {
            v48 = a3;
          }
          v15 = 0x2000;
          LODWORD(v13) = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v126 + 16LL))(
                           v126,
                           0x2000,
                           v48);
          if ( (_BYTE)v13 )
          {
            v49 = a2[2];
            if ( v49 >= a2[3] )
              (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
            else
              a2[2] = v49 + 4;
            a3 = -1;
          }
        }
LABEL_10:
        if ( v124 == 3 )
          goto LABEL_11;
        while ( 1 )
        {
          v27 = a3 == -1 && a2 != 0;
          if ( v27 )
          {
            v65 = (int *)a2[2];
            if ( (unsigned __int64)v65 >= a2[3] )
              v66 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
            else
              v66 = *v65;
            if ( v66 == -1 )
              a2 = 0;
            else
              v27 = 0;
          }
          else
          {
            v27 = a3 == -1;
          }
          v28 = a5 == -1;
          v29 = v28 & (a4 != 0);
          if ( v29 )
          {
            v62 = (int *)a4[2];
            if ( (unsigned __int64)v62 >= a4[3] )
              v63 = (*(__int64 (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4);
            else
              v63 = *v62;
            v64 = v63 == -1;
            v15 = 0;
            if ( v63 != -1 )
              v15 = (__int64)a4;
            v28 = 0;
            if ( v64 )
              v28 = v29;
            a4 = (_QWORD *)v15;
          }
          if ( v28 == v27 )
            break;
          if ( a2 && a3 == -1 )
          {
            v30 = (unsigned int *)a2[2];
            if ( (unsigned __int64)v30 >= a2[3] )
            {
              v31 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
              v25 = v31;
            }
            else
            {
              v25 = *v30;
              v31 = *v30;
            }
            if ( v31 == -1 )
              a2 = 0;
          }
          else
          {
            v25 = a3;
          }
          v15 = 0x2000;
          if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v126 + 16LL))(
                  v126,
                  0x2000,
                  v25) )
            break;
          v26 = a2[2];
          if ( v26 >= a2[3] )
            (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
          else
            a2[2] = v26 + 4;
          a3 = -1;
        }
        goto LABEL_132;
      case 2:
        if ( (*(_BYTE *)(a6 + 25) & 2) != 0 )
          goto LABEL_74;
        v15 = v124;
        LOBYTE(v8) = (_DWORD)v124 == 0 || v125 > 1;
        if ( (_BYTE)v8 )
          goto LABEL_74;
        if ( (_DWORD)v124 == 1 )
        {
          if ( v133 || (_BYTE)v134 == 3 || BYTE2(v134) == 1 )
          {
LABEL_74:
            v43 = 0;
            v127 = *(_QWORD *)(v120 + 56);
            while ( 1 )
            {
              LOBYTE(v13) = a3 == -1;
              LOBYTE(v8) = v13 & (a2 != 0);
              if ( (_BYTE)v8 )
              {
                v84 = (int *)a2[2];
                if ( (unsigned __int64)v84 >= a2[3] )
                  v85 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
                else
                  v85 = *v84;
                if ( v85 == -1 )
                  a2 = 0;
                else
                  LODWORD(v8) = 0;
              }
              else
              {
                LODWORD(v8) = v13;
              }
              LOBYTE(v16) = a5 == -1;
              LOBYTE(v14) = v16 & (a4 != 0);
              if ( (_BYTE)v14 )
              {
                v86 = (int *)a4[2];
                if ( (unsigned __int64)v86 >= a4[3] )
                  v87 = (*(__int64 (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4);
                else
                  v87 = *v86;
                v64 = v87 == -1;
                LODWORD(v16) = 0;
                v88 = 0;
                if ( v64 )
                  LODWORD(v16) = (_DWORD)v14;
                else
                  v88 = a4;
                a4 = v88;
                LODWORD(v8) = v16 ^ v8;
                LOBYTE(v8) = (v43 < v127) & v8;
                if ( !(_BYTE)v8 )
                {
LABEL_214:
                  if ( v43 == v127 )
                    break;
LABEL_167:
                  if ( !v43 )
                  {
                    LOBYTE(v13) = (*(_BYTE *)(a6 + 25) & 2) == 0;
                    LOBYTE(v8) = (*(_BYTE *)(a6 + 25) & 2) != 0;
                    goto LABEL_133;
                  }
LABEL_119:
                  v19 = a3 == -1;
                  goto LABEL_120;
                }
              }
              else
              {
                LODWORD(v8) = v16 ^ v8;
                LOBYTE(v8) = (v43 < v127) & v8;
                if ( !(_BYTE)v8 )
                  goto LABEL_214;
              }
              if ( a2 && a3 == -1 )
              {
                v71 = (int *)a2[2];
                if ( (unsigned __int64)v71 >= a2[3] )
                  v72 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
                else
                  v72 = *v71;
                v15 = v120;
                if ( v72 == -1 )
                  a2 = 0;
                v16 = *(_QWORD *)(v120 + 48);
                if ( *(_DWORD *)(v16 + 4 * v43) != v72 )
                  goto LABEL_167;
              }
              else
              {
                v15 = v120;
                v16 = *(_QWORD *)(v120 + 48);
                if ( *(_DWORD *)(v16 + 4 * v43) != a3 )
                  goto LABEL_167;
              }
              v44 = a2[2];
              if ( v44 >= a2[3] )
                (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
              else
                a2[2] = v44 + 4;
              ++v43;
              a3 = -1;
            }
          }
LABEL_215:
          LODWORD(v13) = 1;
          goto LABEL_133;
        }
        LODWORD(v13) = 1;
        if ( (_DWORD)v124 != 2 )
          goto LABEL_133;
        if ( HIBYTE(v134) == 4 || HIBYTE(v134) == 3 && v133 )
          goto LABEL_74;
        goto LABEL_135;
      case 3:
        if ( *(_QWORD *)(v120 + 72) )
        {
          v68 = a3 == -1 && a2 != 0;
          if ( v68 )
          {
            v107 = (int *)a2[2];
            if ( (unsigned __int64)v107 >= a2[3] )
              v108 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
            else
              v108 = *v107;
            if ( v108 == -1 )
              a2 = 0;
            else
              v68 = 0;
          }
          else
          {
            v68 = a3 == -1;
          }
          v69 = a5 == -1;
          v70 = v69 & (a4 != 0);
          if ( v70 )
          {
            v15 = (__int64)a4;
            v113 = (int *)a4[2];
            if ( (unsigned __int64)v113 >= a4[3] )
              v114 = (*(__int64 (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4);
            else
              v114 = *v113;
            v115 = v114 == -1;
            v16 = 0;
            if ( v114 != -1 )
              v16 = (unsigned __int64)a4;
            v69 = 0;
            if ( v115 )
              v69 = v70;
            a4 = (_QWORD *)v16;
          }
          if ( v69 != v68 )
          {
            if ( a2 && a3 == -1 )
            {
              v101 = (unsigned int *)a2[2];
              if ( (unsigned __int64)v101 >= a2[3] )
                v102 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
              else
                v102 = *v101;
              if ( v102 == -1 )
                a2 = 0;
            }
            else
            {
              v102 = a3;
            }
            v15 = v120;
            v16 = *(_QWORD *)(v120 + 64);
            if ( *(_DWORD *)v16 == v102 )
            {
              v125 = *(_QWORD *)(v120 + 72);
              v103 = a2[2];
              if ( v103 >= a2[3] )
                (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
              else
                a2[2] = v103 + 4;
              LODWORD(v8) = 0;
              a3 = -1;
              goto LABEL_215;
            }
          }
          if ( !*(_QWORD *)(v120 + 88) )
          {
            if ( !*(_QWORD *)(v120 + 72) )
              goto LABEL_66;
LABEL_161:
            v132 = 1;
            LODWORD(v8) = 0;
            LODWORD(v13) = 1;
            goto LABEL_133;
          }
        }
        else if ( !*(_QWORD *)(v120 + 88) )
        {
          goto LABEL_66;
        }
        v89 = a3 == -1 && a2 != 0;
        if ( v89 )
        {
          v111 = (int *)a2[2];
          if ( (unsigned __int64)v111 >= a2[3] )
            v112 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
          else
            v112 = *v111;
          if ( v112 == -1 )
            a2 = 0;
          else
            v89 = 0;
        }
        else
        {
          v89 = a3 == -1;
        }
        v90 = a5 == -1;
        v91 = v90 & (a4 != 0);
        if ( v91 )
        {
          v116 = (int *)a4[2];
          if ( (unsigned __int64)v116 >= a4[3] )
            v117 = (*(__int64 (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4);
          else
            v117 = *v116;
          v118 = v117 == -1;
          v16 = 0;
          if ( v117 != -1 )
            v16 = (unsigned __int64)a4;
          v90 = 0;
          if ( v118 )
            v90 = v91;
          a4 = (_QWORD *)v16;
        }
        if ( v90 == v89 )
          goto LABEL_343;
        if ( a2 && a3 == -1 )
        {
          v104 = (unsigned int *)a2[2];
          if ( (unsigned __int64)v104 >= a2[3] )
            v105 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
          else
            v105 = *v104;
          if ( v105 == -1 )
            a2 = 0;
        }
        else
        {
          v105 = a3;
        }
        v16 = *(_QWORD *)(v120 + 80);
        if ( *(_DWORD *)v16 != v105 )
        {
LABEL_343:
          if ( !*(_QWORD *)(v120 + 72) || *(_QWORD *)(v120 + 88) )
          {
LABEL_66:
            LODWORD(v8) = v133;
            LODWORD(v13) = !v133;
            goto LABEL_133;
          }
          goto LABEL_161;
        }
        v125 = *(_QWORD *)(v120 + 88);
        v106 = a2[2];
        if ( v106 >= a2[3] )
          (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
        else
          a2[2] = v106 + 4;
        v132 = 1;
        LODWORD(v8) = 0;
        a3 = -1;
        LODWORD(v13) = 1;
        goto LABEL_133;
      case 4:
        while ( 2 )
        {
          v32 = a3 == -1 && a2 != 0;
          if ( v32 )
          {
            v52 = (int *)a2[2];
            if ( (unsigned __int64)v52 >= a2[3] )
              v53 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
            else
              v53 = *v52;
            if ( v53 == -1 )
              a2 = 0;
            else
              v32 = 0;
          }
          else
          {
            v32 = a3 == -1;
          }
          v33 = a5 == -1;
          v34 = v33 & (a4 != 0);
          if ( v34 )
          {
            v54 = (int *)a4[2];
            if ( (unsigned __int64)v54 >= a4[3] )
              v55 = (*(__int64 (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4);
            else
              v55 = *v54;
            v15 = (__int64)a4;
            v64 = v55 == -1;
            v56 = 0;
            if ( v64 )
            {
              v56 = v34;
              v15 = 0;
            }
            a4 = (_QWORD *)v15;
            if ( v56 == v32 )
              goto LABEL_117;
          }
          else if ( v33 == v32 )
          {
            goto LABEL_117;
          }
          if ( a2 && a3 == -1 )
          {
            v41 = (unsigned int *)a2[2];
            if ( (unsigned __int64)v41 >= a2[3] )
            {
              v42 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
              v35 = v42;
            }
            else
            {
              v35 = *v41;
              v42 = *v41;
            }
            if ( v42 == -1 )
              a2 = 0;
          }
          else
          {
            v35 = a3;
          }
          v15 = v35;
          v36 = wmemchr((const wchar_t *)(v120 + 112), v35, 0xAu);
          if ( v36 )
          {
            v37 = v139;
            v38 = v139 + 1;
            LODWORD(v13) = (unsigned __int8)off_4CDFAD0[((__int64)v36 - v120 - 108) >> 2];
            v16 = (unsigned __int64)v138;
            v39 = 15;
            if ( v138 != v140 )
              v39 = v140[0];
            if ( v38 > v39 )
            {
              v15 = v139;
              sub_2240BB0(&v138, v139, 0, 0, 1);
              v16 = (unsigned __int64)v138;
            }
            *(_BYTE *)(v16 + v37) = v13;
            ++v122;
            v139 = v38;
            *((_BYTE *)v138 + v37 + 1) = 0;
LABEL_56:
            v40 = a2[2];
            if ( v40 < a2[3] )
            {
LABEL_57:
              a2[2] = v40 + 4;
LABEL_58:
              a3 = -1;
              continue;
            }
LABEL_106:
            (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
            goto LABEL_58;
          }
          break;
        }
        LOBYTE(v13) = v131 | (*(_DWORD *)(v120 + 36) != v35);
        if ( !(_BYTE)v13 )
        {
          if ( *(int *)(v120 + 96) > 0 )
          {
            v61 = v122;
            v131 = 1;
            v122 = 0;
            v130 = v61;
            goto LABEL_56;
          }
          v131 = 0;
LABEL_117:
          LODWORD(v13) = 1;
          goto LABEL_118;
        }
        v15 = v120;
        if ( *(_BYTE *)(v120 + 32) )
        {
          if ( *(_DWORD *)(v120 + 40) == v35 )
          {
            if ( v131 )
            {
              LODWORD(v13) = v131;
            }
            else
            {
              if ( v122 )
              {
                v50 = v136;
                v16 = (unsigned __int64)v135;
                LODWORD(v13) = v122;
                v51 = 15;
                if ( v135 != v137 )
                  v51 = v137[0];
                if ( v136 + 1 > v51 )
                {
                  v15 = v136;
                  sub_2240BB0(&v135, v136, 0, 0, 1);
                  v16 = (unsigned __int64)v135;
                }
                *(_BYTE *)(v16 + v50) = v122;
                v136 = v50 + 1;
                v122 = 0;
                *((_BYTE *)v135 + v50 + 1) = 0;
                v40 = a2[2];
                if ( v40 < a2[3] )
                  goto LABEL_57;
                goto LABEL_106;
              }
              LODWORD(v13) = 0;
            }
          }
          else
          {
            LODWORD(v13) = *(unsigned __int8 *)(v120 + 32);
          }
        }
LABEL_118:
        if ( !v139 )
          goto LABEL_119;
LABEL_132:
        LODWORD(v8) = v13 ^ 1;
LABEL_133:
        if ( (int)v124 + 1 <= 3 && !(_BYTE)v8 )
        {
LABEL_135:
          ++v124;
          continue;
        }
LABEL_11:
        if ( ((unsigned __int8)v13 & (v125 > 1)) != 0 )
        {
          if ( v132 )
            v17 = *(wchar_t **)(v120 + 80);
          else
            v17 = *(wchar_t **)(v120 + 64);
          s = v17;
          for ( i = 1; ; ++i )
          {
            v19 = a3 == -1;
            v20 = v19 & (a2 != 0);
            if ( v20 )
            {
              v82 = (int *)a2[2];
              if ( (unsigned __int64)v82 >= a2[3] )
                v83 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
              else
                v83 = *v82;
              if ( v83 == -1 )
                a2 = 0;
              else
                v20 = 0;
            }
            else
            {
              v20 = a3 == -1;
            }
            v21 = a5 == -1;
            v22 = v21 & (a4 != 0);
            if ( v22 )
            {
              v73 = (int *)a4[2];
              if ( (unsigned __int64)v73 >= a4[3] )
                v74 = (*(__int64 (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4);
              else
                v74 = *v73;
              v75 = v74 == -1;
              v15 = 0;
              if ( v74 != -1 )
                v15 = (__int64)a4;
              v21 = 0;
              if ( v75 )
                v21 = v22;
              a4 = (_QWORD *)v15;
              if ( i >= v125 )
              {
LABEL_176:
                if ( i == v125 )
                {
                  if ( v139 <= 1 )
                    goto LABEL_178;
                  goto LABEL_226;
                }
                goto LABEL_120;
              }
            }
            else if ( i >= v125 )
            {
              goto LABEL_176;
            }
            if ( v21 == v20 )
              goto LABEL_176;
            if ( a2 && a3 == -1 )
            {
              v67 = (unsigned int *)a2[2];
              if ( (unsigned __int64)v67 >= a2[3] )
                v23 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
              else
                v23 = *v67;
              v15 = 0;
              if ( v23 == -1 )
                a2 = 0;
            }
            else
            {
              v23 = a3;
            }
            if ( s[i] != v23 )
              goto LABEL_120;
            v24 = a2[2];
            if ( v24 >= a2[3] )
              (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
            else
              a2[2] = v24 + 4;
            a3 = -1;
          }
        }
        if ( !(_BYTE)v13 )
          goto LABEL_119;
        if ( v139 <= 1 )
          goto LABEL_178;
LABEL_226:
        v92 = sub_2241A40(&v138, 48, 0);
        if ( v92 )
        {
          v93 = v139;
          if ( v92 != -1 )
            goto LABEL_228;
          v92 = v139 - 1;
          if ( !v139 )
          {
            v139 = 0;
            *(_BYTE *)v138 = 0;
            goto LABEL_178;
          }
          if ( v139 != 1 )
          {
LABEL_228:
            if ( v139 > v92 )
              v93 = v92;
            sub_2240CE0(&v138, 0, v93);
          }
        }
LABEL_178:
        if ( v132 && *(_BYTE *)v138 != 48 )
          sub_2240FD0(&v138, 0, 0, 1, 45);
        v76 = v136;
        if ( v136 )
        {
          v77 = v130;
          v78 = (unsigned __int64)v135;
          if ( !v131 )
            v77 = v122;
          v79 = 15;
          if ( v135 != v137 )
            v79 = v137[0];
          if ( v136 + 1 > v79 )
          {
            sub_2240BB0(&v135, v136, 0, 0, 1);
            v78 = (unsigned __int64)v135;
          }
          *(_BYTE *)(v78 + v76) = v77;
          v136 = v76 + 1;
          *((_BYTE *)v135 + v76 + 1) = 0;
          if ( !(unsigned __int8)sub_2255C00(*(_QWORD *)(v120 + 16), *(_QWORD *)(v120 + 24), &v135) )
            *a7 |= 4u;
        }
        v19 = a3 == -1;
        if ( v131 && (v15 = v122, *(_DWORD *)(v120 + 96) != v122) )
        {
LABEL_120:
          LODWORD(v57) = (_DWORD)a7;
          *a7 |= 4u;
          v58 = v19 & (a2 != 0);
          if ( v58 )
          {
LABEL_193:
            v80 = (int *)a2[2];
            if ( (unsigned __int64)v80 >= a2[3] )
              v81 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
            else
              v81 = *v80;
            v64 = v81 == -1;
            v19 = 0;
            LODWORD(v57) = 0;
            if ( v64 )
            {
              v19 = v58;
              a2 = 0;
            }
          }
        }
        else
        {
          v15 = (__int64)&v138;
          v57 = sub_22415E0(a8, &v138);
          v58 = v19 & (a2 != 0);
          if ( v58 )
            goto LABEL_193;
        }
        LOBYTE(v57) = a5 == -1;
        v59 = (unsigned int)v57;
        if ( a4 && a5 == -1 )
        {
          v94 = (int *)a4[2];
          if ( (unsigned __int64)v94 >= a4[3] )
            v95 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*a4 + 72LL))(a4, v15, v59);
          else
            v95 = *v94;
          LOBYTE(v59) = v95 == -1;
        }
        if ( v19 == (_BYTE)v59 )
          *a7 |= 2u;
        if ( v138 != v140 )
          j___libc_free_0((unsigned __int64)v138);
        if ( v135 != v137 )
          j___libc_free_0((unsigned __int64)v135);
        return a2;
      default:
        LODWORD(v8) = 0;
        LODWORD(v13) = 1;
        goto LABEL_133;
    }
  }
}
