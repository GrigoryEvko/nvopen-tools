// Function: sub_112E9E0
// Address: 0x112e9e0
//
unsigned __int8 *__fastcall sub_112E9E0(__int64 a1, __int64 a2, char *a3, __int64 a4)
{
  int v7; // r13d
  __int64 *v8; // r11
  bool v9; // zf
  __int64 *v10; // r9
  unsigned __int8 *result; // rax
  __int64 v12; // r15
  unsigned int v13; // edx
  int v14; // eax
  __int64 v15; // rax
  char v16; // r15
  unsigned int v17; // r12d
  __int64 v18; // rax
  __int64 *v19; // r15
  __int64 v21; // rax
  int v22; // ecx
  unsigned __int64 v23; // rcx
  unsigned __int8 *v25; // rax
  __int16 v26; // r12
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 *v29; // rdx
  __int64 v30; // rcx
  _BYTE *v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r12
  unsigned int v35; // r15d
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 *v38; // r12
  __int64 *v39; // r15
  unsigned int v40; // edx
  __int64 *v41; // rax
  unsigned int **v42; // r12
  _BYTE *v43; // rax
  __int64 v44; // r12
  __int64 v45; // r14
  __int64 v46; // r12
  __int64 v47; // rax
  __int64 *v48; // r9
  __int64 v49; // rax
  __int64 v50; // rdx
  _BYTE *v51; // rcx
  int v52; // eax
  __int64 *v53; // rdx
  __int64 *v54; // r13
  unsigned int v55; // eax
  unsigned int v56; // ebx
  __int64 v57; // r12
  __int64 v58; // r13
  _BYTE *v59; // rdx
  __int64 *v60; // r9
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rdi
  _BYTE *v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 *v68; // r15
  __int64 *v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  unsigned __int64 *v72; // r15
  __int64 v73; // rsi
  _BYTE *v74; // r12
  unsigned int **v75; // rdi
  __int64 v76; // rax
  __int64 *v77; // rdx
  int v78; // ebx
  int v79; // eax
  _BYTE *v80; // rax
  unsigned int v81; // r12d
  __int64 v82; // r15
  __int64 v83; // rax
  char v84; // al
  const void **v85; // rsi
  __int64 v86; // rax
  __int64 v87; // r13
  int v88; // ebx
  __int64 *v89; // rdx
  __int64 *v90; // rsi
  __int64 v91; // r12
  __int64 v92; // r14
  __int64 *v93; // rdx
  __int64 *v94; // rax
  unsigned int v95; // [rsp+0h] [rbp-E0h]
  _BYTE *v96; // [rsp+0h] [rbp-E0h]
  unsigned int v97; // [rsp+8h] [rbp-D8h]
  int v98; // [rsp+8h] [rbp-D8h]
  __int64 *v99; // [rsp+8h] [rbp-D8h]
  __int64 *v100; // [rsp+8h] [rbp-D8h]
  __int64 *v101; // [rsp+8h] [rbp-D8h]
  __int16 v103; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v104; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v105; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v106; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v107; // [rsp+18h] [rbp-C8h]
  __int64 *v108; // [rsp+18h] [rbp-C8h]
  char v109; // [rsp+27h] [rbp-B9h] BYREF
  __int64 *v110; // [rsp+28h] [rbp-B8h] BYREF
  __int64 *v111; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE *v112; // [rsp+38h] [rbp-A8h] BYREF
  __int64 *v113; // [rsp+40h] [rbp-A0h] BYREF
  const void **v114; // [rsp+48h] [rbp-98h] BYREF
  __int64 *v115; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v116; // [rsp+58h] [rbp-88h]
  unsigned __int64 v117; // [rsp+60h] [rbp-80h] BYREF
  const void **v118; // [rsp+68h] [rbp-78h] BYREF
  const void **v119; // [rsp+70h] [rbp-70h] BYREF
  unsigned __int64 v120; // [rsp+80h] [rbp-60h] BYREF
  __int64 **v121; // [rsp+88h] [rbp-58h] BYREF
  __int64 **v122; // [rsp+90h] [rbp-50h]
  __int64 **v123; // [rsp+98h] [rbp-48h] BYREF
  __int16 v124; // [rsp+A0h] [rbp-40h]

  v7 = *(_WORD *)(a2 + 2) & 0x3F;
  v103 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( !sub_9893F0(v7, a4, &v109) )
  {
    v8 = (__int64 *)*((_QWORD *)a3 - 8);
    goto LABEL_9;
  }
  v8 = (__int64 *)*((_QWORD *)a3 - 8);
  v121 = 0;
  v120 = (unsigned __int64)&v115;
  if ( *(_BYTE *)v8 != 42 || !*(v8 - 8) )
  {
LABEL_3:
    v9 = *a3 == 57;
    v120 = 0;
    v121 = &v115;
    v122 = &v115;
    if ( v9 )
    {
      if ( *(_BYTE *)v8 == 44
        && (v99 = v8, (unsigned __int8)sub_10081F0((__int64 **)&v120, *(v8 - 8)))
        && (v53 = (__int64 *)*(v99 - 4)) != 0 )
      {
        *v121 = v53;
        v10 = (__int64 *)*((_QWORD *)a3 - 4);
        if ( v10 == *v122 )
          goto LABEL_88;
      }
      else
      {
        v10 = (__int64 *)*((_QWORD *)a3 - 4);
      }
      if ( *(_BYTE *)v10 != 44 )
        goto LABEL_7;
      v101 = v10;
      if ( !(unsigned __int8)sub_10081F0((__int64 **)&v120, *(v10 - 8)) || (v77 = (__int64 *)*(v101 - 4)) == 0 )
      {
        v10 = (__int64 *)*((_QWORD *)a3 - 4);
LABEL_7:
        v8 = (__int64 *)*((_QWORD *)a3 - 8);
        goto LABEL_10;
      }
      *v121 = v77;
      v8 = (__int64 *)*((_QWORD *)a3 - 8);
      if ( v8 == *v122 )
      {
LABEL_88:
        v54 = v115;
        v55 = sub_BCB060(v115[1]);
        v56 = v55 - 1;
        LODWORD(v121) = v55;
        v57 = 1LL << ((unsigned __int8)v55 - 1);
        if ( v55 > 0x40 )
        {
          sub_C43690((__int64)&v120, 0, 0);
          if ( (unsigned int)v121 > 0x40 )
          {
            *(_QWORD *)(v120 + 8LL * (v56 >> 6)) |= v57;
            v54 = v115;
LABEL_91:
            v58 = sub_AD8D80(v54[1], (__int64)&v120);
            if ( (unsigned int)v121 > 0x40 && v120 )
              j_j___libc_free_0_0(v120);
            v124 = 257;
            v26 = (v109 == 0) + 32;
            result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
            if ( !result )
              return result;
            v29 = v115;
            v30 = v58;
LABEL_39:
            v104 = result;
            sub_1113300((__int64)result, v26, (__int64)v29, v30, (__int64)&v120);
            return v104;
          }
          v54 = v115;
        }
        else
        {
          v120 = 0;
        }
        v120 |= v57;
        goto LABEL_91;
      }
    }
LABEL_9:
    v10 = (__int64 *)*((_QWORD *)a3 - 4);
LABEL_10:
    v110 = v8;
    v111 = v10;
    if ( *(_BYTE *)v10 == 17 && *(_BYTE *)v8 == 61 )
    {
      v50 = *(v8 - 4);
      if ( *(_BYTE *)v50 == 63 )
      {
        v51 = *(_BYTE **)(v50 - 32LL * (*(_DWORD *)(v50 + 4) & 0x7FFFFFF));
        if ( *v51 == 3 )
        {
          result = (unsigned __int8 *)sub_1123BC0((_QWORD *)a1, (__int64)v8, v50, (__int64)v51, a2, v10);
          if ( result )
            return result;
        }
      }
    }
    result = 0;
    if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 > 1 )
      return result;
    v12 = *(_QWORD *)(a2 - 32);
    v13 = *(_DWORD *)(a4 + 8);
    if ( (__int64 *)v12 != v111 )
    {
      if ( v13 > 0x40 )
        goto LABEL_15;
LABEL_80:
      if ( *(_QWORD *)a4 )
        goto LABEL_27;
      goto LABEL_16;
    }
    _RSI = *(_QWORD *)a4;
    v21 = 1LL << ((unsigned __int8)v13 - 1);
    if ( v13 > 0x40 )
    {
      if ( (*(_QWORD *)(_RSI + 8LL * ((v13 - 1) >> 6)) & v21) == 0
        || (v95 = *(_DWORD *)(a4 + 8), v98 = sub_C44500(a4), v52 = sub_C44590(a4), v13 = v95, v95 != v98 + v52) )
      {
LABEL_15:
        v97 = v13;
        v14 = sub_C444A0(a4);
        v13 = v97;
        if ( v97 != v14 )
          goto LABEL_27;
LABEL_16:
        if ( v103 != 32 )
        {
          v15 = *((_QWORD *)a3 + 2);
          if ( !v15 )
          {
            v16 = *a3;
LABEL_19:
            v17 = v13;
            goto LABEL_20;
          }
          if ( *(_QWORD *)(v15 + 8) )
          {
            v16 = *a3;
            goto LABEL_66;
          }
        }
        v19 = v110;
        LOBYTE(v122) = 0;
        v120 = (unsigned __int64)&v112;
        v121 = (__int64 **)&v114;
        v123 = &v115;
        LOBYTE(v124) = 0;
        if ( *(_BYTE *)v110 == 86 )
        {
          v65 = *(_BYTE **)sub_986520((__int64)v110);
          if ( v65 )
          {
            v112 = v65;
            v66 = sub_986520((__int64)v19);
            if ( (unsigned __int8)sub_991580((__int64)&v121, *(_QWORD *)(v66 + 32)) )
            {
              v67 = sub_986520((__int64)v19);
              if ( (unsigned __int8)sub_991580((__int64)&v123, *(_QWORD *)(v67 + 64)) )
              {
                v68 = v111;
                v117 = (unsigned __int64)&v113;
                v118 = v114;
                v119 = (const void **)v115;
                if ( *(_BYTE *)v111 == 86 )
                {
                  v69 = *(__int64 **)sub_986520((__int64)v111);
                  if ( v69 )
                  {
                    v113 = v69;
                    v70 = sub_986520((__int64)v68);
                    if ( sub_10080A0(&v118, *(_QWORD *)(v70 + 32)) )
                    {
                      v71 = sub_986520((__int64)v68);
                      if ( sub_10080A0(&v119, *(_QWORD *)(v71 + 64)) )
                      {
                        v72 = (unsigned __int64 *)v114;
                        if ( !sub_9867B0((__int64)v114) )
                        {
                          v100 = v115;
                          if ( !sub_9867B0((__int64)v115) )
                          {
                            if ( *((_DWORD *)v72 + 2) <= 0x40u )
                            {
                              if ( (*v100 & *v72) == 0 )
                                goto LABEL_118;
                            }
                            else if ( !(unsigned __int8)sub_C446A0((__int64 *)v72, v100) )
                            {
LABEL_118:
                              v124 = 257;
                              v73 = sub_A825B0(*(unsigned int ***)(a1 + 32), v112, v113, (__int64)&v120);
                              if ( v103 == 33 )
                              {
                                v124 = 257;
                                v73 = sub_A82B60(*(unsigned int ***)(a1 + 32), v73, (__int64)&v120);
                              }
                              return sub_F162A0(a1, a2, v73);
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
LABEL_27:
        v15 = *((_QWORD *)a3 + 2);
        v16 = *a3;
        if ( !v15 )
        {
          v13 = *(_DWORD *)(a4 + 8);
          goto LABEL_19;
        }
LABEL_66:
        if ( !*(_QWORD *)(v15 + 8) && v16 == 57 )
        {
          v59 = (_BYTE *)*((_QWORD *)a3 - 8);
          v60 = (__int64 *)*((_QWORD *)a3 - 4);
          v61 = *((_QWORD *)v59 + 2);
          if ( v61 )
          {
            if ( !*(_QWORD *)(v61 + 8) && *v59 == 68 )
            {
              v63 = *((_QWORD *)v59 - 4);
              if ( v63 )
              {
                v110 = (__int64 *)*((_QWORD *)v59 - 4);
                if ( v60 )
                {
                  v111 = v60;
LABEL_103:
                  v64 = *(_QWORD *)(v63 + 8);
                  if ( (unsigned int)*(unsigned __int8 *)(v64 + 8) - 17 <= 1 )
                    v64 = **(_QWORD **)(v64 + 16);
                  if ( !sub_BCAC40(v64, 1) )
                  {
                    v17 = *(_DWORD *)(a4 + 8);
                    v16 = *a3;
LABEL_20:
                    if ( v17 > 0x40 )
                      goto LABEL_21;
LABEL_68:
                    if ( *(_QWORD *)a4 )
                      goto LABEL_23;
LABEL_22:
                    v120 = 0;
                    v121 = &v110;
                    v122 = &v111;
                    v18 = *((_QWORD *)a3 + 2);
                    if ( v18 && !*(_QWORD *)(v18 + 8) )
                    {
                      if ( v16 != 57 )
                        return 0;
                      v46 = *((_QWORD *)a3 - 8);
                      v47 = *(_QWORD *)(v46 + 16);
                      if ( v47
                        && !*(_QWORD *)(v47 + 8)
                        && *(_BYTE *)v46 == 54
                        && (unsigned __int8)sub_995B10((_QWORD **)&v120, *(_QWORD *)(v46 - 64))
                        && (v89 = *(__int64 **)(v46 - 32)) != 0 )
                      {
                        *v121 = v89;
                        v48 = (__int64 *)*((_QWORD *)a3 - 4);
                        if ( v48 )
                        {
                          *v122 = v48;
                          goto LABEL_173;
                        }
                      }
                      else
                      {
                        v48 = (__int64 *)*((_QWORD *)a3 - 4);
                      }
                      v49 = v48[2];
                      if ( v49 )
                      {
                        if ( !*(_QWORD *)(v49 + 8) && *(_BYTE *)v48 == 54 )
                        {
                          v108 = v48;
                          if ( (unsigned __int8)sub_995B10((_QWORD **)&v120, *(v48 - 8)) )
                          {
                            v93 = (__int64 *)*(v108 - 4);
                            if ( v93 )
                            {
                              *v121 = v93;
                              v94 = (__int64 *)*((_QWORD *)a3 - 8);
                              if ( v94 )
                              {
                                *v122 = v94;
LABEL_173:
                                v90 = v111;
                                v124 = 257;
                                v91 = sub_F94560(
                                        *(__int64 **)(a1 + 32),
                                        (__int64)v111,
                                        (__int64)v110,
                                        (__int64)&v120,
                                        0);
                                v92 = sub_AD6530(*(_QWORD *)(v91 + 8), (__int64)v90);
                                v124 = 257;
                                result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
                                if ( result )
                                {
                                  v107 = result;
                                  sub_1113300((__int64)result, v7, v91, v92, (__int64)&v120);
                                  return v107;
                                }
                                return result;
                              }
                            }
                          }
                        }
                      }
                      v16 = *a3;
                    }
LABEL_23:
                    LOBYTE(v122) = 0;
                    v120 = (unsigned __int64)&v112;
                    v121 = &v113;
                    v123 = (__int64 **)&v114;
                    if ( v16 != 57 )
                      return 0;
                    v32 = *((_QWORD *)a3 - 8);
                    v33 = *(_QWORD *)(v32 + 16);
                    if ( !v33 )
                      return 0;
                    if ( *(_QWORD *)(v33 + 8) )
                      return 0;
                    if ( *(_BYTE *)v32 != 42 )
                      return 0;
                    if ( !*(_QWORD *)(v32 - 64) )
                      return 0;
                    v112 = *(_BYTE **)(v32 - 64);
                    if ( !(unsigned __int8)sub_991580((__int64)&v121, *(_QWORD *)(v32 - 32)) )
                      return 0;
                    v34 = *((_QWORD *)a3 - 4);
                    if ( *(_BYTE *)v34 == 17 )
                    {
                      v35 = *(_DWORD *)(v34 + 32);
                      v36 = v34 + 24;
                      if ( v35 > 0x40 )
                      {
                        v78 = sub_C445E0(v34 + 24);
                        if ( v78 )
                        {
                          v79 = sub_C444A0(v34 + 24);
                          v36 = v34 + 24;
                          if ( v35 == v79 + v78 )
                            goto LABEL_53;
                        }
                      }
                      else
                      {
                        v37 = *(_QWORD *)(v34 + 24);
                        if ( v37 && (v37 & (v37 + 1)) == 0 )
                        {
LABEL_53:
                          *v123 = (__int64 *)v36;
LABEL_54:
                          v38 = (__int64 *)v114;
                          if ( (int)sub_C49970(a4, (unsigned __int64 *)v114) <= 0 )
                          {
                            v39 = v113;
                            LODWORD(v118) = *(_DWORD *)(a4 + 8);
                            if ( (unsigned int)v118 > 0x40 )
                              sub_C43780((__int64)&v117, (const void **)a4);
                            else
                              v117 = *(_QWORD *)a4;
                            sub_C46B40((__int64)&v117, v39);
                            v40 = (unsigned int)v118;
                            LODWORD(v118) = 0;
                            LODWORD(v121) = v40;
                            v120 = v117;
                            if ( v40 > 0x40 )
                            {
                              sub_C43B90(&v120, v38);
                              v116 = (unsigned int)v121;
                              v115 = (__int64 *)v120;
                              if ( (unsigned int)v118 > 0x40 && v117 )
                                j_j___libc_free_0_0(v117);
                            }
                            else
                            {
                              v41 = (__int64 *)(*v38 & v117);
                              v116 = v40;
                              v115 = v41;
                            }
                            v42 = *(unsigned int ***)(a1 + 32);
                            v124 = 257;
                            v43 = (_BYTE *)sub_AD8D80(*((_QWORD *)v112 + 1), (__int64)v114);
                            v44 = sub_A82350(v42, v112, v43, (__int64)&v120);
                            v45 = sub_AD8D80(*(_QWORD *)(v44 + 8), (__int64)&v115);
                            v124 = 257;
                            result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
                            if ( result )
                            {
                              v105 = result;
                              sub_1113300((__int64)result, v7, v44, v45, (__int64)&v120);
                              result = v105;
                            }
                            if ( v116 > 0x40 )
                            {
                              if ( v115 )
                              {
                                v106 = result;
                                j_j___libc_free_0_0(v115);
                                return v106;
                              }
                            }
                            return result;
                          }
                          return 0;
                        }
                      }
                      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v34 + 8) + 8LL) - 17 > 1 )
                        return 0;
                    }
                    else
                    {
                      v36 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v34 + 8) + 8LL) - 17;
                      if ( (unsigned int)v36 > 1 )
                        return 0;
                      if ( *(_BYTE *)v34 > 0x15u )
                        return 0;
                    }
                    v80 = sub_AD7630(v34, 1, v36);
                    if ( !v80 || *v80 != 17 )
                      return 0;
                    v81 = *((_DWORD *)v80 + 8);
                    v82 = (__int64)(v80 + 24);
                    if ( v81 > 0x40 )
                    {
                      v88 = sub_C445E0((__int64)(v80 + 24));
                      if ( !v88 )
                        return 0;
                      if ( v81 != (unsigned int)sub_C444A0(v82) + v88 )
                        return 0;
                    }
                    else
                    {
                      v83 = *((_QWORD *)v80 + 3);
                      if ( !v83 || (v83 & (v83 + 1)) != 0 )
                        return 0;
                    }
                    *v123 = (__int64 *)v82;
                    goto LABEL_54;
                  }
                  if ( !sub_9867B0(a4) )
                  {
                    v17 = *(_DWORD *)(a4 + 8);
                    if ( v17 <= 0x40 )
                    {
                      if ( *(_QWORD *)a4 != 1 )
                      {
                        v16 = *a3;
                        goto LABEL_68;
                      }
                    }
                    else if ( (unsigned int)sub_C444A0(a4) != v17 - 1 )
                    {
                      v16 = *a3;
                      goto LABEL_21;
                    }
                  }
                  v124 = 257;
                  v74 = (_BYTE *)sub_A82DA0(*(unsigned int ***)(a1 + 32), (__int64)v111, v110[1], (__int64)&v120, 0, 0);
                  if ( sub_9867B0(a4) == (v7 == 33) )
                  {
                    v124 = 257;
                    return (unsigned __int8 *)sub_B504D0(28, (__int64)v74, (__int64)v110, (__int64)&v120, 0, 0);
                  }
                  else
                  {
                    v75 = *(unsigned int ***)(a1 + 32);
                    v124 = 257;
                    v76 = sub_A82350(v75, v74, v110, (__int64)&v120);
                    v124 = 257;
                    return (unsigned __int8 *)sub_B50640(v76, (__int64)&v120, 0, 0);
                  }
                }
              }
            }
          }
          v62 = v60[2];
          if ( v62 )
          {
            if ( !*(_QWORD *)(v62 + 8) && *(_BYTE *)v60 == 68 )
            {
              v63 = *(v60 - 4);
              if ( v63 )
              {
                v110 = (__int64 *)*(v60 - 4);
                v111 = (__int64 *)v59;
                goto LABEL_103;
              }
            }
          }
        }
        v17 = *(_DWORD *)(a4 + 8);
        if ( v17 <= 0x40 )
          goto LABEL_68;
LABEL_21:
        if ( v17 != (unsigned int)sub_C444A0(a4) )
          goto LABEL_23;
        goto LABEL_22;
      }
    }
    else
    {
      if ( (v21 & _RSI) == 0 )
        goto LABEL_80;
      if ( v13 )
      {
        v22 = 64;
        if ( _RSI << (64 - (unsigned __int8)v13) != -1 )
        {
          _BitScanReverse64(&v23, ~(_RSI << (64 - (unsigned __int8)v13)));
          v22 = v23 ^ 0x3F;
        }
        __asm { tzcnt   rsi, rsi }
        if ( (unsigned int)_RSI > v13 )
          LODWORD(_RSI) = *(_DWORD *)(a4 + 8);
        if ( v22 + (_DWORD)_RSI != v13 )
          goto LABEL_80;
      }
    }
    v25 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(v12 + 8), 1, 0);
    v26 = 3 * (v103 != 32) + 34;
    v27 = sub_AD57F0(v12, v25, 0, 0);
    v124 = 257;
    v28 = v27;
    result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
    if ( !result )
      return result;
    v29 = v110;
    v30 = v28;
    goto LABEL_39;
  }
  v115 = (__int64 *)*(v8 - 8);
  if ( !(unsigned __int8)sub_995B10(&v121, *(v8 - 4))
    || (v31 = (_BYTE *)*((_QWORD *)a3 - 4), v117 = 0, v118 = (const void **)v115, *v31 != 59)
    || ((v96 = v31,
         v84 = sub_995B10((_QWORD **)&v117, *((_QWORD *)v31 - 8)),
         v85 = (const void **)*((_QWORD *)v96 - 4),
         !v84)
     || v85 != v118)
    && (!(unsigned __int8)sub_995B10((_QWORD **)&v117, (__int64)v85) || *((const void ***)v96 - 8) != v118) )
  {
    v8 = (__int64 *)*((_QWORD *)a3 - 8);
    goto LABEL_3;
  }
  v26 = (v109 == 0) + 32;
  v86 = sub_AD6530(v115[1], (__int64)v85);
  v124 = 257;
  v87 = v86;
  result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
  if ( result )
  {
    v29 = v115;
    v30 = v87;
    goto LABEL_39;
  }
  return result;
}
