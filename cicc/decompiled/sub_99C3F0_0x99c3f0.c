// Function: sub_99C3F0
// Address: 0x99c3f0
//
char __fastcall sub_99C3F0(unsigned __int8 *a1, __int64 a2, __int64 a3, _BYTE *a4, char a5)
{
  int v5; // eax
  __int64 v9; // rsi
  __int64 *v10; // r8
  int v11; // ebx
  __int64 *v12; // rsi
  int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // rsi
  char v16; // r8
  char v17; // al
  __int64 v18; // r10
  char v19; // bl
  __int64 *v20; // r15
  unsigned int v21; // eax
  __int64 v22; // rsi
  int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rcx
  unsigned int v28; // r8d
  __int64 *v29; // r15
  __int64 v30; // rsi
  __int64 v31; // rsi
  int v32; // eax
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 *v35; // r15
  __int64 *v36; // rsi
  __int64 v37; // rsi
  __int64 *v38; // r15
  int v39; // eax
  __int64 v40; // rsi
  __int64 *v41; // r15
  __int64 *v42; // r15
  int v43; // eax
  __int64 v44; // rsi
  __int64 v45; // rax
  unsigned __int8 *v46; // rdx
  __int64 v47; // rax
  unsigned __int8 *v48; // rdx
  __int64 v49; // rsi
  __int64 v50; // rsi
  __int64 *v51; // rsi
  int v52; // eax
  unsigned int v53; // r15d
  __int64 v54; // rsi
  bool v55; // al
  unsigned int v56; // edx
  int v57; // eax
  bool v58; // bl
  unsigned int v59; // eax
  __int64 *v60; // r8
  __int64 v61; // rdx
  __int64 v62; // rsi
  __int64 v63; // rdx
  unsigned __int64 v64; // rax
  __int64 v65; // rsi
  __int64 *v66; // rbx
  _QWORD *v67; // rbx
  unsigned int v69; // r15d
  bool v71; // al
  __int64 *v72; // rsi
  unsigned int v73; // r8d
  __int64 v74; // rdx
  unsigned int v75; // eax
  unsigned __int64 v76; // rdx
  unsigned __int64 v77; // rdx
  unsigned __int64 v78; // rsi
  bool v79; // zf
  unsigned __int64 v80; // rax
  __int64 v81; // rsi
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // rdx
  unsigned __int64 v85; // rax
  unsigned int v86; // eax
  int v87; // eax
  __int64 *v88; // r9
  unsigned int v89; // edx
  unsigned int v90; // eax
  __int64 v91; // rcx
  __int64 *v92; // rsi
  int v93; // eax
  bool v94; // al
  __int64 *v95; // r8
  int v96; // r12d
  __int64 v97; // rax
  __int64 v98; // rsi
  __int64 v99; // rax
  unsigned __int64 v100; // rdx
  char v101; // al
  unsigned int v102; // ebx
  int v103; // eax
  unsigned int v104; // edi
  unsigned __int64 v105; // rsi
  unsigned int v106; // eax
  unsigned __int64 v107; // rdx
  unsigned __int64 v108; // rdx
  unsigned __int64 v109; // rsi
  unsigned __int64 v110; // rax
  int v111; // eax
  int v112; // ebx
  int v113; // eax
  int v114; // eax
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // r8
  unsigned __int64 v118; // rax
  __int64 v119; // rcx
  int v120; // eax
  int v121; // eax
  int v122; // eax
  __int64 v123; // rdx
  __int64 v124; // rcx
  __int64 v125; // r8
  unsigned int v128; // eax
  __int64 *v130; // [rsp+8h] [rbp-98h]
  char v131; // [rsp+8h] [rbp-98h]
  unsigned int v132; // [rsp+8h] [rbp-98h]
  int v133; // [rsp+8h] [rbp-98h]
  __int64 v134; // [rsp+10h] [rbp-90h]
  __int64 v136; // [rsp+10h] [rbp-90h]
  __int64 v137; // [rsp+10h] [rbp-90h]
  __int64 *v138; // [rsp+10h] [rbp-90h]
  __int64 *v139; // [rsp+10h] [rbp-90h]
  __int64 *v140; // [rsp+10h] [rbp-90h]
  __int64 *v141; // [rsp+10h] [rbp-90h]
  unsigned int v142; // [rsp+18h] [rbp-88h]
  unsigned int v143; // [rsp+18h] [rbp-88h]
  unsigned int v144; // [rsp+18h] [rbp-88h]
  __int64 *v145; // [rsp+28h] [rbp-78h] BYREF
  __int64 v146[2]; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v147; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v148; // [rsp+48h] [rbp-58h]
  unsigned __int64 v149; // [rsp+50h] [rbp-50h] BYREF
  __int64 v150; // [rsp+58h] [rbp-48h]
  unsigned __int64 v151; // [rsp+60h] [rbp-40h] BYREF
  __int64 v152; // [rsp+68h] [rbp-38h]

  v142 = *(_DWORD *)(a2 + 8);
  LOBYTE(v5) = *a1 - 42;
  switch ( *a1 )
  {
    case '*':
      v15 = *((_QWORD *)a1 - 4);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      LOBYTE(v5) = sub_991580((__int64)&v151, v15);
      if ( !(_BYTE)v5 )
        return v5;
      v130 = v145;
      LOBYTE(v5) = sub_9867B0((__int64)v145);
      if ( (_BYTE)v5 )
        return v5;
      v16 = a5;
      if ( !*a4 )
        return v5;
      v136 = (__int64)v130;
      v131 = v16;
      v17 = sub_B44900(a1);
      v18 = v136;
      v19 = v17;
      if ( v17 && v131 )
        goto LABEL_19;
      LOBYTE(v5) = sub_B448F0(a1);
      v18 = v136;
      if ( (_BYTE)v5 )
      {
        LOBYTE(v5) = sub_986BE0(a2, v136);
      }
      else if ( v19 )
      {
LABEL_19:
        v137 = v18;
        if ( sub_986C60((__int64 *)v18, *(_DWORD *)(v18 + 8) - 1) )
        {
          sub_986680((__int64)&v151, v142);
          sub_984320((__int64 *)a2, (__int64 *)&v151);
          sub_969240((__int64 *)&v151);
          v20 = v145;
          sub_9865E0((__int64)&v147, v142);
          sub_C45EE0(&v147, v20);
          v21 = v148;
          v148 = 0;
          LODWORD(v150) = v21;
          v149 = v147;
          goto LABEL_24;
        }
        sub_986680((__int64)&v149, v142);
        sub_C45EE0(&v149, v137);
        v122 = v150;
        LODWORD(v150) = 0;
        LODWORD(v152) = v122;
        v151 = v149;
        sub_984320((__int64 *)a2, (__int64 *)&v151);
        sub_969240((__int64 *)&v151);
        sub_969240((__int64 *)&v149);
        sub_9865E0((__int64)&v149, v142);
        goto LABEL_141;
      }
      return v5;
    case '0':
      v22 = *((_QWORD *)a1 - 4);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      if ( (unsigned __int8)sub_991580((__int64)&v151, v22) && !sub_9867B0((__int64)v145) )
      {
        sub_9691E0((__int64)&v147, v142, -1, 1u, 0);
        sub_C4A1D0(&v149, &v147, v145);
LABEL_24:
        sub_C46A40(&v149, 1);
        v23 = v150;
        LODWORD(v150) = 0;
        LODWORD(v152) = v23;
        v151 = v149;
        sub_984320((__int64 *)a3, (__int64 *)&v151);
        sub_969240((__int64 *)&v151);
        sub_969240((__int64 *)&v149);
        LOBYTE(v5) = sub_969240((__int64 *)&v147);
        return v5;
      }
      v49 = *((_QWORD *)a1 - 8);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      LOBYTE(v5) = sub_991580((__int64)&v151, v49);
      if ( !(_BYTE)v5 )
        return v5;
      goto LABEL_8;
    case '1':
      v24 = *((_QWORD *)a1 - 4);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      if ( (unsigned __int8)sub_991580((__int64)&v151, v24) )
      {
        sub_986680((__int64)v146, v142);
        sub_9865E0((__int64)&v147, v142);
        v88 = v145;
        v89 = *((_DWORD *)v145 + 2);
        if ( v89 )
        {
          if ( v89 <= 0x40 )
          {
            v118 = *v145;
            if ( *v145 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v89) )
            {
              v91 = v89 - 64;
              if ( v118 )
              {
                _BitScanReverse64(&v118, v118);
                v89 = v91 + (v118 ^ 0x3F);
              }
              goto LABEL_118;
            }
          }
          else
          {
            v133 = *((_DWORD *)v145 + 2);
            v140 = v145;
            if ( v133 != (unsigned int)sub_C445E0(v145) )
            {
              v90 = sub_C444A0(v140);
              v88 = v140;
              v89 = v90;
LABEL_118:
              if ( v142 - 1 > v89 )
              {
                sub_C4A3E0(&v151, v146, v88, v91);
                sub_984320((__int64 *)a2, (__int64 *)&v151);
                sub_969240((__int64 *)&v151);
                sub_C4A3E0(&v151, &v147, v145, v119);
                sub_984320((__int64 *)a3, (__int64 *)&v151);
                sub_969240((__int64 *)&v151);
                if ( (int)sub_C4C880(a2, a3) > 0 )
                {
                  v120 = *(_DWORD *)(a2 + 8);
                  *(_DWORD *)(a2 + 8) = 0;
                  LODWORD(v152) = v120;
                  v151 = *(_QWORD *)a2;
                  *(_QWORD *)a2 = *(_QWORD *)a3;
                  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a3 + 8);
                  *(_DWORD *)(a3 + 8) = 0;
                  sub_984320((__int64 *)a3, (__int64 *)&v151);
                  sub_969240((__int64 *)&v151);
                }
                sub_9865C0((__int64)&v149, a3);
                sub_C46A40(&v149, 1);
                v121 = v150;
                LODWORD(v150) = 0;
                LODWORD(v152) = v121;
                v151 = v149;
                sub_984320((__int64 *)a3, (__int64 *)&v151);
                sub_969240((__int64 *)&v151);
                sub_969240((__int64 *)&v149);
              }
              goto LABEL_119;
            }
          }
        }
        sub_9865C0((__int64)&v149, (__int64)v146);
        sub_C46A40(&v149, 1);
        v113 = v150;
        LODWORD(v150) = 0;
        LODWORD(v152) = v113;
        v151 = v149;
        sub_984320((__int64 *)a2, (__int64 *)&v151);
        sub_969240((__int64 *)&v151);
        sub_969240((__int64 *)&v149);
        sub_9865C0((__int64)&v149, (__int64)&v147);
        sub_C46A40(&v149, 1);
        v114 = v150;
        LODWORD(v150) = 0;
        LODWORD(v152) = v114;
        v151 = v149;
        sub_984320((__int64 *)a3, (__int64 *)&v151);
        sub_969240((__int64 *)&v151);
        sub_969240((__int64 *)&v149);
LABEL_119:
        sub_969240((__int64 *)&v147);
        LOBYTE(v5) = sub_969240(v146);
        return v5;
      }
      v25 = *((_QWORD *)a1 - 8);
      v151 = (unsigned __int64)&v145;
      LOBYTE(v152) = 0;
      LOBYTE(v5) = sub_991580((__int64)&v151, v25);
      if ( !(_BYTE)v5 )
        return v5;
      v29 = v145;
      if ( !(unsigned __int8)sub_986B30(v145, v25, v26, v27, v28) )
      {
        v104 = *((_DWORD *)v29 + 2);
        v105 = *v29;
        if ( v104 > 0x40 )
          v105 = *(_QWORD *)(v105 + 8LL * ((v104 - 1) >> 6));
        v106 = *((_DWORD *)v29 + 2);
        if ( (v105 & (1LL << ((unsigned __int8)v104 - 1))) == 0 )
        {
          LODWORD(v150) = *((_DWORD *)v29 + 2);
          if ( v106 > 0x40 )
            sub_C43780(&v149, v29);
          else
            v149 = *v29;
          goto LABEL_151;
        }
        LODWORD(v152) = *((_DWORD *)v29 + 2);
        if ( v106 > 0x40 )
        {
          sub_C43780(&v151, v29);
          v106 = v152;
          if ( (unsigned int)v152 > 0x40 )
          {
            sub_C43D10(&v151, v29, v123, v124, v125);
LABEL_150:
            sub_C46250(&v151);
            LODWORD(v150) = v152;
            v149 = v151;
LABEL_151:
            sub_C46A40(&v149, 1);
            v111 = v150;
            LODWORD(v150) = 0;
            LODWORD(v152) = v111;
            v151 = v149;
            sub_984320((__int64 *)a3, (__int64 *)&v151);
            sub_969240((__int64 *)&v151);
            sub_969240((__int64 *)&v149);
            v81 = a3;
            sub_9865C0((__int64)&v147, a3);
            if ( v148 > 0x40 )
            {
LABEL_123:
              sub_C43D10(&v147, v81, v84, v82, v83);
            }
            else
            {
              v147 = ~v147;
              sub_9842C0(&v147);
            }
LABEL_113:
            sub_C46250(&v147);
            v86 = v148;
            v148 = 0;
            LODWORD(v150) = v86;
            v149 = v147;
            sub_C46A40(&v149, 1);
            v87 = v150;
            LODWORD(v150) = 0;
            LODWORD(v152) = v87;
            v151 = v149;
            sub_984320((__int64 *)a2, (__int64 *)&v151);
            sub_969240((__int64 *)&v151);
            sub_969240((__int64 *)&v149);
            LOBYTE(v5) = sub_969240((__int64 *)&v147);
            return v5;
          }
          v107 = v151;
        }
        else
        {
          v107 = *v29;
        }
        v108 = ~v107;
        v109 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v106;
        v79 = v106 == 0;
        v110 = 0;
        if ( !v79 )
          v110 = v109;
        v151 = v110 & v108;
        goto LABEL_150;
      }
      if ( *(_DWORD *)(a2 + 8) <= 0x40u && *((_DWORD *)v29 + 2) <= 0x40u )
      {
        *(_QWORD *)a2 = *v29;
        *(_DWORD *)(a2 + 8) = *((_DWORD *)v29 + 2);
      }
      else
      {
        sub_C43990(a2, v29);
      }
      sub_9865C0((__int64)&v149, a2);
      if ( (unsigned int)v150 > 0x40 )
      {
        sub_C482E0(&v149, 1);
      }
      else if ( (_DWORD)v150 == 1 )
      {
        v149 = 0;
      }
      else
      {
        v149 >>= 1;
      }
LABEL_141:
      sub_C46A40(&v149, 1);
      v103 = v150;
      LODWORD(v150) = 0;
      LODWORD(v152) = v103;
      v151 = v149;
      sub_984320((__int64 *)a3, (__int64 *)&v151);
      sub_969240((__int64 *)&v151);
      LOBYTE(v5) = sub_969240((__int64 *)&v149);
      return v5;
    case '3':
      v30 = *((_QWORD *)a1 - 4);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      if ( (unsigned __int8)sub_991580((__int64)&v151, v30) )
      {
        v92 = v145;
        if ( *(_DWORD *)(a3 + 8) <= 0x40u && *((_DWORD *)v145 + 2) <= 0x40u )
        {
          *(_QWORD *)a3 = *v145;
          v5 = *((_DWORD *)v92 + 2);
          *(_DWORD *)(a3 + 8) = v5;
        }
        else
        {
          LOBYTE(v5) = sub_C43990(a3, v145);
        }
      }
      else
      {
        v31 = *((_QWORD *)a1 - 8);
        v151 = (unsigned __int64)&v145;
        LOBYTE(v152) = 0;
        LOBYTE(v5) = sub_991580((__int64)&v151, v31);
        if ( (_BYTE)v5 )
        {
          sub_9865C0((__int64)&v149, (__int64)v145);
          sub_C46A40(&v149, 1);
          v32 = v150;
          LODWORD(v150) = 0;
          LODWORD(v152) = v32;
          v151 = v149;
          sub_984320((__int64 *)a3, (__int64 *)&v151);
          sub_969240((__int64 *)&v151);
          LOBYTE(v5) = sub_969240((__int64 *)&v149);
        }
      }
      return v5;
    case '4':
      v33 = *((_QWORD *)a1 - 4);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      if ( !(unsigned __int8)sub_991580((__int64)&v151, v33) )
      {
        v34 = *((_QWORD *)a1 - 8);
        v151 = (unsigned __int64)&v145;
        LOBYTE(v152) = 0;
        LOBYTE(v5) = sub_991580((__int64)&v151, v34);
        if ( !(_BYTE)v5 )
          return v5;
        v35 = v145;
        if ( !sub_986C60(v145, *((_DWORD *)v145 + 2) - 1) )
        {
          v12 = v35;
          goto LABEL_9;
        }
        if ( *(_DWORD *)(a3 + 8) > 0x40u )
        {
          **(_QWORD **)a3 = 1;
          memset(
            (void *)(*(_QWORD *)a3 + 8LL),
            0,
            8 * (unsigned int)(((unsigned __int64)*(unsigned int *)(a3 + 8) + 63) >> 6) - 8);
        }
        else
        {
          *(_QWORD *)a3 = 1;
          sub_9842C0((unsigned __int64 *)a3);
        }
LABEL_42:
        v36 = v145;
        if ( *(_DWORD *)(a2 + 8) <= 0x40u && *((_DWORD *)v145 + 2) <= 0x40u )
        {
          *(_QWORD *)a2 = *v145;
          v5 = *((_DWORD *)v36 + 2);
          *(_DWORD *)(a2 + 8) = v5;
        }
        else
        {
          LOBYTE(v5) = sub_C43990(a2, v145);
        }
        return v5;
      }
      v72 = v145;
      v73 = *((_DWORD *)v145 + 2);
      v74 = *v145;
      if ( v73 > 0x40 )
        v74 = *(_QWORD *)(v74 + 8LL * ((v73 - 1) >> 6));
      v75 = *((_DWORD *)v145 + 2);
      if ( (v74 & (1LL << ((unsigned __int8)v73 - 1))) == 0 )
      {
        LODWORD(v150) = *((_DWORD *)v145 + 2);
        if ( v75 > 0x40 )
          sub_C43780(&v149, v145);
        else
          v149 = *v145;
        goto LABEL_109;
      }
      LODWORD(v152) = *((_DWORD *)v145 + 2);
      if ( v75 > 0x40 )
      {
        sub_C43780(&v151, v145);
        v75 = v152;
        if ( (unsigned int)v152 > 0x40 )
        {
          sub_C43D10(&v151, v72, v115, v116, v117);
LABEL_108:
          sub_C46250(&v151);
          LODWORD(v150) = v152;
          v149 = v151;
LABEL_109:
          sub_984320((__int64 *)a3, (__int64 *)&v149);
          sub_969240((__int64 *)&v149);
          v81 = a3;
          sub_9865C0((__int64)&v147, a3);
          v84 = v148;
          if ( v148 > 0x40 )
            goto LABEL_123;
          v85 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v148) & ~v147;
          if ( !v148 )
            v85 = 0;
          v147 = v85;
          goto LABEL_113;
        }
        v76 = v151;
      }
      else
      {
        v76 = *v145;
      }
      v77 = ~v76;
      v78 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v75;
      v79 = v75 == 0;
      v80 = 0;
      if ( !v79 )
        v80 = v78;
      v151 = v80 & v77;
      goto LABEL_108;
    case '6':
      v9 = *((_QWORD *)a1 - 8);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      if ( !(unsigned __int8)sub_991580((__int64)&v151, v9) )
      {
        v65 = *((_QWORD *)a1 - 4);
        v151 = (unsigned __int64)&v145;
        LOBYTE(v152) = 0;
        LOBYTE(v5) = sub_991580((__int64)&v151, v65);
        if ( !(_BYTE)v5 )
          return v5;
        v66 = v145;
        LOBYTE(v5) = sub_986EE0((__int64)v145, v142);
        if ( !(_BYTE)v5 )
          return v5;
        if ( *((_DWORD *)v66 + 2) <= 0x40u )
          v67 = (_QWORD *)*v66;
        else
          v67 = *(_QWORD **)*v66;
        sub_9691E0((__int64)&v149, v142, 0, 0, 0);
        sub_9870B0((__int64)&v149, (unsigned int)v67, (unsigned int)v150);
        goto LABEL_10;
      }
      v10 = v145;
      if ( *a4 )
      {
        v141 = v145;
        v101 = sub_B448F0(a1);
        v10 = v141;
        if ( v101 )
        {
          sub_986BE0(a2, (__int64)v141);
          v102 = sub_9871A0(a2);
          sub_9865C0((__int64)&v149, a2);
          sub_984A70((__int64)&v149, v102);
          goto LABEL_141;
        }
      }
      v134 = (__int64)v10;
      if ( (unsigned __int8)sub_B44900(a1) )
      {
        if ( sub_986C60((__int64 *)v134, *(_DWORD *)(v134 + 8) - 1) )
        {
          v11 = sub_9871D0(v134);
          sub_9865C0((__int64)&v151, v134);
          sub_984A70((__int64)&v151, (unsigned int)(v11 - 1));
          goto LABEL_7;
        }
        v112 = sub_9871A0(v134);
        sub_986BE0(a2, v134);
        sub_9865C0((__int64)&v149, (__int64)v145);
        sub_984A70((__int64)&v149, (unsigned int)(v112 - 1));
LABEL_10:
        sub_C46A40(&v149, 1);
        v13 = v150;
        LODWORD(v150) = 0;
        LODWORD(v152) = v13;
        v151 = v149;
        sub_984320((__int64 *)a3, (__int64 *)&v151);
        sub_969240((__int64 *)&v151);
        LOBYTE(v5) = sub_969240((__int64 *)&v149);
        return v5;
      }
      v94 = sub_986C60((__int64 *)v134, 0);
      v95 = (__int64 *)v134;
      if ( !v94 )
        goto LABEL_125;
      LODWORD(v152) = v142;
      if ( v142 > 0x40 )
      {
        sub_C43690(&v151, 0, 0);
        if ( (unsigned int)v152 > 0x40 )
        {
          *(_QWORD *)v151 |= 1uLL;
          goto LABEL_172;
        }
      }
      else
      {
        v151 = 0;
      }
      v151 |= 1u;
LABEL_172:
      sub_984320((__int64 *)a2, (__int64 *)&v151);
      sub_969240((__int64 *)&v151);
      v95 = v145;
LABEL_125:
      if ( *((_DWORD *)v95 + 2) > 0x40u )
        v96 = sub_C44630(v95);
      else
        v96 = sub_39FAC40(*v95);
      LODWORD(v150) = v142;
      if ( v142 > 0x40 )
      {
        sub_C43690(&v149, 0, 0);
        v142 = v150;
      }
      else
      {
        v149 = 0;
      }
      sub_9870B0((__int64)&v149, v142 - v96, v142);
      goto LABEL_141;
    case '7':
      v37 = *((_QWORD *)a1 - 4);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      if ( (unsigned __int8)sub_991580((__int64)&v151, v37) && sub_986EE0((__int64)v145, v142) )
      {
        sub_9691E0((__int64)&v147, v142, -1, 1u, 0);
        v38 = v145;
        sub_9865C0((__int64)&v149, (__int64)&v147);
        sub_C48380(&v149, v38);
        sub_C46A40(&v149, 1);
        v39 = v150;
        LODWORD(v150) = 0;
        LODWORD(v152) = v39;
        v151 = v149;
        sub_984320((__int64 *)a3, (__int64 *)&v151);
        sub_969240((__int64 *)&v151);
        sub_969240((__int64 *)&v149);
        LOBYTE(v5) = sub_969240((__int64 *)&v147);
        return v5;
      }
      v50 = *((_QWORD *)a1 - 8);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      LOBYTE(v5) = sub_991580((__int64)&v151, v50);
      if ( !(_BYTE)v5 )
        return v5;
      v138 = v145;
      v143 = v142 - 1;
      v51 = v145;
      if ( !sub_9867B0((__int64)v145) )
      {
        if ( *a4 )
        {
          v52 = *a1;
          if ( ((unsigned __int8)(v52 - 55) <= 1u || (unsigned int)(v52 - 48) <= 1) && (a1[1] & 2) != 0 )
          {
            v53 = *((_DWORD *)v138 + 2);
            if ( v53 <= 0x40 )
            {
              _RAX = *v138;
              __asm { tzcnt   rdx, rax }
              v128 = 64;
              if ( *v138 )
                v128 = _RDX;
              if ( v53 <= v128 )
                v128 = *((_DWORD *)v138 + 2);
              v143 = v128;
            }
            else
            {
              v51 = v138;
              v143 = sub_C44590(v138);
            }
          }
        }
      }
      sub_9865C0((__int64)&v151, (__int64)v51);
      if ( (unsigned int)v152 > 0x40 )
      {
        sub_C482E0(&v151, v143);
      }
      else if ( v143 == (_DWORD)v152 )
      {
        v151 = 0;
      }
      else
      {
        v151 >>= v143;
      }
      goto LABEL_7;
    case '8':
      v40 = *((_QWORD *)a1 - 4);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      if ( (unsigned __int8)sub_991580((__int64)&v151, v40) && sub_986EE0((__int64)v145, v142) )
      {
        sub_986680((__int64)&v149, v142);
        v41 = v145;
        sub_9865C0((__int64)&v151, (__int64)&v149);
        sub_C44D10(&v151, v41);
        sub_984320((__int64 *)a2, (__int64 *)&v151);
        sub_969240((__int64 *)&v151);
        sub_969240((__int64 *)&v149);
        sub_9865E0((__int64)&v147, v142);
        v42 = v145;
        sub_9865C0((__int64)&v149, (__int64)&v147);
        sub_C44D10(&v149, v42);
        sub_C46A40(&v149, 1);
        v43 = v150;
        LODWORD(v150) = 0;
        LODWORD(v152) = v43;
        v151 = v149;
        sub_984320((__int64 *)a3, (__int64 *)&v151);
        sub_969240((__int64 *)&v151);
        sub_969240((__int64 *)&v149);
        LOBYTE(v5) = sub_969240((__int64 *)&v147);
        return v5;
      }
      v54 = *((_QWORD *)a1 - 8);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      LOBYTE(v5) = sub_991580((__int64)&v151, v54);
      if ( !(_BYTE)v5 )
        return v5;
      v139 = v145;
      v144 = v142 - 1;
      v55 = sub_9867B0((__int64)v145);
      v56 = *((_DWORD *)v139 + 2);
      if ( v55 )
        goto LABEL_96;
      if ( !*a4 )
        goto LABEL_96;
      v57 = *a1;
      if ( (unsigned int)(v57 - 48) > 1 && (unsigned __int8)(v57 - 55) > 1u )
        goto LABEL_96;
      if ( (a1[1] & 2) == 0 )
        goto LABEL_96;
      if ( v56 <= 0x40 )
      {
        _RAX = *v139;
        v69 = 64;
        __asm { tzcnt   rcx, rax }
        if ( *v139 )
          v69 = _RCX;
        if ( v69 > v56 )
          v69 = *((_DWORD *)v139 + 2);
        v144 = v69;
LABEL_96:
        v132 = *((_DWORD *)v139 + 2);
        v71 = sub_986C60(v139, v56 - 1);
        v60 = v139;
        if ( v71 )
        {
          if ( *(_DWORD *)(a2 + 8) <= 0x40u && v132 <= 0x40 )
          {
            *(_QWORD *)a2 = *v139;
            *(_DWORD *)(a2 + 8) = *((_DWORD *)v139 + 2);
LABEL_78:
            sub_9865C0((__int64)&v149, (__int64)v60);
            if ( (unsigned int)v150 > 0x40 )
            {
              sub_C44B70(&v149, v144);
            }
            else
            {
              v61 = 0;
              if ( (_DWORD)v150 )
                v61 = (__int64)(v149 << (64 - (unsigned __int8)v150)) >> (64 - (unsigned __int8)v150);
              v62 = v61 >> 63;
              v63 = v61 >> v144;
              if ( v144 != (_DWORD)v150 )
                v62 = v63;
              v64 = 0;
              if ( (_DWORD)v150 )
                v64 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v150;
              v149 = v62 & v64;
            }
            goto LABEL_141;
          }
LABEL_77:
          sub_C43990(a2, v60);
          v60 = v145;
          goto LABEL_78;
        }
      }
      else
      {
        v58 = sub_986C60(v139, v56 - 1);
        v59 = sub_C44590(v139);
        v60 = v139;
        v144 = v59;
        if ( v58 )
          goto LABEL_77;
      }
      sub_9865C0((__int64)&v151, (__int64)v60);
      if ( (unsigned int)v152 > 0x40 )
      {
        sub_C44B70(&v151, v144);
      }
      else
      {
        v97 = 0;
        if ( (_DWORD)v152 )
          v97 = (__int64)(v151 << (64 - (unsigned __int8)v152)) >> (64 - (unsigned __int8)v152);
        v98 = v97 >> 63;
        v99 = v97 >> v144;
        if ( v144 == (_DWORD)v152 )
          v99 = v98;
        v100 = 0;
        if ( (_DWORD)v152 )
          v100 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v152;
        v151 = v100 & v99;
      }
LABEL_7:
      sub_984320((__int64 *)a2, (__int64 *)&v151);
      sub_969240((__int64 *)&v151);
LABEL_8:
      v12 = v145;
LABEL_9:
      sub_9865C0((__int64)&v149, (__int64)v12);
      goto LABEL_10;
    case '9':
      v44 = *((_QWORD *)a1 - 4);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      if ( (unsigned __int8)sub_991580((__int64)&v151, v44) )
      {
        sub_9865C0((__int64)&v149, (__int64)v145);
        sub_C46A40(&v149, 1);
        v93 = v150;
        LODWORD(v150) = 0;
        LODWORD(v152) = v93;
        v151 = v149;
        sub_984320((__int64 *)a3, (__int64 *)&v151);
        sub_969240((__int64 *)&v151);
        sub_969240((__int64 *)&v149);
      }
      v45 = *((_QWORD *)a1 - 4);
      v46 = (unsigned __int8 *)*((_QWORD *)a1 - 8);
      v149 = 0;
      v150 = v45;
      if ( !sub_99C280((__int64)&v149, 15, v46) )
      {
        v47 = *((_QWORD *)a1 - 8);
        v48 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
        v151 = 0;
        v152 = v47;
        LOBYTE(v5) = sub_99C280((__int64)&v151, 15, v48);
        if ( !(_BYTE)v5 )
          return v5;
      }
      sub_986680((__int64)&v149, v142);
      goto LABEL_141;
    case ':':
      v14 = *((_QWORD *)a1 - 4);
      LOBYTE(v152) = 0;
      v151 = (unsigned __int64)&v145;
      LOBYTE(v5) = sub_991580((__int64)&v151, v14);
      if ( (_BYTE)v5 )
        goto LABEL_42;
      return v5;
    default:
      return v5;
  }
}
