// Function: sub_71D150
// Address: 0x71d150
//
__int64 __fastcall sub_71D150(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // rcx
  __int64 i; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  int v13; // eax
  __int64 j; // r13
  __int64 v15; // rax
  _QWORD *v16; // rbx
  __int64 k; // r15
  __int64 v18; // rcx
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // r15
  __int64 v26; // r14
  unsigned __int16 v27; // ax
  __int64 v28; // rbx
  _QWORD *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 *v34; // r13
  __int64 *m; // rbx
  char v36; // al
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 *v40; // rax
  __int64 v41; // r12
  __int64 v42; // r15
  __int64 n; // rbx
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 *v47; // rax
  __int64 *v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rbx
  __int64 v51; // rax
  __int64 *v52; // rbx
  __int64 *v53; // r12
  __int64 *v54; // rbx
  char v55; // al
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rax
  __int64 v61; // r15
  __int64 kk; // r13
  __int64 v63; // rdx
  __int64 v64; // rdx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // r13
  __int64 v70; // r14
  _QWORD *v71; // rbx
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 mm; // r8
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // r15
  __int64 v78; // r14
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // r14
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 nn; // rbx
  __int64 v85; // rax
  __int64 v86; // r12
  __int64 v87; // rax
  __int64 *v88; // r12
  __int64 v89; // rax
  __int64 v90; // rdi
  __int64 v91; // rax
  __int64 v92; // rax
  _QWORD *v93; // r12
  __int64 v94; // rax
  __int64 v95; // rdi
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // r8
  __int64 v99; // r9
  __int64 ii; // r8
  __int64 v101; // rdx
  __int64 v102; // rdx
  __int64 v103; // r15
  __int64 v104; // r14
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // r14
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 jj; // r13
  __int64 v111; // rax
  __int64 v112; // r13
  __int64 v113; // rax
  __int64 *v114; // r13
  __int64 v115; // rax
  __int64 v116; // rdi
  __int64 v117; // rax
  __int64 v118; // rax
  _QWORD *v119; // r13
  __int64 v120; // rax
  __int64 v121; // rdi
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 v124; // rax
  _QWORD *v125; // rax
  __int64 v126; // [rsp-120h] [rbp-120h]
  __int64 v127; // [rsp-118h] [rbp-118h]
  __int64 v128; // [rsp-110h] [rbp-110h]
  __int64 v129; // [rsp-108h] [rbp-108h]
  __int64 v130; // [rsp-108h] [rbp-108h]
  __int64 v131; // [rsp-100h] [rbp-100h]
  __int16 v132; // [rsp-F0h] [rbp-F0h]
  __int64 v133; // [rsp-E8h] [rbp-E8h]
  int v134; // [rsp-E8h] [rbp-E8h]
  __int16 v135; // [rsp-E0h] [rbp-E0h]
  __int64 v136; // [rsp-E0h] [rbp-E0h]
  int v137; // [rsp-D8h] [rbp-D8h]
  __int64 v138; // [rsp-D8h] [rbp-D8h]
  __int64 v139; // [rsp-D0h] [rbp-D0h]
  __int64 v140; // [rsp-C8h] [rbp-C8h]
  __int64 v141; // [rsp-C8h] [rbp-C8h]
  __int64 v142; // [rsp-C0h] [rbp-C0h]
  __int64 *v143; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD *v144; // [rsp-B0h] [rbp-B0h] BYREF
  unsigned int v145[3]; // [rsp-A4h] [rbp-A4h] BYREF
  __int64 v146[2]; // [rsp-98h] [rbp-98h] BYREF
  __int64 v147; // [rsp-88h] [rbp-88h]

  result = *(unsigned __int8 *)(a1 + 193);
  v7 = *(_BYTE *)(a1 + 193) & 0x10;
  LOBYTE(v7) = *(_BYTE *)(a1 + 206) & 8 | v7;
  if ( !(_BYTE)v7 )
    return result;
  result &= 0x20u;
  if ( (unsigned __int8)result | *(_BYTE *)(a1 + 206) & 0x10 )
    return result;
  v8 = a1;
  if ( *(_DWORD *)(a1 + 160) || *(_QWORD *)(a1 + 344) )
    return result;
  v9 = *(unsigned __int8 *)(a1 + 174);
  result = (unsigned int)(v9 - 1);
  if ( (unsigned __int8)(v9 - 1) <= 1u )
    goto LABEL_12;
  if ( (_BYTE)v9 != 5 )
  {
    if ( (_BYTE)v9 == 3 && (*(_BYTE *)(a1 + 195) & 1) != 0 )
      return sub_71CE90(a1);
    return result;
  }
  v13 = *(unsigned __int8 *)(a1 + 176);
  if ( (_BYTE)v13 == 15 )
  {
LABEL_12:
    for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v11 = *(_BYTE *)(a1 + 206) & 0x18;
    v12 = *(_QWORD *)(*(_QWORD *)i + 96LL);
    if ( (_BYTE)v11 != 8 )
      goto LABEL_15;
    if ( sub_5EB8C0(a1) )
      goto LABEL_40;
    v11 = dword_4F077BC;
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 > 0x1869Fu )
LABEL_39:
          sub_5F9380(a1, v7, dword_4F077BC, v33, a5);
LABEL_40:
        v9 = *(unsigned __int8 *)(a1 + 174);
LABEL_15:
        if ( (_BYTE)v9 != 1 )
          return sub_71BE30(a1);
        if ( (*(_BYTE *)(v12 + 183) & 0x40) == 0 )
          return sub_71BE30(a1);
        result = sub_72F310(a1, 1, v11, v9, a5, a6);
        if ( !(_DWORD)result )
          return sub_71BE30(a1);
        *(_BYTE *)(v12 + 183) |= 0x80u;
        return result;
      }
    }
    else if ( !(_DWORD)qword_4F077B4 )
    {
      goto LABEL_39;
    }
    if ( qword_4F077A0 <= 0x15F8Fu )
      goto LABEL_40;
    goto LABEL_39;
  }
  if ( (unsigned __int8)(v13 - 30) <= 4u || (result = (unsigned int)(v13 - 16), (unsigned __int8)result <= 1u) )
  {
    for ( j = *(_QWORD *)(a1 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v15 = *(_QWORD *)(j + 168);
    v16 = *(_QWORD **)v15;
    result = sub_8D2FB0(*(_QWORD *)(*(_QWORD *)v15 + 8LL));
    if ( (_DWORD)result )
    {
      result = sub_8D46C0(v16[1]);
      for ( k = result; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
    }
    else
    {
      for ( k = v16[1]; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
    }
    if ( (*(_BYTE *)(k + 177) & 0x20) == 0 )
    {
      v142 = sub_71B6A0(a1, j, k, v145);
      do
      {
        v19 = v16[1];
        v20 = sub_735FB0(v19, 3, 0xFFFFFFFFLL, v18);
        *(_BYTE *)(v20 + 169) |= 0x80u;
        v21 = v20;
        *(_BYTE *)(v20 + 89) |= 1u;
        *(_QWORD *)(v20 + 256) = v19;
        sub_72FBE0(v20);
        v16 = (_QWORD *)*v16;
      }
      while ( v16 );
      v22 = *(unsigned __int8 *)(v8 + 176);
      switch ( (_BYTE)v22 )
      {
        case 0x1E:
          v34 = v146;
          v137 = dword_4F07508[0];
          v135 = dword_4F07508[1];
          *(_QWORD *)dword_4F07508 = *(_QWORD *)(k + 64);
          v140 = sub_726B30(11);
          *(_QWORD *)(v142 + 80) = v140;
          v147 = 0;
          for ( m = **(__int64 ***)(k + 168); m; m = (__int64 *)*m )
          {
            v36 = *((_BYTE *)m + 96);
            if ( (v36 & 1) != 0 && ((v36 & 2) == 0 || !(unsigned int)sub_8E35E0(m, k)) )
            {
              sub_71ADE0(&v143, (__int64 *)&v144);
              v37 = sub_73E4A0(v143, m);
              v143 = (__int64 *)sub_73DCD0(v37);
              v38 = sub_73E4A0(v144, m);
              v144 = (_QWORD *)sub_73DCD0(v38);
              v39 = sub_692440((__int64)v143, (__int64)v144);
              v40 = sub_71B3F0(v39);
              v34[2] = (__int64)v40;
              v34 = v40;
              v40[3] = v140;
            }
          }
          if ( **(_QWORD **)(*(_QWORD *)k + 96LL) )
          {
            v133 = v8;
            v41 = **(_QWORD **)(*(_QWORD *)k + 96LL);
            do
            {
              if ( *(_BYTE *)(v41 + 80) == 8 )
              {
                v42 = *(_QWORD *)(v41 + 88);
                for ( n = *(_QWORD *)(v42 + 120); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
                  ;
                if ( (unsigned int)sub_8D3410(n) )
                {
                  for ( ii = sub_8D40F0(n); *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
                    ;
                  v128 = ii;
                  sub_71ADE0(&v143, (__int64 *)&v144);
                  v143 = (__int64 *)sub_73E470(v143, v42, v101);
                  v144 = (_QWORD *)sub_73E470(v144, v42, v102);
                  v130 = sub_72BA30(byte_4F06A51[0]);
                  v103 = sub_736020(v130, 0);
                  v104 = sub_731250(v103);
                  v105 = sub_73A830(0, byte_4F06A51[0]);
                  v106 = sub_73E690(v104, v105);
                  v34[2] = v106;
                  v107 = v106;
                  *(_QWORD *)(v106 + 24) = v140;
                  v108 = sub_731250(v103);
                  v109 = sub_73DBF0(37, v130, v108);
                  for ( jj = v109; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
                    ;
                  *(_QWORD *)(v109 + 16) = sub_73A8E0(*(_QWORD *)(n + 128) / *(_QWORD *)(v128 + 128), byte_4F06A51[0]);
                  v111 = sub_6EFF80();
                  v112 = sub_73DBF0(61, v111, jj);
                  v113 = sub_726B30(12);
                  *(_QWORD *)(v107 + 16) = v113;
                  v48 = (__int64 *)v113;
                  *(_QWORD *)(v113 + 48) = v112;
                  *(_QWORD *)(v113 + 24) = v140;
                  v114 = sub_71AE50((__int64)v143);
                  v143 = v114;
                  v115 = sub_73E830(v103);
                  v116 = *v114;
                  v114[2] = v115;
                  v117 = sub_8D46C0(v116);
                  v118 = sub_73DBF0(92, v117, v114);
                  *(_BYTE *)(v118 + 25) |= 1u;
                  v143 = (__int64 *)v118;
                  v119 = sub_71AE50((__int64)v144);
                  v144 = v119;
                  v120 = sub_73E830(v103);
                  v121 = *v119;
                  v119[2] = v120;
                  v122 = sub_8D46C0(v121);
                  v123 = sub_73DBF0(92, v122, v119);
                  *(_BYTE *)(v123 + 25) |= 1u;
                  v144 = (_QWORD *)v123;
                  v124 = sub_692440((__int64)v143, v123);
                  v125 = sub_71B3F0(v124);
                  v48[9] = (__int64)v125;
                  v125[3] = v48;
                }
                else
                {
                  sub_71ADE0(&v143, (__int64 *)&v144);
                  v143 = (__int64 *)sub_73E470(v143, v42, v44);
                  v144 = (_QWORD *)sub_73E470(v144, v42, v45);
                  v46 = sub_692440((__int64)v143, (__int64)v144);
                  v47 = sub_71B3F0(v46);
                  v34[2] = (__int64)v47;
                  v48 = v47;
                  v47[3] = v140;
                }
                v34 = v48;
              }
              v41 = *(_QWORD *)(v41 + 16);
            }
            while ( v41 );
            v8 = v133;
          }
          v49 = sub_726B30(8);
          v34[2] = v49;
          v50 = v49;
          *(_QWORD *)(v49 + 24) = v140;
          v51 = sub_6EFF80();
          *(_QWORD *)(v50 + 48) = sub_73A7F0(v51);
          *(_QWORD *)(v140 + 72) = v147;
          dword_4F07508[0] = v137;
          LOWORD(dword_4F07508[1]) = v135;
          break;
        case 0x1F:
          v69 = sub_726B30(11);
          *(_QWORD *)(v142 + 80) = v69;
          sub_71ADE0(&v144, v146);
          v144 = (_QWORD *)sub_73DCD0(v144);
          v146[0] = sub_73DCD0(v146[0]);
          v70 = sub_692440((__int64)v144, v146[0]);
          v71 = (_QWORD *)sub_726B30(8);
          *v71 = *(_QWORD *)dword_4F07508;
          v72 = sub_6EFF80();
          v73 = sub_73DBF0(29, v72, v70);
          v71[3] = v69;
          v71[6] = v73;
          *(_QWORD *)(v69 + 72) = v71;
          break;
        case 0x22:
          v134 = dword_4F07508[0];
          v132 = dword_4F07508[1];
          v138 = *(_QWORD *)(j + 160);
          *(_QWORD *)dword_4F07508 = *(_QWORD *)(k + 64);
          v141 = sub_726B30(11);
          *(_QWORD *)(v142 + 80) = v141;
          v52 = v146;
          v147 = 0;
          if ( **(_QWORD **)(k + 168) )
          {
            v136 = v8;
            v53 = v146;
            v54 = **(__int64 ***)(k + 168);
            do
            {
              v55 = *((_BYTE *)v54 + 96);
              if ( (v55 & 1) != 0 && ((v55 & 2) == 0 || !(unsigned int)sub_8E35E0(v54, k)) )
              {
                sub_71ADE0(&v143, (__int64 *)&v144);
                v56 = sub_73E4A0(v143, v54);
                v143 = (__int64 *)sub_73DCD0(v56);
                v57 = sub_73E4A0(v144, v54);
                v144 = (_QWORD *)sub_73DCD0(v57);
                v60 = sub_71B470(v143, (__int64)v144, v138, v141, v58, v59);
                v53[2] = v60;
                if ( v60 )
                  v53 = *(__int64 **)(v60 + 16);
              }
              v54 = (__int64 *)*v54;
            }
            while ( v54 );
            v52 = v53;
            v8 = v136;
          }
          v139 = **(_QWORD **)(*(_QWORD *)k + 96LL);
          if ( v139 )
          {
            v131 = j;
            v129 = v8;
            do
            {
              if ( *(_BYTE *)(v139 + 80) == 8 )
              {
                v61 = *(_QWORD *)(v139 + 88);
                for ( kk = *(_QWORD *)(v61 + 120); *(_BYTE *)(kk + 140) == 12; kk = *(_QWORD *)(kk + 160) )
                  ;
                if ( (unsigned int)sub_8D3410(kk) )
                {
                  for ( mm = sub_8D40F0(kk); *(_BYTE *)(mm + 140) == 12; mm = *(_QWORD *)(mm + 160) )
                    ;
                  v126 = mm;
                  sub_71ADE0(&v143, (__int64 *)&v144);
                  v143 = (__int64 *)sub_73E470(v143, v61, v75);
                  v144 = (_QWORD *)sub_73E470(v144, v61, v76);
                  v127 = sub_72BA30(byte_4F06A51[0]);
                  v77 = sub_736020(v127, 0);
                  v78 = sub_731250(v77);
                  v79 = sub_73A830(0, byte_4F06A51[0]);
                  v80 = sub_73E690(v78, v79);
                  v52[2] = v80;
                  v81 = v80;
                  *(_QWORD *)(v80 + 24) = v141;
                  v82 = sub_731250(v77);
                  v83 = sub_73DBF0(37, v127, v82);
                  for ( nn = v83; *(_BYTE *)(kk + 140) == 12; kk = *(_QWORD *)(kk + 160) )
                    ;
                  *(_QWORD *)(v83 + 16) = sub_73A8E0(*(_QWORD *)(kk + 128) / *(_QWORD *)(v126 + 128), byte_4F06A51[0]);
                  v85 = sub_6EFF80();
                  v86 = sub_73DBF0(61, v85, nn);
                  v87 = sub_726B30(12);
                  *(_QWORD *)(v81 + 16) = v87;
                  v52 = (__int64 *)v87;
                  *(_QWORD *)(v87 + 48) = v86;
                  *(_QWORD *)(v87 + 24) = v141;
                  v88 = sub_71AE50((__int64)v143);
                  v143 = v88;
                  v89 = sub_73E830(v77);
                  v90 = *v88;
                  v88[2] = v89;
                  v91 = sub_8D46C0(v90);
                  v92 = sub_73DBF0(92, v91, v88);
                  *(_BYTE *)(v92 + 25) |= 1u;
                  v143 = (__int64 *)v92;
                  v93 = sub_71AE50((__int64)v144);
                  v144 = v93;
                  v94 = sub_73E830(v77);
                  v95 = *v93;
                  v93[2] = v94;
                  v96 = sub_8D46C0(v95);
                  v97 = sub_73DBF0(92, v96, v93);
                  *(_BYTE *)(v97 + 25) |= 1u;
                  v144 = (_QWORD *)v97;
                  v52[9] = sub_71B470(v143, v97, v138, (__int64)v52, v98, v99);
                }
                else
                {
                  sub_71ADE0(&v143, (__int64 *)&v144);
                  v143 = (__int64 *)sub_73E470(v143, v61, v63);
                  v144 = (_QWORD *)sub_73E470(v144, v61, v64);
                  v67 = sub_71B470(v143, (__int64)v144, v138, v141, v65, v66);
                  v52[2] = v67;
                  if ( v67 )
                    v52 = *(__int64 **)(v67 + 16);
                }
              }
              v139 = *(_QWORD *)(v139 + 16);
            }
            while ( v139 );
            j = v131;
            v8 = v129;
          }
          v68 = sub_726B30(8);
          v52[2] = v68;
          *(_QWORD *)(v68 + 24) = v141;
          sub_692550(j, v68);
          *(_QWORD *)(v141 + 72) = v147;
          dword_4F07508[0] = v134;
          LOWORD(dword_4F07508[1]) = v132;
          break;
        default:
          if ( (((_BYTE)v22 - 16) & 0xEE) != 0 )
            sub_721090(v21);
          v23 = sub_726B30(11);
          *(_QWORD *)(v142 + 80) = v23;
          sub_71ADE0(&v144, v146);
          v144 = (_QWORD *)sub_73DCD0(v144);
          v24 = sub_73DCD0(v146[0]);
          v25 = (__int64)v144;
          v26 = v24;
          v146[0] = v24;
          v27 = sub_691DE0((unsigned __int8)v22);
          v28 = sub_69C6F0(v27, v25, v26);
          v29 = (_QWORD *)sub_726B30(8);
          v29[6] = v28;
          v32 = *(_QWORD *)dword_4F07508;
          v29[3] = v23;
          *v29 = v32;
          *(_QWORD *)(v23 + 72) = v29;
          break;
      }
      return sub_71B580(v8, v142, v145, v30, v31);
    }
  }
  return result;
}
