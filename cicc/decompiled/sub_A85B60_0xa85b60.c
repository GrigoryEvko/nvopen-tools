// Function: sub_A85B60
// Address: 0xa85b60
//
__int64 __fastcall sub_A85B60(_QWORD *a1)
{
  __int64 v1; // r14
  __int64 v2; // r15
  unsigned int v3; // r14d
  __int64 v4; // rsi
  __int64 v5; // rbx
  unsigned __int8 v6; // al
  __int64 v7; // rax
  _BYTE *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int8 v27; // al
  _QWORD *v28; // rax
  unsigned __int8 v29; // al
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rsi
  char v34; // cl
  __int64 v35; // rdx
  unsigned __int8 v36; // al
  __int64 *v37; // rax
  __int64 v38; // rax
  __int64 v39; // r13
  __int64 v40; // rax
  bool v41; // r13
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // rax
  __int64 v47; // rdx
  unsigned __int8 v48; // al
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int8 v52; // al
  __int64 *v53; // rax
  __int64 v54; // rax
  __int64 v55; // r13
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // rax
  __int64 v62; // rdx
  unsigned __int8 v63; // al
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rdx
  unsigned __int8 v67; // al
  __int64 *v68; // rax
  __int64 v69; // rax
  __int64 v70; // r13
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  unsigned __int8 v76; // al
  __int64 v77; // rdx
  unsigned __int8 v78; // al
  __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v81; // rdx
  unsigned __int8 v82; // al
  __int64 v83; // rax
  _BYTE *v84; // rdi
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rdx
  unsigned __int8 v88; // al
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdi
  __int64 v92; // rax
  __int64 v93; // r13
  __int64 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rcx
  unsigned __int8 v97; // al
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // rax
  char v103; // cl
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // rax
  _QWORD *v107; // rcx
  _QWORD *v108; // r12
  _QWORD *v109; // r14
  __int64 v110; // rdx
  __int64 v111; // rdx
  _QWORD *v112; // rsi
  _BYTE *v113; // rsi
  _QWORD *v114; // rsi
  __int64 v115; // rdx
  unsigned __int8 v116; // al
  __int64 *v117; // rcx
  unsigned __int8 v118; // al
  __int64 v119; // rcx
  __int64 v120; // rdi
  __int64 v121; // rax
  __int64 v122; // rdi
  __int64 v123; // rax
  unsigned int v124; // [rsp+4h] [rbp-12Ch]
  _BYTE *v125; // [rsp+10h] [rbp-120h]
  unsigned int v126; // [rsp+20h] [rbp-110h]
  unsigned __int8 v127; // [rsp+27h] [rbp-109h]
  __int64 v128; // [rsp+30h] [rbp-100h]
  int v129; // [rsp+38h] [rbp-F8h]
  int v130; // [rsp+38h] [rbp-F8h]
  int v131; // [rsp+38h] [rbp-F8h]
  __int64 v132; // [rsp+48h] [rbp-E8h]
  __int64 v133; // [rsp+50h] [rbp-E0h]
  unsigned __int8 v134; // [rsp+58h] [rbp-D8h]
  unsigned __int8 v136; // [rsp+68h] [rbp-C8h]
  char v137; // [rsp+69h] [rbp-C7h]
  char v138; // [rsp+6Ah] [rbp-C6h]
  unsigned __int8 v139; // [rsp+6Bh] [rbp-C5h]
  int v140; // [rsp+6Ch] [rbp-C4h]
  _QWORD *v141; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v142; // [rsp+78h] [rbp-B8h]
  _QWORD v143[2]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v144; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v145; // [rsp+98h] [rbp-98h]
  _QWORD v146[2]; // [rsp+A0h] [rbp-90h] BYREF
  _QWORD *v147; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v148; // [rsp+B8h] [rbp-78h]
  _QWORD v149[14]; // [rsp+C0h] [rbp-70h] BYREF

  sub_A84F90(a1);
  v1 = a1[108];
  v134 = 0;
  if ( v1 )
  {
    v132 = sub_BCB2B0(*a1);
    v128 = sub_BCB2D0(*a1);
    v140 = sub_B91A00(v1);
    if ( v140 )
    {
      v139 = 0;
      v2 = v1;
      v138 = 0;
      v3 = 0;
      v137 = 0;
      while ( 1 )
      {
        v4 = v3;
        v5 = sub_B91A10(v2, v3);
        v6 = *(_BYTE *)(v5 - 16);
        if ( (v6 & 2) != 0 )
        {
          if ( *(_DWORD *)(v5 - 24) != 3 )
            goto LABEL_5;
          v133 = v5 - 16;
          v7 = *(_QWORD *)(v5 - 32);
        }
        else
        {
          if ( ((*(_WORD *)(v5 - 16) >> 6) & 0xF) != 3 )
            goto LABEL_5;
          v133 = v5 - 16;
          v7 = v5 - 8LL * ((v6 >> 2) & 0xF) - 16;
        }
        v8 = *(_BYTE **)(v7 + 8);
        if ( !v8 || *v8 )
          goto LABEL_5;
        v9 = sub_B91420(*(_QWORD *)(v7 + 8), v3);
        if ( v10 == 30
          && !(*(_QWORD *)v9 ^ 0x76697463656A624FLL | *(_QWORD *)(v9 + 8) ^ 0x67616D4920432D65LL)
          && *(_QWORD *)(v9 + 16) == 0x56206F666E492065LL
          && *(_DWORD *)(v9 + 24) == 1769173605 )
        {
          v103 = v137;
          if ( *(_WORD *)(v9 + 28) == 28271 )
            v103 = 1;
          v137 = v103;
        }
        v11 = sub_B91420(v8, v3);
        if ( v12 == 28
          && !(*(_QWORD *)v11 ^ 0x76697463656A624FLL | *(_QWORD *)(v11 + 8) ^ 0x73616C4320432D65LL)
          && *(_QWORD *)(v11 + 16) == 0x7265706F72502073LL )
        {
          v34 = v138;
          if ( *(_DWORD *)(v11 + 24) == 1936025972 )
            v34 = 1;
          v138 = v34;
          v13 = sub_B91420(v8, v3);
          if ( v35 != 9 )
          {
LABEL_15:
            v15 = sub_B91420(v8, v3);
            if ( v16 != 9 )
              goto LABEL_16;
            goto LABEL_51;
          }
        }
        else
        {
          v13 = sub_B91420(v8, v3);
          if ( v14 != 9 )
            goto LABEL_15;
        }
        if ( *(_QWORD *)v13 != 0x6576654C20434950LL || *(_BYTE *)(v13 + 8) != 108 )
          goto LABEL_15;
        v36 = *(_BYTE *)(v5 - 16);
        v37 = (v36 & 2) != 0 ? *(__int64 **)(v5 - 32) : (__int64 *)(v5 - 8LL * ((v36 >> 2) & 0xF) - 16);
        v38 = *v37;
        if ( !v38 )
          goto LABEL_15;
        if ( *(_BYTE *)v38 != 1 )
          goto LABEL_15;
        v39 = *(_QWORD *)(v38 + 136);
        if ( *(_BYTE *)v39 != 17 )
          goto LABEL_15;
        if ( *(_DWORD *)(v39 + 32) > 0x40u )
        {
          v129 = *(_DWORD *)(v39 + 32);
          if ( v129 - (unsigned int)sub_C444A0(v39 + 24) > 0x40 )
            goto LABEL_15;
          v40 = **(_QWORD **)(v39 + 24);
        }
        else
        {
          v40 = *(_QWORD *)(v39 + 24);
        }
        v41 = v40 == 7 || v40 == 1;
        if ( !v41 )
          goto LABEL_15;
        v42 = sub_BCB2D0(*a1);
        v43 = sub_ACD640(v42, 8, 0);
        v147 = (_QWORD *)sub_B98A20(v43, 8, v44, v45);
        v46 = sub_B91420(v8, 8);
        v148 = sub_B9B140(*a1, v46, v47);
        v48 = *(_BYTE *)(v5 - 16);
        if ( (v48 & 2) != 0 )
          v49 = *(_QWORD *)(v5 - 32);
        else
          v49 = v5 - 8LL * ((v48 >> 2) & 0xF) - 16;
        v149[0] = *(_QWORD *)(v49 + 16);
        v50 = sub_B9C770(*a1, &v147, 3, 0, 1);
        v4 = v3;
        sub_B970B0(v2, v3, v50);
        v134 = v41;
        v15 = sub_B91420(v8, v3);
        if ( v51 != 9 )
        {
LABEL_16:
          v17 = sub_B91420(v8, v4);
          if ( v18 == 25 )
            goto LABEL_64;
          goto LABEL_17;
        }
LABEL_51:
        if ( *(_QWORD *)v15 != 0x6576654C20454950LL || *(_BYTE *)(v15 + 8) != 108 )
          goto LABEL_16;
        v52 = *(_BYTE *)(v5 - 16);
        v53 = (v52 & 2) != 0 ? *(__int64 **)(v5 - 32) : (__int64 *)(v5 - 8LL * ((v52 >> 2) & 0xF) - 16);
        v54 = *v53;
        if ( !v54 )
          goto LABEL_16;
        if ( *(_BYTE *)v54 != 1 )
          goto LABEL_16;
        v55 = *(_QWORD *)(v54 + 136);
        if ( *(_BYTE *)v55 != 17 )
          goto LABEL_16;
        if ( *(_DWORD *)(v55 + 32) > 0x40u )
        {
          v130 = *(_DWORD *)(v55 + 32);
          if ( v130 - (unsigned int)sub_C444A0(v55 + 24) > 0x40 )
            goto LABEL_16;
          v56 = **(_QWORD **)(v55 + 24);
        }
        else
        {
          v56 = *(_QWORD *)(v55 + 24);
        }
        if ( v56 != 1 )
          goto LABEL_16;
        v57 = sub_BCB2D0(*a1);
        v58 = sub_ACD640(v57, 7, 0);
        v147 = (_QWORD *)sub_B98A20(v58, 7, v59, v60);
        v61 = sub_B91420(v8, 7);
        v148 = sub_B9B140(*a1, v61, v62);
        v63 = *(_BYTE *)(v5 - 16);
        if ( (v63 & 2) != 0 )
          v64 = *(_QWORD *)(v5 - 32);
        else
          v64 = v5 - 8LL * ((v63 >> 2) & 0xF) - 16;
        v149[0] = *(_QWORD *)(v64 + 16);
        v65 = sub_B9C770(*a1, &v147, 3, 0, 1);
        v4 = v3;
        sub_B970B0(v2, v3, v65);
        v134 = 1;
        v17 = sub_B91420(v8, v3);
        if ( v66 == 25 )
        {
LABEL_64:
          if ( !(*(_QWORD *)v17 ^ 0x742D68636E617262LL | *(_QWORD *)(v17 + 8) ^ 0x6E652D7465677261LL)
            && *(_QWORD *)(v17 + 16) == 0x6E656D6563726F66LL
            && *(_BYTE *)(v17 + 24) == 116 )
          {
            goto LABEL_67;
          }
        }
LABEL_17:
        v19 = sub_B91420(v8, v4);
        if ( v20 <= 0x12
          || *(_QWORD *)v19 ^ 0x7465722D6E676973LL | *(_QWORD *)(v19 + 8) ^ 0x726464612D6E7275LL
          || *(_WORD *)(v19 + 16) != 29541
          || *(_BYTE *)(v19 + 18) != 115 )
        {
          goto LABEL_18;
        }
LABEL_67:
        v67 = *(_BYTE *)(v5 - 16);
        if ( (v67 & 2) != 0 )
          v68 = *(__int64 **)(v5 - 32);
        else
          v68 = (__int64 *)(v5 - 8LL * ((v67 >> 2) & 0xF) - 16);
        v69 = *v68;
        if ( !v69 || *(_BYTE *)v69 != 1 || (v70 = *(_QWORD *)(v69 + 136), *(_BYTE *)v70 != 17) )
        {
LABEL_18:
          v21 = sub_B91420(v8, v4);
          if ( v22 != 30 )
            goto LABEL_19;
          goto LABEL_80;
        }
        if ( *(_DWORD *)(v70 + 32) > 0x40u )
        {
          v131 = *(_DWORD *)(v70 + 32);
          if ( v131 - (unsigned int)sub_C444A0(v70 + 24) > 0x40 )
            goto LABEL_18;
          v71 = **(_QWORD **)(v70 + 24);
        }
        else
        {
          v71 = *(_QWORD *)(v70 + 24);
        }
        if ( v71 != 1 )
          goto LABEL_18;
        v72 = sub_BCB2D0(*a1);
        v73 = sub_AD64C0(v72, 8, 0);
        v147 = (_QWORD *)sub_B98A20(v73, 8, v74, v75);
        v76 = *(_BYTE *)(v5 - 16);
        if ( (v76 & 2) != 0 )
          v77 = *(_QWORD *)(v5 - 32);
        else
          v77 = v133 - 8LL * ((v76 >> 2) & 0xF);
        v148 = *(_QWORD *)(v77 + 8);
        v78 = *(_BYTE *)(v5 - 16);
        if ( (v78 & 2) != 0 )
          v79 = *(_QWORD *)(v5 - 32);
        else
          v79 = v133 - 8LL * ((v78 >> 2) & 0xF);
        v149[0] = *(_QWORD *)(v79 + 16);
        v80 = sub_B9C770(*a1, &v147, 3, 0, 1);
        v4 = v3;
        sub_B970B0(v2, v3, v80);
        v134 = 1;
        v21 = sub_B91420(v8, v3);
        if ( v81 != 30 )
          goto LABEL_19;
LABEL_80:
        if ( *(_QWORD *)v21 ^ 0x76697463656A624FLL | *(_QWORD *)(v21 + 8) ^ 0x67616D4920432D65LL
          || *(_QWORD *)(v21 + 16) != 0x53206F666E492065LL
          || *(_DWORD *)(v21 + 24) != 1769235301
          || *(_WORD *)(v21 + 28) != 28271 )
        {
          goto LABEL_19;
        }
        v82 = *(_BYTE *)(v5 - 16);
        v83 = (v82 & 2) != 0 ? *(_QWORD *)(v5 - 32) : v5 - 8LL * ((v82 >> 2) & 0xF) - 16;
        v84 = *(_BYTE **)(v83 + 16);
        if ( !v84 || *v84 )
          goto LABEL_19;
        v147 = v149;
        v148 = 0x400000000LL;
        v85 = sub_B91420(v84, v4);
        v4 = (__int64)&v147;
        v144 = (_QWORD *)v85;
        v145 = v86;
        sub_C937F0(&v144, &v147, " ", 1, 0xFFFFFFFFLL, 1);
        if ( (unsigned int)v148 != 1 )
        {
          v106 = 2LL * (unsigned int)v148;
          v142 = 0;
          v141 = v143;
          LOBYTE(v143[0]) = 0;
          if ( &v147[v106] == v147 )
          {
            v114 = v143;
            v115 = 0;
          }
          else
          {
            v107 = v146;
            v125 = v8;
            v108 = &v147[v106];
            v124 = v3;
            v109 = v147;
            do
            {
              v113 = (_BYTE *)*v109;
              if ( *v109 )
              {
                v110 = v109[1];
                v144 = v146;
                sub_A7BD10((__int64 *)&v144, v113, (__int64)&v113[v110]);
                v111 = v145;
                v112 = v144;
              }
              else
              {
                v112 = v146;
                v144 = v146;
                v111 = 0;
                v145 = 0;
                LOBYTE(v146[0]) = 0;
              }
              sub_2241490(&v141, v112, v111, v107);
              if ( v144 != v146 )
                j_j___libc_free_0(v144, v146[0] + 1LL);
              v109 += 2;
            }
            while ( v108 != v109 );
            v8 = v125;
            v3 = v124;
            v114 = v141;
            v115 = v142;
          }
          v116 = *(_BYTE *)(v5 - 16);
          if ( (v116 & 2) != 0 )
            v117 = *(__int64 **)(v5 - 32);
          else
            v117 = (__int64 *)(v133 - 8LL * ((v116 >> 2) & 0xF));
          v144 = (_QWORD *)*v117;
          v118 = *(_BYTE *)(v5 - 16);
          if ( (v118 & 2) != 0 )
            v119 = *(_QWORD *)(v5 - 32);
          else
            v119 = v133 - 8LL * ((v118 >> 2) & 0xF);
          v120 = *a1;
          v145 = *(_QWORD *)(v119 + 8);
          v121 = sub_B9B140(v120, v114, v115);
          v122 = *a1;
          v146[0] = v121;
          v123 = sub_B9C770(v122, &v144, 3, 0, 1);
          v4 = v3;
          sub_B970B0(v2, v3, v123);
          if ( v141 != v143 )
          {
            v4 = v143[0] + 1LL;
            j_j___libc_free_0(v141, v143[0] + 1LL);
          }
          v134 = 1;
        }
        if ( v147 == v149 )
        {
LABEL_19:
          v23 = sub_B91420(v8, v4);
          if ( v24 != 30 )
            goto LABEL_20;
          goto LABEL_91;
        }
        _libc_free(v147, v4);
        v23 = sub_B91420(v8, v4);
        if ( v87 != 30 )
          goto LABEL_20;
LABEL_91:
        if ( *(_QWORD *)v23 ^ 0x76697463656A624FLL | *(_QWORD *)(v23 + 8) ^ 0x6272614720432D65LL
          || *(_QWORD *)(v23 + 16) != 0x6C6C6F4320656761LL
          || *(_DWORD *)(v23 + 24) != 1769235301
          || *(_WORD *)(v23 + 28) != 28271 )
        {
          goto LABEL_20;
        }
        v88 = *(_BYTE *)(v5 - 16);
        v89 = (v88 & 2) != 0 ? *(_QWORD *)(v5 - 32) : v5 - 8LL * ((v88 >> 2) & 0xF) - 16;
        v90 = *(_QWORD *)(v89 + 16);
        if ( *(_BYTE *)v90 != 1 )
          goto LABEL_20;
        v91 = *(_QWORD *)(v90 + 136);
        if ( v132 == *(_QWORD *)(v91 + 8) )
        {
LABEL_5:
          if ( v140 == ++v3 )
            goto LABEL_29;
        }
        else
        {
          v92 = sub_AD8340(v91);
          if ( *(_DWORD *)(v92 + 8) > 0x40u )
            v92 = *(_QWORD *)v92;
          v93 = *(_QWORD *)v92;
          if ( (*(_QWORD *)v92 & 0xFFFFFF00) != 0 )
          {
            v139 = 1;
            v126 = BYTE1(v93);
            v136 = BYTE3(*(_QWORD *)v92);
            v127 = BYTE2(*(_QWORD *)v92);
          }
          v94 = sub_ACD640(v128, 1, 0);
          v147 = (_QWORD *)sub_B98A20(v94, 1, v95, v96);
          v97 = *(_BYTE *)(v5 - 16);
          if ( (v97 & 2) != 0 )
            v98 = *(_QWORD *)(v5 - 32);
          else
            v98 = v5 - 8LL * ((v97 >> 2) & 0xF) - 16;
          v148 = *(_QWORD *)(v98 + 8);
          v99 = sub_ACD640(v132, (unsigned __int8)v93, 0);
          v149[0] = sub_B98A20(v99, (unsigned __int8)v93, v100, v101);
          v102 = sub_B9C770(*a1, &v147, 3, 0, 1);
          v4 = v3;
          sub_B970B0(v2, v3, v102);
          v134 = 1;
LABEL_20:
          v25 = sub_B91420(v8, v4);
          if ( v26 != 26
            || *(_QWORD *)v25 ^ 0x635F757067646D61LL | *(_QWORD *)(v25 + 8) ^ 0x656A626F5F65646FLL
            || *(_QWORD *)(v25 + 16) != 0x69737265765F7463LL
            || *(_WORD *)(v25 + 24) != 28271 )
          {
            goto LABEL_5;
          }
          v27 = *(_BYTE *)(v5 - 16);
          if ( (v27 & 2) != 0 )
            v28 = *(_QWORD **)(v5 - 32);
          else
            v28 = (_QWORD *)(v5 - 8LL * ((v27 >> 2) & 0xF) - 16);
          v147 = (_QWORD *)*v28;
          v148 = sub_B9B140(*a1, "amdhsa_code_object_version", 26);
          v29 = *(_BYTE *)(v5 - 16);
          if ( (v29 & 2) != 0 )
            v30 = *(_QWORD *)(v5 - 32);
          else
            v30 = v5 - 8LL * ((v29 >> 2) & 0xF) - 16;
          v149[0] = *(_QWORD *)(v30 + 16);
          v31 = sub_B9C770(*a1, &v147, 3, 0, 1);
          v32 = v3++;
          sub_B970B0(v2, v32, v31);
          v134 = 1;
          if ( v140 == v3 )
          {
LABEL_29:
            if ( (((unsigned __int8)v138 ^ 1) & (unsigned __int8)v137) != 0 )
            {
              sub_BA93D0(a1, 4, "Objective-C Class Properties", 28, 0);
              v134 = (v138 ^ 1) & v137;
              if ( !v139 )
                return v134;
            }
            else if ( !v139 )
            {
              return v134;
            }
            sub_BA93D0(a1, 1, "Swift ABI Version", 17, v126);
            v104 = sub_ACD640(v132, v136, 0);
            sub_BA9390(a1, 1, "Swift Major Version", 19, v104);
            v105 = sub_ACD640(v132, v127, 0);
            sub_BA9390(a1, 1, "Swift Minor Version", 19, v105);
            return v139;
          }
        }
      }
    }
  }
  return v134;
}
