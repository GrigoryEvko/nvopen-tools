// Function: sub_24941A0
// Address: 0x24941a0
//
__int64 __fastcall sub_24941A0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  unsigned __int8 v8; // al
  unsigned __int8 *v9; // r13
  __int64 v10; // rbx
  _QWORD *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned __int8 *v15; // r15
  __int64 (__fastcall *v16)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v17; // r14
  _QWORD *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned __int8 *v22; // r15
  __int64 (__fastcall *v23)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v24; // r10
  __int64 v25; // rax
  unsigned __int8 *v26; // r14
  __int64 v27; // rdi
  __int64 (__fastcall *v28)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v33; // rbx
  _QWORD *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 **v37; // rdi
  unsigned __int64 v38; // rax
  __int64 v39; // rdi
  __int64 (__fastcall *v40)(__int64, _BYTE *, __int64, __int64); // rax
  __int64 v41; // r14
  __int64 v42; // r15
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdi
  unsigned __int8 *v46; // r15
  __int64 (__fastcall *v47)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v48; // r10
  __int64 v49; // rax
  unsigned __int8 *v50; // r14
  __int64 v51; // rdi
  __int64 (__fastcall *v52)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v53; // rax
  _QWORD *v54; // rax
  _QWORD *v55; // r10
  __int64 v56; // r14
  __int64 v57; // rbx
  __int64 v58; // r15
  __int64 v59; // rdx
  unsigned int v60; // esi
  _QWORD *v61; // rax
  __int64 v62; // rbx
  __int64 v63; // r15
  __int64 v64; // rdx
  unsigned int v65; // esi
  __int64 v66; // r15
  __int64 v67; // r14
  __int64 v68; // rdx
  unsigned int v69; // esi
  _QWORD *v70; // rax
  _QWORD *v71; // r10
  __int64 v72; // rcx
  __int64 v73; // rbx
  __int64 v74; // r12
  __int64 v75; // r15
  __int64 v76; // rdx
  unsigned int v77; // esi
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rbx
  __int64 v82; // r15
  __int64 v83; // rdx
  unsigned int v84; // esi
  __int64 v85; // r15
  __int64 v86; // r14
  __int64 v87; // rdx
  unsigned int v88; // esi
  __int64 v89; // r15
  _QWORD *v90; // rdi
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rdi
  unsigned __int8 *v94; // rbx
  __int64 (__fastcall *v95)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v96; // r14
  _QWORD *v97; // rdi
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rdi
  unsigned __int8 *v101; // rbx
  __int64 (__fastcall *v102)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v103; // r10
  __int64 v104; // rax
  unsigned __int8 *v105; // r14
  __int64 v106; // rdi
  __int64 (__fastcall *v107)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v108; // rax
  _QWORD *v109; // rax
  _QWORD *v110; // r10
  __int64 v111; // rcx
  __int64 v112; // r14
  __int64 v113; // r12
  __int64 v114; // rbx
  __int64 v115; // rdx
  unsigned int v116; // esi
  _QWORD *v117; // rax
  __int64 v118; // rcx
  __int64 v119; // r12
  __int64 v120; // rbx
  __int64 v121; // rdx
  unsigned int v122; // esi
  __int64 v123; // r14
  __int64 v124; // rbx
  __int64 v125; // rdx
  unsigned int v126; // esi
  __int64 v127; // [rsp+0h] [rbp-110h]
  __int64 v128; // [rsp+0h] [rbp-110h]
  __int64 v129; // [rsp+0h] [rbp-110h]
  _QWORD *v130; // [rsp+8h] [rbp-108h]
  __int64 v131; // [rsp+8h] [rbp-108h]
  __int64 v132; // [rsp+8h] [rbp-108h]
  __int64 v133; // [rsp+8h] [rbp-108h]
  _QWORD *v134; // [rsp+8h] [rbp-108h]
  __int64 v135; // [rsp+8h] [rbp-108h]
  __int64 v136; // [rsp+8h] [rbp-108h]
  __int64 v137; // [rsp+8h] [rbp-108h]
  _QWORD *v138; // [rsp+8h] [rbp-108h]
  __int64 v139; // [rsp+8h] [rbp-108h]
  __int64 v140; // [rsp+8h] [rbp-108h]
  __int64 v141; // [rsp+8h] [rbp-108h]
  __int64 v142; // [rsp+10h] [rbp-100h]
  __int64 v143; // [rsp+18h] [rbp-F8h]
  _QWORD *v144; // [rsp+18h] [rbp-F8h]
  __int64 v145; // [rsp+18h] [rbp-F8h]
  __int64 v146; // [rsp+20h] [rbp-F0h]
  int v152; // [rsp+58h] [rbp-B8h]
  _QWORD v153[4]; // [rsp+60h] [rbp-B0h] BYREF
  _BYTE v154[32]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v155; // [rsp+A0h] [rbp-70h]
  _BYTE v156[32]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v157; // [rsp+D0h] [rbp-40h]

  if ( *(_BYTE *)a2 <= 0x15u )
    goto LABEL_31;
  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(_BYTE *)(v7 + 8);
  switch ( v8 )
  {
    case 2u:
      v33 = 0;
      goto LABEL_33;
    case 3u:
      v33 = 1;
LABEL_33:
      v155 = 257;
      v34 = *(_QWORD **)(a1 + 8);
      v153[0] = a2;
      v153[1] = a3;
      v35 = sub_BCB2D0(v34);
      v36 = sub_ACD640(v35, (int)a6, 0);
      v37 = *(__int64 ***)(a1 + 48);
      v153[2] = v36;
      if ( (_DWORD)a6 == 2 )
      {
        v38 = sub_AD64C0((__int64)v37, a6 >> 32, 0);
        goto LABEL_37;
      }
      if ( (unsigned int)a6 <= 2 )
      {
        if ( (_DWORD)a6 == 1 )
          goto LABEL_76;
      }
      else
      {
        if ( (unsigned int)a6 <= 4 )
        {
          v157 = 257;
          v38 = sub_24932B0((__int64 *)a4, 0x2Fu, a5, v37, (__int64)v156, 0, v152, 0);
LABEL_37:
          v153[3] = v38;
          return sub_921880(
                   (unsigned int **)a4,
                   *(_QWORD *)(16 * v33 + a1 + 152),
                   *(_QWORD *)(16 * v33 + a1 + 160),
                   (int)v153,
                   4,
                   (__int64)v154,
                   0);
        }
        if ( (_DWORD)a6 == 5 )
        {
LABEL_76:
          v38 = sub_AD64C0((__int64)v37, 0, 0);
          goto LABEL_37;
        }
      }
LABEL_154:
      BUG();
    case 4u:
      v33 = 2;
      goto LABEL_33;
  }
  if ( (unsigned int)v8 - 17 <= 1 )
  {
    v9 = 0;
    if ( *(int *)(v7 + 32) <= 0 )
      return (__int64)v9;
    v143 = *(unsigned int *)(v7 + 32);
    v10 = 0;
    while ( 1 )
    {
      v11 = *(_QWORD **)(a4 + 72);
      v155 = 257;
      v12 = sub_BCB2E0(v11);
      v13 = sub_ACD640(v12, v10, 0);
      v14 = *(_QWORD *)(a4 + 80);
      v15 = (unsigned __int8 *)v13;
      v16 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v14 + 96LL);
      if ( v16 != sub_948070 )
        break;
      if ( *(_BYTE *)a2 <= 0x15u && *v15 <= 0x15u )
      {
        v17 = sub_AD5840(a2, v15, 0);
        goto LABEL_12;
      }
LABEL_69:
      v157 = 257;
      v61 = sub_BD2C40(72, 2u);
      v17 = (__int64)v61;
      if ( v61 )
        sub_B4DE80((__int64)v61, a2, (__int64)v15, (__int64)v156, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
        *(_QWORD *)(a4 + 88),
        v17,
        v154,
        *(_QWORD *)(a4 + 56),
        *(_QWORD *)(a4 + 64));
      if ( *(_QWORD *)a4 != *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8) )
      {
        v133 = v10;
        v62 = *(_QWORD *)a4;
        v63 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
        do
        {
          v64 = *(_QWORD *)(v62 + 8);
          v65 = *(_DWORD *)v62;
          v62 += 16;
          sub_B99FD0(v17, v65, v64);
        }
        while ( v63 != v62 );
        v10 = v133;
      }
LABEL_13:
      v18 = *(_QWORD **)(a4 + 72);
      v155 = 257;
      v19 = sub_BCB2E0(v18);
      v20 = sub_ACD640(v19, v10, 0);
      v21 = *(_QWORD *)(a4 + 80);
      v22 = (unsigned __int8 *)v20;
      v23 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v21 + 96LL);
      if ( v23 != sub_948070 )
      {
        v24 = v23(v21, a3, v22);
LABEL_17:
        if ( v24 )
          goto LABEL_18;
        goto LABEL_63;
      }
      if ( *a3 <= 0x15u && *v22 <= 0x15u )
      {
        v24 = sub_AD5840((__int64)a3, v22, 0);
        goto LABEL_17;
      }
LABEL_63:
      v157 = 257;
      v54 = sub_BD2C40(72, 2u);
      v55 = v54;
      if ( v54 )
      {
        v130 = v54;
        sub_B4DE80((__int64)v54, (__int64)a3, (__int64)v22, (__int64)v156, 0, 0);
        v55 = v130;
      }
      v131 = (__int64)v55;
      (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
        *(_QWORD *)(a4 + 88),
        v55,
        v154,
        *(_QWORD *)(a4 + 56),
        *(_QWORD *)(a4 + 64));
      v24 = v131;
      if ( *(_QWORD *)a4 != *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8) )
      {
        v132 = v17;
        v56 = v24;
        v127 = v10;
        v57 = *(_QWORD *)a4;
        v58 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
        do
        {
          v59 = *(_QWORD *)(v57 + 8);
          v60 = *(_DWORD *)v57;
          v57 += 16;
          sub_B99FD0(v56, v60, v59);
        }
        while ( v58 != v57 );
        v24 = v56;
        v10 = v127;
        v17 = v132;
      }
LABEL_18:
      v25 = sub_24941A0(a1, v17, v24, a4, a5, a6);
      v26 = (unsigned __int8 *)v25;
      if ( v9 )
      {
        v27 = *(_QWORD *)(a4 + 80);
        v155 = 257;
        v28 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v27 + 16LL);
        if ( v28 == sub_9202E0 )
        {
          if ( *v9 > 0x15u || *v26 > 0x15u )
          {
LABEL_79:
            v157 = 257;
            v9 = (unsigned __int8 *)sub_B504D0(29, (__int64)v9, (__int64)v26, (__int64)v156, 0, 0);
            (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
              *(_QWORD *)(a4 + 88),
              v9,
              v154,
              *(_QWORD *)(a4 + 56),
              *(_QWORD *)(a4 + 64));
            v66 = *(_QWORD *)a4;
            v67 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
            if ( *(_QWORD *)a4 != v67 )
            {
              do
              {
                v68 = *(_QWORD *)(v66 + 8);
                v69 = *(_DWORD *)v66;
                v66 += 16;
                sub_B99FD0((__int64)v9, v69, v68);
              }
              while ( v67 != v66 );
            }
            goto LABEL_26;
          }
          if ( (unsigned __int8)sub_AC47B0(29) )
            v29 = sub_AD5570(29, (__int64)v9, v26, 0, 0);
          else
            v29 = sub_AABE40(0x1Du, v9, v26);
        }
        else
        {
          v29 = v28(v27, 29u, v9, v26);
        }
        if ( !v29 )
          goto LABEL_79;
        v9 = (unsigned __int8 *)v29;
      }
      else
      {
        v9 = (unsigned __int8 *)v25;
      }
LABEL_26:
      if ( v143 == ++v10 )
        return (__int64)v9;
    }
    v17 = v16(v14, (_BYTE *)a2, v15);
LABEL_12:
    if ( v17 )
      goto LABEL_13;
    goto LABEL_69;
  }
  if ( v8 == 16 )
  {
    v9 = 0;
    v145 = *(_QWORD *)(v7 + 32);
    if ( !v145 )
      return (__int64)v9;
    v89 = 0;
    while ( 1 )
    {
      v90 = *(_QWORD **)(a4 + 72);
      v155 = 257;
      v91 = sub_BCB2E0(v90);
      v92 = sub_ACD640(v91, v89, 0);
      v93 = *(_QWORD *)(a4 + 80);
      v94 = (unsigned __int8 *)v92;
      v95 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v93 + 96LL);
      if ( v95 != sub_948070 )
        break;
      if ( *(_BYTE *)a2 <= 0x15u && *v94 <= 0x15u )
      {
        v96 = sub_AD5840(a2, v94, 0);
        goto LABEL_118;
      }
LABEL_141:
      v157 = 257;
      v117 = sub_BD2C40(72, 2u);
      v96 = (__int64)v117;
      if ( v117 )
        sub_B4DE80((__int64)v117, a2, (__int64)v94, (__int64)v156, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
        *(_QWORD *)(a4 + 88),
        v96,
        v154,
        *(_QWORD *)(a4 + 56),
        *(_QWORD *)(a4 + 64));
      v118 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
      if ( *(_QWORD *)a4 != v118 )
      {
        v141 = a4;
        v119 = *(_QWORD *)a4;
        v120 = v118;
        do
        {
          v121 = *(_QWORD *)(v119 + 8);
          v122 = *(_DWORD *)v119;
          v119 += 16;
          sub_B99FD0(v96, v122, v121);
        }
        while ( v120 != v119 );
        a4 = v141;
      }
LABEL_119:
      v97 = *(_QWORD **)(a4 + 72);
      v155 = 257;
      v98 = sub_BCB2E0(v97);
      v99 = sub_ACD640(v98, v89, 0);
      v100 = *(_QWORD *)(a4 + 80);
      v101 = (unsigned __int8 *)v99;
      v102 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v100 + 96LL);
      if ( v102 != sub_948070 )
      {
        v103 = v102(v100, a3, v101);
LABEL_123:
        if ( v103 )
          goto LABEL_124;
        goto LABEL_135;
      }
      if ( *a3 <= 0x15u && *v101 <= 0x15u )
      {
        v103 = sub_AD5840((__int64)a3, v101, 0);
        goto LABEL_123;
      }
LABEL_135:
      v157 = 257;
      v109 = sub_BD2C40(72, 2u);
      v110 = v109;
      if ( v109 )
      {
        v138 = v109;
        sub_B4DE80((__int64)v109, (__int64)a3, (__int64)v101, (__int64)v156, 0, 0);
        v110 = v138;
      }
      v139 = (__int64)v110;
      (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
        *(_QWORD *)(a4 + 88),
        v110,
        v154,
        *(_QWORD *)(a4 + 56),
        *(_QWORD *)(a4 + 64));
      v103 = v139;
      v111 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
      if ( *(_QWORD *)a4 != v111 )
      {
        v140 = v96;
        v112 = v103;
        v129 = a4;
        v113 = *(_QWORD *)a4;
        v114 = v111;
        do
        {
          v115 = *(_QWORD *)(v113 + 8);
          v116 = *(_DWORD *)v113;
          v113 += 16;
          sub_B99FD0(v112, v116, v115);
        }
        while ( v114 != v113 );
        v103 = v112;
        a4 = v129;
        v96 = v140;
      }
LABEL_124:
      v104 = sub_24941A0(a1, v96, v103, a4, a5, a6);
      v105 = (unsigned __int8 *)v104;
      if ( v9 )
      {
        v106 = *(_QWORD *)(a4 + 80);
        v155 = 257;
        v107 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v106 + 16LL);
        if ( v107 == sub_9202E0 )
        {
          if ( *v9 > 0x15u || *v105 > 0x15u )
          {
LABEL_147:
            v157 = 257;
            v9 = (unsigned __int8 *)sub_B504D0(29, (__int64)v9, (__int64)v105, (__int64)v156, 0, 0);
            (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
              *(_QWORD *)(a4 + 88),
              v9,
              v154,
              *(_QWORD *)(a4 + 56),
              *(_QWORD *)(a4 + 64));
            v123 = *(_QWORD *)a4;
            v124 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
            if ( *(_QWORD *)a4 != v124 )
            {
              do
              {
                v125 = *(_QWORD *)(v123 + 8);
                v126 = *(_DWORD *)v123;
                v123 += 16;
                sub_B99FD0((__int64)v9, v126, v125);
              }
              while ( v124 != v123 );
            }
            goto LABEL_132;
          }
          if ( (unsigned __int8)sub_AC47B0(29) )
            v108 = sub_AD5570(29, (__int64)v9, v105, 0, 0);
          else
            v108 = sub_AABE40(0x1Du, v9, v105);
        }
        else
        {
          v108 = v107(v106, 29u, v9, v105);
        }
        if ( !v108 )
          goto LABEL_147;
        v9 = (unsigned __int8 *)v108;
      }
      else
      {
        v9 = (unsigned __int8 *)v104;
      }
LABEL_132:
      if ( v145 == ++v89 )
        return (__int64)v9;
    }
    v96 = v95(v93, (_BYTE *)a2, v94);
LABEL_118:
    if ( v96 )
      goto LABEL_119;
    goto LABEL_141;
  }
  if ( v8 != 15 )
    goto LABEL_154;
  v146 = *(unsigned int *)(v7 + 12);
  if ( !*(_DWORD *)(v7 + 12) )
  {
LABEL_31:
    v31 = sub_BCB2D0(*(_QWORD **)(a4 + 72));
    return sub_ACD640(v31, 0, 0);
  }
  v142 = *(_QWORD *)(a2 + 8);
  v30 = 0;
  v9 = 0;
  v144 = (_QWORD *)(a1 + 16);
  do
  {
    LODWORD(v153[0]) = v30;
    if ( !sub_2491640(v144, *(_QWORD *)(*(_QWORD *)(v142 + 16) + 8LL * (unsigned int)v30)) )
      goto LABEL_29;
    v39 = *(_QWORD *)(a4 + 80);
    v155 = 257;
    v40 = *(__int64 (__fastcall **)(__int64, _BYTE *, __int64, __int64))(*(_QWORD *)v39 + 80LL);
    if ( v40 != sub_92FAE0 )
    {
      v41 = v40(v39, (_BYTE *)a2, (__int64)v153, 1);
LABEL_47:
      if ( v41 )
        goto LABEL_48;
      goto LABEL_91;
    }
    if ( *(_BYTE *)a2 <= 0x15u )
    {
      v41 = sub_AAADB0(a2, (unsigned int *)v153, 1);
      goto LABEL_47;
    }
LABEL_91:
    v157 = 257;
    v41 = (__int64)sub_BD2C40(104, unk_3F10A14);
    if ( v41 )
    {
      v78 = sub_B501B0(*(_QWORD *)(a2 + 8), (unsigned int *)v153, 1);
      sub_B44260(v41, v78, 64, 1u, 0, 0);
      if ( *(_QWORD *)(v41 - 32) )
      {
        v79 = *(_QWORD *)(v41 - 24);
        **(_QWORD **)(v41 - 16) = v79;
        if ( v79 )
          *(_QWORD *)(v79 + 16) = *(_QWORD *)(v41 - 16);
      }
      *(_QWORD *)(v41 - 32) = a2;
      v80 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(v41 - 24) = v80;
      if ( v80 )
        *(_QWORD *)(v80 + 16) = v41 - 24;
      *(_QWORD *)(v41 - 16) = a2 + 16;
      *(_QWORD *)(a2 + 16) = v41 - 32;
      *(_QWORD *)(v41 + 72) = v41 + 88;
      *(_QWORD *)(v41 + 80) = 0x400000000LL;
      sub_B50030(v41, v153, 1, (__int64)v156);
    }
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
      *(_QWORD *)(a4 + 88),
      v41,
      v154,
      *(_QWORD *)(a4 + 56),
      *(_QWORD *)(a4 + 64));
    if ( *(_QWORD *)a4 != *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8) )
    {
      v137 = v30;
      v81 = *(_QWORD *)a4;
      v82 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
      do
      {
        v83 = *(_QWORD *)(v81 + 8);
        v84 = *(_DWORD *)v81;
        v81 += 16;
        sub_B99FD0(v41, v84, v83);
      }
      while ( v82 != v81 );
      v30 = v137;
    }
LABEL_48:
    v42 = LODWORD(v153[0]);
    v155 = 257;
    v43 = sub_BCB2E0(*(_QWORD **)(a4 + 72));
    v44 = sub_ACD640(v43, v42, 0);
    v45 = *(_QWORD *)(a4 + 80);
    v46 = (unsigned __int8 *)v44;
    v47 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v45 + 96LL);
    if ( v47 != sub_948070 )
    {
      v48 = v47(v45, a3, v46);
LABEL_52:
      if ( v48 )
        goto LABEL_53;
      goto LABEL_84;
    }
    if ( *a3 <= 0x15u && *v46 <= 0x15u )
    {
      v48 = sub_AD5840((__int64)a3, v46, 0);
      goto LABEL_52;
    }
LABEL_84:
    v157 = 257;
    v70 = sub_BD2C40(72, 2u);
    v71 = v70;
    if ( v70 )
    {
      v134 = v70;
      sub_B4DE80((__int64)v70, (__int64)a3, (__int64)v46, (__int64)v156, 0, 0);
      v71 = v134;
    }
    v135 = (__int64)v71;
    (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
      *(_QWORD *)(a4 + 88),
      v71,
      v154,
      *(_QWORD *)(a4 + 56),
      *(_QWORD *)(a4 + 64));
    v48 = v135;
    v72 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
    if ( *(_QWORD *)a4 != v72 )
    {
      v136 = v30;
      v73 = v48;
      v128 = a4;
      v74 = *(_QWORD *)a4;
      v75 = v72;
      do
      {
        v76 = *(_QWORD *)(v74 + 8);
        v77 = *(_DWORD *)v74;
        v74 += 16;
        sub_B99FD0(v73, v77, v76);
      }
      while ( v75 != v74 );
      v48 = v73;
      a4 = v128;
      v30 = v136;
    }
LABEL_53:
    v49 = sub_24941A0(a1, v41, v48, a4, a5, a6);
    v50 = (unsigned __int8 *)v49;
    if ( !v9 )
    {
      v9 = (unsigned __int8 *)v49;
      goto LABEL_29;
    }
    v51 = *(_QWORD *)(a4 + 80);
    v155 = 257;
    v52 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v51 + 16LL);
    if ( v52 == sub_9202E0 )
    {
      if ( *v9 > 0x15u || *v50 > 0x15u )
        goto LABEL_104;
      if ( (unsigned __int8)sub_AC47B0(29) )
        v53 = sub_AD5570(29, (__int64)v9, v50, 0, 0);
      else
        v53 = sub_AABE40(0x1Du, v9, v50);
    }
    else
    {
      v53 = v52(v51, 29u, v9, v50);
    }
    if ( v53 )
    {
      v9 = (unsigned __int8 *)v53;
      goto LABEL_29;
    }
LABEL_104:
    v157 = 257;
    v9 = (unsigned __int8 *)sub_B504D0(29, (__int64)v9, (__int64)v50, (__int64)v156, 0, 0);
    (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
      *(_QWORD *)(a4 + 88),
      v9,
      v154,
      *(_QWORD *)(a4 + 56),
      *(_QWORD *)(a4 + 64));
    v85 = *(_QWORD *)a4;
    v86 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
    if ( *(_QWORD *)a4 != v86 )
    {
      do
      {
        v87 = *(_QWORD *)(v85 + 8);
        v88 = *(_DWORD *)v85;
        v85 += 16;
        sub_B99FD0((__int64)v9, v88, v87);
      }
      while ( v86 != v85 );
    }
LABEL_29:
    ++v30;
  }
  while ( v146 != v30 );
  if ( !v9 )
    goto LABEL_31;
  return (__int64)v9;
}
