// Function: sub_1D87110
// Address: 0x1d87110
//
__int64 __fastcall sub_1D87110(__int64 a1, unsigned int a2, _DWORD *a3, double a4, double a5, double a6)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rbx
  __int64 v15; // r15
  __int64 v16; // r13
  _BYTE *v17; // rsi
  unsigned int *v18; // rbx
  unsigned int v19; // r13d
  _QWORD *v20; // rax
  __int64 *v21; // r13
  __int64 v22; // r15
  __int64 v23; // r14
  __int64 **v24; // rdx
  __int64 **v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rdi
  __int64 *v30; // r15
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // rsi
  unsigned __int8 *v35; // rsi
  _QWORD *v36; // rax
  __int64 v37; // rdi
  __int64 *v38; // r14
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // rsi
  __int64 v42; // rsi
  unsigned __int8 *v43; // rsi
  __int64 v44; // r13
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 *v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 v51; // rsi
  __int64 v52; // rsi
  unsigned __int8 *v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 *v56; // rbx
  __int64 v57; // rax
  __int64 v58; // rcx
  __int64 v59; // rsi
  __int64 v60; // rsi
  unsigned __int8 *v61; // rsi
  __int64 v62; // rax
  __int64 v63; // rdi
  __int64 v64; // rsi
  __int64 v65; // rax
  __int64 v66; // rsi
  __int64 v67; // rsi
  __int64 v68; // rdx
  unsigned __int8 *v69; // rsi
  __int64 v70; // rax
  __int64 v71; // rdi
  __int64 v72; // rsi
  __int64 v73; // rax
  __int64 v74; // rsi
  __int64 v75; // rsi
  __int64 v76; // rdx
  unsigned __int8 *v77; // rsi
  __int64 v78; // rax
  __int64 v79; // rdi
  __int64 *v80; // r13
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // rsi
  __int64 v84; // rsi
  unsigned __int8 *v85; // rsi
  __int64 v86; // rax
  __int64 v87; // rdi
  __int64 *v88; // r13
  __int64 v89; // rax
  __int64 v90; // rcx
  __int64 v91; // rsi
  __int64 v92; // rsi
  unsigned __int8 *v93; // rsi
  unsigned int v94; // r13d
  _QWORD *v95; // rax
  unsigned __int8 *v96; // rax
  unsigned __int8 *v97; // rdi
  __int64 v98; // rsi
  __int64 v99; // rax
  __int64 v100; // rax
  unsigned __int8 *v101; // rax
  unsigned __int8 *v102; // rdi
  __int64 v103; // rsi
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // r13
  __int64 v109; // rsi
  __int64 v110; // rsi
  unsigned __int8 *v111; // rsi
  __int64 *v113; // [rsp+18h] [rbp-D8h]
  __int64 *v114; // [rsp+18h] [rbp-D8h]
  __int64 **v115; // [rsp+20h] [rbp-D0h]
  unsigned int v116; // [rsp+28h] [rbp-C8h]
  unsigned int v117; // [rsp+2Ch] [rbp-C4h]
  __int64 *v118; // [rsp+30h] [rbp-C0h] BYREF
  unsigned __int8 *v119; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v120; // [rsp+40h] [rbp-B0h] BYREF
  _BYTE *v121; // [rsp+48h] [rbp-A8h]
  _BYTE *v122; // [rsp+50h] [rbp-A0h]
  unsigned __int8 *v123; // [rsp+60h] [rbp-90h] BYREF
  __int64 v124; // [rsp+68h] [rbp-88h]
  __int64 v125; // [rsp+70h] [rbp-80h]
  __int64 v126[2]; // [rsp+80h] [rbp-70h] BYREF
  __int16 v127; // [rsp+90h] [rbp-60h]
  unsigned __int8 *v128; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v129; // [rsp+A8h] [rbp-48h]
  __int64 v130; // [rsp+B0h] [rbp-40h]

  v7 = (unsigned int)*a3;
  v8 = *(unsigned int *)(a1 + 200);
  v120 = 0;
  v9 = v8 - v7;
  v10 = *(_QWORD *)(a1 + 56);
  v121 = 0;
  v122 = 0;
  v125 = 0;
  v123 = 0;
  if ( v9 > v10 )
    LODWORD(v9) = v10;
  v11 = *(_QWORD *)(a1 + 64);
  v124 = 0;
  v116 = v9;
  if ( v11 != *(_QWORD *)(a1 + 72) )
  {
    v12 = *(_QWORD *)(v11 + 8LL * a2);
    *(_QWORD *)(a1 + 128) = v12;
    *(_QWORD *)(a1 + 136) = v12 + 40;
    goto LABEL_5;
  }
  v107 = *(_QWORD *)a1;
  v108 = a1 + 120;
  *(_QWORD *)(a1 + 128) = *(_QWORD *)(*(_QWORD *)a1 + 40LL);
  *(_QWORD *)(a1 + 136) = v107 + 24;
  v109 = *(_QWORD *)(v107 + 48);
  v128 = (unsigned __int8 *)v109;
  if ( v109 )
  {
    sub_1623A60((__int64)&v128, v109, 2);
    v110 = *(_QWORD *)(a1 + 120);
    if ( !v110 )
      goto LABEL_122;
    goto LABEL_121;
  }
  v110 = *(_QWORD *)(a1 + 120);
  if ( v110 )
  {
LABEL_121:
    sub_161E7C0(v108, v110);
LABEL_122:
    v111 = v128;
    *(_QWORD *)(a1 + 120) = v128;
    if ( v111 )
      sub_1623210((__int64)&v128, v111, v108);
  }
LABEL_5:
  v115 = 0;
  if ( (_DWORD)v9 != 1 )
  {
    v94 = 8 * *(_DWORD *)(a1 + 40);
    v95 = (_QWORD *)sub_16498A0(*(_QWORD *)a1);
    v115 = (__int64 **)sub_1644900(v95, v94);
    if ( !(_DWORD)v9 )
    {
      v126[0] = a1;
LABEL_110:
      sub_1D86E80(&v128, v126, &v120, a4, a5, a6);
      v96 = v128;
      v97 = v123;
      v128 = 0;
      v98 = v125;
      v123 = v96;
      v99 = v129;
      v129 = 0;
      v124 = v99;
      v100 = v130;
      v130 = 0;
      v125 = v100;
      if ( v97 )
      {
        j_j___libc_free_0(v97, v98 - (_QWORD)v97);
        if ( v128 )
          j_j___libc_free_0(v128, v130 - (_QWORD)v128);
      }
      while ( v124 - (_QWORD)v123 != 8 )
      {
        sub_1D86E80(&v128, v126, (__int64 *)&v123, a4, a5, a6);
        v101 = v128;
        v102 = v123;
        v128 = 0;
        v103 = v125;
        v123 = v101;
        v104 = v129;
        v129 = 0;
        v124 = v104;
        v105 = v130;
        v130 = 0;
        v125 = v105;
        if ( v102 )
        {
          j_j___libc_free_0(v102, v103 - (_QWORD)v102);
          if ( v128 )
            j_j___libc_free_0(v128, v130 - (_QWORD)v128);
        }
      }
      LOWORD(v130) = 257;
      v106 = sub_15A0680(*v118, 0, 0);
      v44 = sub_12AA0C0((__int64 *)(a1 + 120), 0x21u, *(_BYTE **)v123, v106, (__int64)&v128);
      goto LABEL_60;
    }
  }
  v117 = 0;
  v13 = (unsigned int)*a3;
  do
  {
    v18 = (unsigned int *)(*(_QWORD *)(a1 + 192) + 16 * v13);
    v19 = 8 * *v18;
    v20 = (_QWORD *)sub_16498A0(*(_QWORD *)a1);
    v21 = (__int64 *)sub_1644900(v20, v19);
    v22 = *(_QWORD *)(*(_QWORD *)a1 - 24LL * (*(_DWORD *)(*(_QWORD *)a1 + 20LL) & 0xFFFFFFF));
    v23 = *(_QWORD *)(*(_QWORD *)a1 + 24 * (1LL - (*(_DWORD *)(*(_QWORD *)a1 + 20LL) & 0xFFFFFFF)));
    if ( *(__int64 **)v22 != v21 )
    {
      v127 = 257;
      v24 = (__int64 **)sub_1647190(v21, 0);
      if ( v24 != *(__int64 ***)v22 )
      {
        if ( *(_BYTE *)(v22 + 16) > 0x10u )
        {
          LOWORD(v130) = 257;
          v62 = sub_15FDBD0(47, v22, (__int64)v24, (__int64)&v128, 0);
          v63 = *(_QWORD *)(a1 + 128);
          v22 = v62;
          if ( v63 )
          {
            v113 = *(__int64 **)(a1 + 136);
            sub_157E9D0(v63 + 40, v62);
            v64 = *v113;
            v65 = *(_QWORD *)(v22 + 24) & 7LL;
            *(_QWORD *)(v22 + 32) = v113;
            v64 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v22 + 24) = v64 | v65;
            *(_QWORD *)(v64 + 8) = v22 + 24;
            *v113 = *v113 & 7 | (v22 + 24);
          }
          sub_164B780(v22, v126);
          v66 = *(_QWORD *)(a1 + 120);
          if ( v66 )
          {
            v119 = *(unsigned __int8 **)(a1 + 120);
            sub_1623A60((__int64)&v119, v66, 2);
            v67 = *(_QWORD *)(v22 + 48);
            v68 = v22 + 48;
            if ( v67 )
            {
              sub_161E7C0(v22 + 48, v67);
              v68 = v22 + 48;
            }
            v69 = v119;
            *(_QWORD *)(v22 + 48) = v119;
            if ( v69 )
              sub_1623210((__int64)&v119, v69, v68);
          }
        }
        else
        {
          v22 = sub_15A46C0(47, (__int64 ***)v22, v24, 0);
        }
      }
    }
    if ( v21 != *(__int64 **)v23 )
    {
      v127 = 257;
      v25 = (__int64 **)sub_1647190(v21, 0);
      if ( v25 != *(__int64 ***)v23 )
      {
        if ( *(_BYTE *)(v23 + 16) > 0x10u )
        {
          LOWORD(v130) = 257;
          v70 = sub_15FDBD0(47, v23, (__int64)v25, (__int64)&v128, 0);
          v71 = *(_QWORD *)(a1 + 128);
          v23 = v70;
          if ( v71 )
          {
            v114 = *(__int64 **)(a1 + 136);
            sub_157E9D0(v71 + 40, v70);
            v72 = *v114;
            v73 = *(_QWORD *)(v23 + 24) & 7LL;
            *(_QWORD *)(v23 + 32) = v114;
            v72 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v23 + 24) = v72 | v73;
            *(_QWORD *)(v72 + 8) = v23 + 24;
            *v114 = *v114 & 7 | (v23 + 24);
          }
          sub_164B780(v23, v126);
          v74 = *(_QWORD *)(a1 + 120);
          if ( v74 )
          {
            v119 = *(unsigned __int8 **)(a1 + 120);
            sub_1623A60((__int64)&v119, v74, 2);
            v75 = *(_QWORD *)(v23 + 48);
            v76 = v23 + 48;
            if ( v75 )
            {
              sub_161E7C0(v23 + 48, v75);
              v76 = v23 + 48;
            }
            v77 = v119;
            *(_QWORD *)(v23 + 48) = v119;
            if ( v77 )
              sub_1623210((__int64)&v119, v77, v76);
          }
        }
        else
        {
          v23 = sub_15A46C0(47, (__int64 ***)v23, v25, 0);
        }
      }
    }
    if ( *((_QWORD *)v18 + 1) )
    {
      LOWORD(v130) = 257;
      v26 = sub_159C470((__int64)v21, *((_QWORD *)v18 + 1) / (unsigned __int64)*v18, 0);
      v22 = sub_12815B0((__int64 *)(a1 + 120), (__int64)v21, (_BYTE *)v22, v26, (__int64)&v128);
      LOWORD(v130) = 257;
      v27 = sub_159C470((__int64)v21, *((_QWORD *)v18 + 1) / (unsigned __int64)*v18, 0);
      v23 = sub_12815B0((__int64 *)(a1 + 120), (__int64)v21, (_BYTE *)v23, v27, (__int64)&v128);
      if ( *(_BYTE *)(v22 + 16) <= 0x10u )
      {
LABEL_8:
        v14 = sub_14D8290(v22, (__int64)v21, *(_BYTE **)(a1 + 112));
        if ( v14 )
          goto LABEL_9;
      }
    }
    else if ( *(_BYTE *)(v22 + 16) <= 0x10u )
    {
      goto LABEL_8;
    }
    LOWORD(v130) = 257;
    v28 = sub_1648A60(64, 1u);
    v14 = (__int64)v28;
    if ( v28 )
      sub_15F9210((__int64)v28, (__int64)v21, v22, 0, 0, 0);
    v29 = *(_QWORD *)(a1 + 128);
    if ( v29 )
    {
      v30 = *(__int64 **)(a1 + 136);
      sub_157E9D0(v29 + 40, v14);
      v31 = *(_QWORD *)(v14 + 24);
      v32 = *v30;
      *(_QWORD *)(v14 + 32) = v30;
      v32 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v14 + 24) = v32 | v31 & 7;
      *(_QWORD *)(v32 + 8) = v14 + 24;
      *v30 = *v30 & 7 | (v14 + 24);
    }
    sub_164B780(v14, (__int64 *)&v128);
    v33 = *(_QWORD *)(a1 + 120);
    if ( v33 )
    {
      v126[0] = *(_QWORD *)(a1 + 120);
      sub_1623A60((__int64)v126, v33, 2);
      v34 = *(_QWORD *)(v14 + 48);
      if ( v34 )
        sub_161E7C0(v14 + 48, v34);
      v35 = (unsigned __int8 *)v126[0];
      *(_QWORD *)(v14 + 48) = v126[0];
      if ( v35 )
      {
        sub_1623210((__int64)v126, v35, v14 + 48);
        if ( *(_BYTE *)(v23 + 16) > 0x10u )
          goto LABEL_49;
        goto LABEL_10;
      }
    }
LABEL_9:
    if ( *(_BYTE *)(v23 + 16) > 0x10u )
      goto LABEL_49;
LABEL_10:
    v15 = sub_14D8290(v23, (__int64)v21, *(_BYTE **)(a1 + 112));
    if ( v15 )
      goto LABEL_11;
LABEL_49:
    LOWORD(v130) = 257;
    v36 = sub_1648A60(64, 1u);
    v15 = (__int64)v36;
    if ( v36 )
      sub_15F9210((__int64)v36, (__int64)v21, v23, 0, 0, 0);
    v37 = *(_QWORD *)(a1 + 128);
    if ( v37 )
    {
      v38 = *(__int64 **)(a1 + 136);
      sub_157E9D0(v37 + 40, v15);
      v39 = *(_QWORD *)(v15 + 24);
      v40 = *v38;
      *(_QWORD *)(v15 + 32) = v38;
      v40 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v15 + 24) = v40 | v39 & 7;
      *(_QWORD *)(v40 + 8) = v15 + 24;
      *v38 = *v38 & 7 | (v15 + 24);
    }
    sub_164B780(v15, (__int64 *)&v128);
    v41 = *(_QWORD *)(a1 + 120);
    if ( !v41 )
      goto LABEL_11;
    v126[0] = *(_QWORD *)(a1 + 120);
    sub_1623A60((__int64)v126, v41, 2);
    v42 = *(_QWORD *)(v15 + 48);
    if ( v42 )
      sub_161E7C0(v15 + 48, v42);
    v43 = (unsigned __int8 *)v126[0];
    *(_QWORD *)(v15 + 48) = v126[0];
    if ( !v43 )
    {
LABEL_11:
      if ( v116 == 1 )
        goto LABEL_58;
      goto LABEL_12;
    }
    sub_1623210((__int64)v126, v43, v15 + 48);
    if ( v116 == 1 )
    {
LABEL_58:
      LOWORD(v130) = 257;
      v44 = sub_12AA0C0((__int64 *)(a1 + 120), 0x21u, (_BYTE *)v14, v15, (__int64)&v128);
      ++*a3;
      goto LABEL_59;
    }
LABEL_12:
    if ( v21 != (__int64 *)v115 )
    {
      v127 = 257;
      if ( *(__int64 ***)v14 != v115 )
      {
        if ( *(_BYTE *)(v14 + 16) > 0x10u )
        {
          LOWORD(v130) = 257;
          v86 = sub_15FDBD0(37, v14, (__int64)v115, (__int64)&v128, 0);
          v87 = *(_QWORD *)(a1 + 128);
          v14 = v86;
          if ( v87 )
          {
            v88 = *(__int64 **)(a1 + 136);
            sub_157E9D0(v87 + 40, v86);
            v89 = *(_QWORD *)(v14 + 24);
            v90 = *v88;
            *(_QWORD *)(v14 + 32) = v88;
            v90 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v14 + 24) = v90 | v89 & 7;
            *(_QWORD *)(v90 + 8) = v14 + 24;
            *v88 = *v88 & 7 | (v14 + 24);
          }
          sub_164B780(v14, v126);
          v91 = *(_QWORD *)(a1 + 120);
          if ( v91 )
          {
            v119 = *(unsigned __int8 **)(a1 + 120);
            sub_1623A60((__int64)&v119, v91, 2);
            v92 = *(_QWORD *)(v14 + 48);
            if ( v92 )
              sub_161E7C0(v14 + 48, v92);
            v93 = v119;
            *(_QWORD *)(v14 + 48) = v119;
            if ( v93 )
              sub_1623210((__int64)&v119, v93, v14 + 48);
          }
        }
        else
        {
          v14 = sub_15A46C0(37, (__int64 ***)v14, v115, 0);
        }
      }
      v127 = 257;
      if ( *(__int64 ***)v15 != v115 )
      {
        if ( *(_BYTE *)(v15 + 16) > 0x10u )
        {
          LOWORD(v130) = 257;
          v78 = sub_15FDBD0(37, v15, (__int64)v115, (__int64)&v128, 0);
          v79 = *(_QWORD *)(a1 + 128);
          v15 = v78;
          if ( v79 )
          {
            v80 = *(__int64 **)(a1 + 136);
            sub_157E9D0(v79 + 40, v78);
            v81 = *(_QWORD *)(v15 + 24);
            v82 = *v80;
            *(_QWORD *)(v15 + 32) = v80;
            v82 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v15 + 24) = v82 | v81 & 7;
            *(_QWORD *)(v82 + 8) = v15 + 24;
            *v80 = *v80 & 7 | (v15 + 24);
          }
          sub_164B780(v15, v126);
          v83 = *(_QWORD *)(a1 + 120);
          if ( v83 )
          {
            v119 = *(unsigned __int8 **)(a1 + 120);
            sub_1623A60((__int64)&v119, v83, 2);
            v84 = *(_QWORD *)(v15 + 48);
            if ( v84 )
              sub_161E7C0(v15 + 48, v84);
            v85 = v119;
            *(_QWORD *)(v15 + 48) = v119;
            if ( v85 )
              sub_1623210((__int64)&v119, v85, v15 + 48);
          }
        }
        else
        {
          v15 = sub_15A46C0(37, (__int64 ***)v15, v115, 0);
        }
      }
    }
    v127 = 257;
    if ( *(_BYTE *)(v14 + 16) > 0x10u
      || *(_BYTE *)(v15 + 16) > 0x10u
      || (v16 = sub_15A2A30((__int64 *)0x1C, (__int64 *)v14, v15, 0, 0, a4, a5, a6)) == 0 )
    {
      LOWORD(v130) = 257;
      v46 = sub_15FB440(28, (__int64 *)v14, v15, (__int64)&v128, 0);
      v47 = *(_QWORD *)(a1 + 128);
      v16 = v46;
      if ( v47 )
      {
        v48 = *(__int64 **)(a1 + 136);
        sub_157E9D0(v47 + 40, v46);
        v49 = *(_QWORD *)(v16 + 24);
        v50 = *v48;
        *(_QWORD *)(v16 + 32) = v48;
        v50 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v16 + 24) = v50 | v49 & 7;
        *(_QWORD *)(v50 + 8) = v16 + 24;
        *v48 = *v48 & 7 | (v16 + 24);
      }
      sub_164B780(v16, v126);
      v51 = *(_QWORD *)(a1 + 120);
      if ( v51 )
      {
        v119 = *(unsigned __int8 **)(a1 + 120);
        sub_1623A60((__int64)&v119, v51, 2);
        v52 = *(_QWORD *)(v16 + 48);
        if ( v52 )
          sub_161E7C0(v16 + 48, v52);
        v53 = v119;
        *(_QWORD *)(v16 + 48) = v119;
        if ( v53 )
          sub_1623210((__int64)&v119, v53, v16 + 48);
      }
    }
    v118 = (__int64 *)v16;
    v127 = 257;
    if ( *(__int64 ***)v16 != v115 )
    {
      if ( *(_BYTE *)(v16 + 16) > 0x10u )
      {
        LOWORD(v130) = 257;
        v54 = sub_15FDBD0(37, v16, (__int64)v115, (__int64)&v128, 0);
        v55 = *(_QWORD *)(a1 + 128);
        v16 = v54;
        if ( v55 )
        {
          v56 = *(__int64 **)(a1 + 136);
          sub_157E9D0(v55 + 40, v54);
          v57 = *(_QWORD *)(v16 + 24);
          v58 = *v56;
          *(_QWORD *)(v16 + 32) = v56;
          v58 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v16 + 24) = v58 | v57 & 7;
          *(_QWORD *)(v58 + 8) = v16 + 24;
          *v56 = *v56 & 7 | (v16 + 24);
        }
        sub_164B780(v16, v126);
        v59 = *(_QWORD *)(a1 + 120);
        if ( v59 )
        {
          v119 = *(unsigned __int8 **)(a1 + 120);
          sub_1623A60((__int64)&v119, v59, 2);
          v60 = *(_QWORD *)(v16 + 48);
          if ( v60 )
            sub_161E7C0(v16 + 48, v60);
          v61 = v119;
          *(_QWORD *)(v16 + 48) = v119;
          if ( v61 )
            sub_1623210((__int64)&v119, v61, v16 + 48);
        }
      }
      else
      {
        v16 = sub_15A46C0(37, (__int64 ***)v16, v115, 0);
      }
    }
    v118 = (__int64 *)v16;
    v17 = v121;
    if ( v121 == v122 )
    {
      sub_1287830((__int64)&v120, v121, &v118);
    }
    else
    {
      if ( v121 )
      {
        *(_QWORD *)v121 = v16;
        v17 = v121;
      }
      v121 = v17 + 8;
    }
    ++v117;
    v13 = (unsigned int)(*a3 + 1);
    *a3 = v13;
  }
  while ( v116 > v117 );
  v44 = 0;
LABEL_59:
  v126[0] = a1;
  if ( !v44 )
    goto LABEL_110;
LABEL_60:
  if ( v123 )
    j_j___libc_free_0(v123, v125 - (_QWORD)v123);
  if ( v120 )
    j_j___libc_free_0(v120, &v122[-v120]);
  return v44;
}
