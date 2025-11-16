// Function: sub_178AA50
// Address: 0x178aa50
//
__int64 __fastcall sub_178AA50(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdi
  __int64 v5; // r14
  __int64 **v6; // rax
  __int64 *v7; // r15
  __int64 *v8; // r11
  __int64 *v9; // rbx
  unsigned int v10; // ecx
  __int64 v11; // rdx
  unsigned __int8 v12; // al
  __int64 v13; // r12
  __int64 **v14; // r9
  __int64 *v15; // r8
  __int64 *v16; // r9
  int v17; // eax
  int v18; // edx
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r9
  __int64 *v24; // r11
  __int64 v25; // r12
  __int64 v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // r8
  int v29; // eax
  __int64 v30; // rax
  int v31; // edx
  __int64 v32; // rdx
  __int64 **v33; // rax
  __int64 *v34; // rcx
  unsigned __int64 v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rbx
  __int64 v39; // rcx
  __int64 v40; // rsi
  __int64 v41; // rcx
  __int64 v42; // r9
  __int64 v43; // r10
  __int64 v44; // r11
  __int64 v45; // r8
  __int64 v46; // r12
  __int64 v47; // rbx
  int v48; // r10d
  __int64 v49; // rax
  __int64 v50; // r14
  __int64 v51; // r15
  __int64 *v52; // rdx
  __int64 v53; // rdx
  __int64 v54; // rcx
  int v55; // eax
  __int64 v56; // rax
  int v57; // ecx
  __int64 v58; // rcx
  __int64 *v59; // rax
  unsigned __int64 v60; // rcx
  __int64 v61; // rcx
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r15
  __int64 v66; // r15
  __int64 v67; // rdx
  __int64 v68; // r14
  int v69; // eax
  __int64 v70; // rax
  int v71; // edx
  __int64 v72; // rdx
  _QWORD *v73; // rax
  __int64 v74; // rcx
  unsigned __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rdx
  __int64 *v79; // rax
  unsigned int v80; // ebx
  int v81; // r14d
  __int64 v82; // rdx
  __int64 v83; // rax
  const char *v84; // rax
  int v85; // r12d
  int v86; // r12d
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // r10
  __int64 v93; // r15
  __int64 v94; // rdx
  __int64 v95; // r12
  int v96; // eax
  __int64 v97; // rax
  int v98; // edx
  __int64 v99; // rdx
  __int64 **v100; // rax
  __int64 *v101; // rcx
  unsigned __int64 v102; // rsi
  __int64 v103; // rdx
  __int64 v104; // rdx
  __int64 v105; // r15
  __int64 v106; // rcx
  int v107; // [rsp+0h] [rbp-A0h]
  unsigned int v108; // [rsp+4h] [rbp-9Ch]
  int v109; // [rsp+8h] [rbp-98h]
  __int64 v110; // [rsp+8h] [rbp-98h]
  int v111; // [rsp+10h] [rbp-90h]
  __int64 v112; // [rsp+10h] [rbp-90h]
  __int64 *v114; // [rsp+20h] [rbp-80h]
  __int64 v115; // [rsp+28h] [rbp-78h]
  __int64 v116; // [rsp+28h] [rbp-78h]
  __int64 v117; // [rsp+28h] [rbp-78h]
  __int64 v118; // [rsp+30h] [rbp-70h]
  __int64 *v119; // [rsp+30h] [rbp-70h]
  char v120; // [rsp+38h] [rbp-68h]
  int v121; // [rsp+38h] [rbp-68h]
  __int64 *v122; // [rsp+38h] [rbp-68h]
  __int64 *v123; // [rsp+38h] [rbp-68h]
  __int64 v124; // [rsp+38h] [rbp-68h]
  __int64 v125; // [rsp+38h] [rbp-68h]
  __int64 *v126; // [rsp+38h] [rbp-68h]
  __int64 v127; // [rsp+38h] [rbp-68h]
  const char *v128; // [rsp+40h] [rbp-60h] BYREF
  __int64 v129; // [rsp+48h] [rbp-58h]
  const char **v130; // [rsp+50h] [rbp-50h] BYREF
  const char *v131; // [rsp+58h] [rbp-48h]
  __int16 v132; // [rsp+60h] [rbp-40h]

  v3 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v4 = *(_QWORD *)(a2 - 8);
  else
    v4 = a2 - 24LL * (unsigned int)v3;
  v5 = *(_QWORD *)v4;
  v120 = *(_BYTE *)(*(_QWORD *)v4 + 16LL);
  if ( (*(_BYTE *)(*(_QWORD *)v4 + 23LL) & 0x40) != 0 )
    v6 = *(__int64 ***)(v5 - 8);
  else
    v6 = (__int64 **)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
  v7 = v6[3];
  v114 = *v6;
  v118 = **v6;
  v115 = *v7;
  if ( (_DWORD)v3 != 1 )
  {
    v8 = *v6;
    v9 = v6[3];
    v10 = 1;
    do
    {
      v11 = *(_QWORD *)(v4 + 24LL * v10);
      v12 = *(_BYTE *)(v11 + 16);
      if ( v12 != v120 || v12 <= 0x17u )
        return 0;
      v13 = *(_QWORD *)(v11 + 8);
      if ( !v13 )
        return v13;
      v13 = *(_QWORD *)(v13 + 8);
      if ( v13 )
        return 0;
      if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
        v14 = *(__int64 ***)(v11 - 8);
      else
        v14 = (__int64 **)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
      v15 = *v14;
      if ( v118 != **v14 )
        return v13;
      v16 = v14[3];
      if ( v115 != *v16 )
        return v13;
      if ( (unsigned __int8)(v12 - 75) <= 1u )
      {
        v17 = *(unsigned __int16 *)(v5 + 18);
        v18 = *(unsigned __int16 *)(v11 + 18);
        BYTE1(v17) &= ~0x80u;
        BYTE1(v18) &= ~0x80u;
        if ( v17 != v18 )
          return v13;
      }
      if ( v15 != v8 )
        v8 = 0;
      if ( v16 != v9 )
        v9 = 0;
      ++v10;
    }
    while ( (_DWORD)v3 != v10 );
    v13 = (unsigned __int64)v9 | (unsigned __int64)v8;
    if ( !((unsigned __int64)v9 | (unsigned __int64)v8) )
      return v13;
    if ( !v8 )
    {
      v84 = sub_1649960((__int64)v114);
      v85 = *(_DWORD *)(a2 + 20);
      v128 = v84;
      v130 = &v128;
      v86 = v85 & 0xFFFFFFF;
      v129 = v87;
      v132 = 773;
      v131 = ".pn";
      v88 = sub_1648B60(64);
      v92 = v88;
      if ( v88 )
      {
        v93 = v88;
        v124 = v88;
        sub_15F1EA0(v88, v118, 53, 0, 0, 0);
        *(_DWORD *)(v124 + 56) = v86;
        sub_164B780(v124, (__int64 *)&v130);
        v3 = *(unsigned int *)(v124 + 56);
        sub_1648880(v124, v3, 1);
        v92 = v124;
      }
      else
      {
        v93 = 0;
      }
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v94 = *(_QWORD *)(a2 - 8);
      else
        v94 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v95 = *(_QWORD *)(v94 + 24LL * *(unsigned int *)(a2 + 56) + 8);
      v96 = *(_DWORD *)(v92 + 20) & 0xFFFFFFF;
      if ( v96 == *(_DWORD *)(v92 + 56) )
      {
        v127 = v92;
        sub_15F55D0(v92, v3, v94, v89, v90, v91);
        v92 = v127;
        v96 = *(_DWORD *)(v127 + 20) & 0xFFFFFFF;
      }
      v97 = (v96 + 1) & 0xFFFFFFF;
      v98 = v97 | *(_DWORD *)(v92 + 20) & 0xF0000000;
      *(_DWORD *)(v92 + 20) = v98;
      if ( (v98 & 0x40000000) != 0 )
        v99 = *(_QWORD *)(v92 - 8);
      else
        v99 = v93 - 24 * v97;
      v100 = (__int64 **)(v99 + 24LL * (unsigned int)(v97 - 1));
      if ( *v100 )
      {
        v101 = v100[1];
        v102 = (unsigned __int64)v100[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v102 = v101;
        if ( v101 )
          v101[2] = v102 | v101[2] & 3;
      }
      *v100 = v114;
      v103 = v114[1];
      v100[1] = (__int64 *)v103;
      if ( v103 )
        *(_QWORD *)(v103 + 16) = (unsigned __int64)(v100 + 1) | *(_QWORD *)(v103 + 16) & 3LL;
      v100[2] = (__int64 *)((unsigned __int64)(v114 + 1) | (unsigned __int64)v100[2] & 3);
      v114[1] = (__int64)v100;
      v104 = *(_DWORD *)(v92 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v92 + 23) & 0x40) != 0 )
        v105 = *(_QWORD *)(v92 - 8);
      else
        v105 = v93 - 24 * v104;
      v125 = v92;
      *(_QWORD *)(v105 + 8LL * (unsigned int)(v104 - 1) + 24LL * *(unsigned int *)(v92 + 56) + 8) = v95;
      v7 = v9;
      sub_157E9D0(*(_QWORD *)(a2 + 40) + 40LL, v92);
      v106 = *(_QWORD *)(a2 + 24);
      *(_QWORD *)(v125 + 32) = a2 + 24;
      v106 &= 0xFFFFFFFFFFFFFFF8LL;
      v40 = v125;
      *(_QWORD *)(v125 + 24) = v106 | *(_QWORD *)(v125 + 24) & 7LL;
      *(_QWORD *)(v106 + 8) = v125 + 24;
      *(_QWORD *)(a2 + 24) = *(_QWORD *)(a2 + 24) & 7LL | (v125 + 24);
      sub_170B990(*a1, v125);
      v43 = v125;
      v44 = 0;
      v114 = (__int64 *)v125;
      goto LABEL_41;
    }
    if ( !v9 )
    {
      v119 = v8;
      v128 = sub_1649960((__int64)v7);
      v130 = &v128;
      v131 = ".pn";
      v19 = *(_DWORD *)(a2 + 20);
      v129 = v20;
      v132 = 773;
      v121 = v19 & 0xFFFFFFF;
      v21 = sub_1648B60(64);
      v24 = v119;
      v25 = v21;
      if ( v21 )
      {
        v26 = v21;
        sub_15F1EA0(v21, v115, 53, 0, 0, 0);
        *(_DWORD *)(v25 + 56) = v121;
        sub_164B780(v25, (__int64 *)&v130);
        v3 = *(unsigned int *)(v25 + 56);
        sub_1648880(v25, v3, 1);
        v24 = v119;
      }
      else
      {
        v26 = 0;
      }
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v27 = *(_QWORD *)(a2 - 8);
      else
        v27 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v28 = *(_QWORD *)(v27 + 24LL * *(unsigned int *)(a2 + 56) + 8);
      v29 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
      if ( v29 == *(_DWORD *)(v25 + 56) )
      {
        v117 = *(_QWORD *)(v27 + 24LL * *(unsigned int *)(a2 + 56) + 8);
        v126 = v24;
        sub_15F55D0(v25, v3, v27, v22, v28, v23);
        v28 = v117;
        v24 = v126;
        v29 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
      }
      v30 = (v29 + 1) & 0xFFFFFFF;
      v31 = v30 | *(_DWORD *)(v25 + 20) & 0xF0000000;
      *(_DWORD *)(v25 + 20) = v31;
      if ( (v31 & 0x40000000) != 0 )
        v32 = *(_QWORD *)(v25 - 8);
      else
        v32 = v26 - 24 * v30;
      v33 = (__int64 **)(v32 + 24LL * (unsigned int)(v30 - 1));
      if ( *v33 )
      {
        v34 = v33[1];
        v35 = (unsigned __int64)v33[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v35 = v34;
        if ( v34 )
          v34[2] = v35 | v34[2] & 3;
      }
      *v33 = v7;
      v36 = v7[1];
      v33[1] = (__int64 *)v36;
      if ( v36 )
        *(_QWORD *)(v36 + 16) = (unsigned __int64)(v33 + 1) | *(_QWORD *)(v36 + 16) & 3LL;
      v33[2] = (__int64 *)((unsigned __int64)(v7 + 1) | (unsigned __int64)v33[2] & 3);
      v7[1] = (__int64)v33;
      v37 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v25 + 23) & 0x40) != 0 )
        v38 = *(_QWORD *)(v25 - 8);
      else
        v38 = v26 - 24 * v37;
      v122 = v24;
      v7 = (__int64 *)v25;
      *(_QWORD *)(v38 + 8LL * (unsigned int)(v37 - 1) + 24LL * *(unsigned int *)(v25 + 56) + 8) = v28;
      sub_157E9D0(*(_QWORD *)(a2 + 40) + 40LL, v25);
      v39 = *(_QWORD *)(a2 + 24);
      v40 = v25;
      *(_QWORD *)(v25 + 32) = a2 + 24;
      v39 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v25 + 24) = v39 | *(_QWORD *)(v25 + 24) & 7LL;
      *(_QWORD *)(v39 + 8) = v25 + 24;
      *(_QWORD *)(a2 + 24) = *(_QWORD *)(a2 + 24) & 7LL | (v25 + 24);
      sub_170B990(*a1, v25);
      v43 = 0;
      v114 = v122;
      v44 = v25;
LABEL_41:
      v45 = 1;
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 1 )
      {
        v46 = v43;
        v123 = v7;
        v47 = v44;
        v48 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        v116 = v5;
        do
        {
          if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
            v49 = *(_QWORD *)(a2 - 8);
          else
            v49 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          v50 = (unsigned int)v45;
          v51 = *(_QWORD *)(v49 + 24LL * (unsigned int)v45);
          if ( v46 )
          {
            if ( (*(_BYTE *)(v51 + 23) & 0x40) != 0 )
              v52 = *(__int64 **)(v51 - 8);
            else
              v52 = (__int64 *)(v51 - 24LL * (*(_DWORD *)(v51 + 20) & 0xFFFFFFF));
            v53 = *v52;
            v54 = 8LL * (unsigned int)v45 + 24LL * *(unsigned int *)(a2 + 56);
            v42 = *(_QWORD *)(v49 + v54 + 8);
            v55 = *(_DWORD *)(v46 + 20) & 0xFFFFFFF;
            if ( v55 == *(_DWORD *)(v46 + 56) )
            {
              v107 = v48;
              v108 = v45;
              v110 = v53;
              v112 = v42;
              sub_15F55D0(v46, 3LL * *(unsigned int *)(a2 + 56), v53, v54, v45, v42);
              v48 = v107;
              v45 = v108;
              v53 = v110;
              v42 = v112;
              v55 = *(_DWORD *)(v46 + 20) & 0xFFFFFFF;
            }
            v56 = (v55 + 1) & 0xFFFFFFF;
            v40 = (unsigned int)(v56 - 1);
            v57 = v56 | *(_DWORD *)(v46 + 20) & 0xF0000000;
            *(_DWORD *)(v46 + 20) = v57;
            if ( (v57 & 0x40000000) != 0 )
              v58 = *(_QWORD *)(v46 - 8);
            else
              v58 = v46 - 24 * v56;
            v59 = (__int64 *)(v58 + 24LL * (unsigned int)v40);
            if ( *v59 )
            {
              v40 = v59[1];
              v60 = v59[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v60 = v40;
              if ( v40 )
                *(_QWORD *)(v40 + 16) = *(_QWORD *)(v40 + 16) & 3LL | v60;
            }
            *v59 = v53;
            if ( v53 )
            {
              v61 = *(_QWORD *)(v53 + 8);
              v59[1] = v61;
              if ( v61 )
              {
                v40 = (unsigned __int64)(v59 + 1) | *(_QWORD *)(v61 + 16) & 3LL;
                *(_QWORD *)(v61 + 16) = v40;
              }
              v59[2] = v59[2] & 3 | (v53 + 8);
              *(_QWORD *)(v53 + 8) = v59;
            }
            v62 = *(_DWORD *)(v46 + 20) & 0xFFFFFFF;
            v63 = (unsigned int)(v62 - 1);
            if ( (*(_BYTE *)(v46 + 23) & 0x40) != 0 )
              v64 = *(_QWORD *)(v46 - 8);
            else
              v64 = v46 - 24 * v62;
            v41 = 3LL * *(unsigned int *)(v46 + 56);
            *(_QWORD *)(v64 + 8 * v63 + 24LL * *(unsigned int *)(v46 + 56) + 8) = v42;
          }
          if ( v47 )
          {
            if ( (*(_BYTE *)(v51 + 23) & 0x40) != 0 )
              v65 = *(_QWORD *)(v51 - 8);
            else
              v65 = v51 - 24LL * (*(_DWORD *)(v51 + 20) & 0xFFFFFFF);
            v66 = *(_QWORD *)(v65 + 24);
            if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
              v67 = *(_QWORD *)(a2 - 8);
            else
              v67 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
            v68 = *(_QWORD *)(v67 + 8 * v50 + 24LL * *(unsigned int *)(a2 + 56) + 8);
            v69 = *(_DWORD *)(v47 + 20) & 0xFFFFFFF;
            if ( v69 == *(_DWORD *)(v47 + 56) )
            {
              v109 = v48;
              v111 = v45;
              sub_15F55D0(v47, v40, v67, v41, v45, v42);
              v48 = v109;
              LODWORD(v45) = v111;
              v69 = *(_DWORD *)(v47 + 20) & 0xFFFFFFF;
            }
            v70 = (v69 + 1) & 0xFFFFFFF;
            v71 = v70 | *(_DWORD *)(v47 + 20) & 0xF0000000;
            *(_DWORD *)(v47 + 20) = v71;
            if ( (v71 & 0x40000000) != 0 )
              v72 = *(_QWORD *)(v47 - 8);
            else
              v72 = v47 - 24 * v70;
            v73 = (_QWORD *)(v72 + 24LL * (unsigned int)(v70 - 1));
            if ( *v73 )
            {
              v74 = v73[1];
              v75 = v73[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v75 = v74;
              if ( v74 )
              {
                v40 = *(_QWORD *)(v74 + 16) & 3LL;
                *(_QWORD *)(v74 + 16) = v40 | v75;
              }
            }
            *v73 = v66;
            if ( v66 )
            {
              v76 = *(_QWORD *)(v66 + 8);
              v40 = v66 + 8;
              v73[1] = v76;
              if ( v76 )
                *(_QWORD *)(v76 + 16) = (unsigned __int64)(v73 + 1) | *(_QWORD *)(v76 + 16) & 3LL;
              v73[2] = v40 | v73[2] & 3LL;
              *(_QWORD *)(v66 + 8) = v73;
            }
            v77 = *(_DWORD *)(v47 + 20) & 0xFFFFFFF;
            if ( (*(_BYTE *)(v47 + 23) & 0x40) != 0 )
              v41 = *(_QWORD *)(v47 - 8);
            else
              v41 = v47 - 24 * v77;
            *(_QWORD *)(v41 + 8LL * (unsigned int)(v77 - 1) + 24LL * *(unsigned int *)(v47 + 56) + 8) = v68;
          }
          v45 = (unsigned int)(v45 + 1);
        }
        while ( (_DWORD)v45 != v48 );
        v7 = v123;
        v5 = v116;
      }
      v120 = *(_BYTE *)(v5 + 16);
      goto LABEL_96;
    }
    v114 = v8;
    v7 = v9;
  }
LABEL_96:
  v132 = 257;
  if ( (unsigned __int8)(v120 - 75) > 1u )
  {
    v13 = sub_15FB440((unsigned int)*(unsigned __int8 *)(v5 + 16) - 24, v114, (__int64)v7, (__int64)&v130, 0);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v79 = *(__int64 **)(a2 - 8);
    else
      v79 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v80 = 1;
    sub_15F2530((unsigned __int8 *)v13, *v79, 1);
    v81 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( v81 != 1 )
    {
      do
      {
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v82 = *(_QWORD *)(a2 - 8);
        else
          v82 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v83 = v80++;
        sub_15F2780((unsigned __int8 *)v13, *(_QWORD *)(v82 + 24 * v83));
      }
      while ( v80 != v81 );
    }
    sub_1789760((__int64)a1, v13, a2);
  }
  else
  {
    v13 = sub_15FEEB0(
            (unsigned int)*(unsigned __int8 *)(v5 + 16) - 24,
            *(_WORD *)(v5 + 18) & 0x7FFF,
            (__int64)v114,
            (__int64)v7,
            (__int64)&v130,
            0);
    sub_1789760((__int64)a1, v13, a2);
  }
  return v13;
}
