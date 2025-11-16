// Function: sub_1D62410
// Address: 0x1d62410
//
__int64 __fastcall sub_1D62410(__int64 *a1, __int64 a2, int a3, unsigned int a4, _BYTE *a5)
{
  _QWORD *v5; // r15
  __int64 v6; // r14
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rdx
  unsigned __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // r13d
  int v16; // r8d
  int v17; // r9d
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  unsigned __int64 v21; // rax
  __int64 *v22; // rax
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rax
  __int64 v26; // r13
  unsigned __int64 v27; // rax
  unsigned int v28; // r8d
  __int64 (__fastcall *v30)(__int64, __int64, __m128, double, double, double, double, double, double, __m128, __int64, int *, __int64, __int64, _QWORD *); // r13
  __int64 v31; // rdx
  __int64 v32; // rax
  _QWORD *v33; // rdi
  __int64 v34; // r13
  __int64 *v35; // rax
  __int64 v36; // r15
  unsigned int v37; // ebx
  int v38; // r9d
  unsigned int v39; // eax
  bool v40; // al
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rdx
  unsigned int v44; // esi
  __int64 v45; // rdx
  __int64 v46; // rax
  unsigned int v47; // eax
  char v48; // bl
  __int64 ***v49; // rax
  __int64 **v50; // rax
  unsigned int v51; // edx
  __int64 *v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdi
  __int64 (*v56)(); // rcx
  __int64 *v57; // r13
  __int64 v58; // r8
  unsigned int v59; // r12d
  __int64 v60; // rbx
  unsigned __int64 v61; // r8
  __int64 v62; // r15
  __int64 v63; // r14
  int v64; // ebx
  __int64 v65; // rbx
  __int64 v66; // rax
  __int64 v67; // r8
  __int64 v68; // rax
  bool v69; // cc
  _QWORD *v70; // rax
  char v71; // al
  __int64 **v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rsi
  unsigned __int64 v76; // r14
  __int64 v77; // rax
  __int64 v78; // rax
  unsigned int v79; // esi
  __int64 *v80; // rax
  __int64 v81; // rcx
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rbx
  __int64 v85; // rdx
  char v86; // si
  __int64 v87; // rdi
  unsigned __int64 v88; // rcx
  __int64 *v89; // rax
  __int64 v90; // rcx
  int v91; // r8d
  int v92; // r9d
  __int64 v93; // rax
  __int64 v94; // r12
  __int64 v95; // rax
  __int64 v96; // r13
  int v97; // r8d
  int v98; // r9d
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rax
  _QWORD *v102; // rax
  __int64 i; // rdx
  __int64 *v104; // rax
  _QWORD *v105; // rax
  __int64 j; // rdx
  __int64 ***v107; // rax
  __int64 v108; // rdx
  char v109; // al
  __int64 v110; // rax
  __int64 v111; // rcx
  unsigned __int8 v112; // al
  __int64 v113; // rdi
  unsigned __int64 v114; // rax
  __int64 v115; // rdx
  __int64 *v116; // rax
  unsigned int v117; // [rsp+Ch] [rbp-94h]
  unsigned int v118; // [rsp+10h] [rbp-90h]
  __int64 v119; // [rsp+10h] [rbp-90h]
  int v120; // [rsp+18h] [rbp-88h]
  __int64 v121; // [rsp+18h] [rbp-88h]
  __int64 v123; // [rsp+20h] [rbp-80h]
  __int64 v124; // [rsp+28h] [rbp-78h]
  __int64 v125; // [rsp+28h] [rbp-78h]
  int v126; // [rsp+28h] [rbp-78h]
  unsigned __int64 v127; // [rsp+28h] [rbp-78h]
  __int64 v128; // [rsp+30h] [rbp-70h]
  __int64 v129; // [rsp+30h] [rbp-70h]
  __int64 v131; // [rsp+38h] [rbp-68h]
  __int64 v132; // [rsp+38h] [rbp-68h]
  __int64 v134; // [rsp+38h] [rbp-68h]
  __int64 v135; // [rsp+40h] [rbp-60h]
  char v136; // [rsp+40h] [rbp-60h]
  unsigned __int64 v137; // [rsp+40h] [rbp-60h]
  __int64 v138; // [rsp+40h] [rbp-60h]
  char v139; // [rsp+48h] [rbp-58h]
  __int64 v140; // [rsp+48h] [rbp-58h]
  __int64 v141; // [rsp+48h] [rbp-58h]
  __int64 v142; // [rsp+50h] [rbp-50h]
  __int64 v143; // [rsp+50h] [rbp-50h]
  unsigned __int64 v144; // [rsp+50h] [rbp-50h]
  unsigned __int64 v145; // [rsp+50h] [rbp-50h]
  unsigned __int64 v146; // [rsp+50h] [rbp-50h]
  __int64 v147; // [rsp+58h] [rbp-48h]
  unsigned __int8 v148; // [rsp+58h] [rbp-48h]
  unsigned __int8 v149; // [rsp+58h] [rbp-48h]
  __int64 v150; // [rsp+58h] [rbp-48h]
  unsigned int v151[13]; // [rsp+6Ch] [rbp-34h] BYREF

  v5 = (_QWORD *)a2;
  v6 = (__int64)a1;
  if ( a5 )
    *a5 = 0;
  switch ( a3 )
  {
    case 11:
      v9 = a1[7];
      v10 = 0;
      v147 = *(_QWORD *)v9;
      v139 = *(_BYTE *)(v9 + 16);
      v142 = *(_QWORD *)(v9 + 8);
      v131 = *(_QWORD *)(v9 + 32);
      v135 = *(_QWORD *)(v9 + 24);
      v128 = *(_QWORD *)(v9 + 40);
      v11 = a1[10];
      v124 = *(_QWORD *)(v9 + 48);
      v12 = *(unsigned int *)(*a1 + 8);
      v13 = *(unsigned int *)(v11 + 8);
      v120 = *(_DWORD *)(*a1 + 8);
      if ( (_DWORD)v13 )
        v10 = *(_QWORD *)(*(_QWORD *)v11 + 8 * v13 - 8);
      v14 = sub_13CF970(a2);
      v15 = a4 + 1;
      if ( (unsigned __int8)sub_1D61F00((__int64)a1, *(_QWORD *)(v14 + 24), a4 + 1) )
      {
        v18 = (__int64 *)sub_13CF970(a2);
        if ( (unsigned __int8)sub_1D61F00((__int64)a1, *v18, v15) )
          return 1;
      }
      v19 = a1[7];
      *(_QWORD *)v19 = v147;
      *(_BYTE *)(v19 + 16) = v139;
      *(_QWORD *)(v19 + 8) = v142;
      *(_QWORD *)(v19 + 32) = v131;
      *(_QWORD *)(v19 + 24) = v135;
      *(_QWORD *)(v19 + 48) = v124;
      *(_QWORD *)(v19 + 40) = v128;
      v20 = *a1;
      v21 = *(unsigned int *)(*(_QWORD *)v6 + 8LL);
      if ( v12 < v21 )
        goto LABEL_9;
      if ( v12 > v21 )
      {
        if ( v12 > *(unsigned int *)(v20 + 12) )
        {
          v123 = *(_QWORD *)v6;
          sub_16CD150(v20, (const void *)(v20 + 16), v12, 8, v16, v17);
          v20 = v123;
          v21 = *(unsigned int *)(v123 + 8);
        }
        v102 = (_QWORD *)(*(_QWORD *)v20 + 8 * v21);
        for ( i = *(_QWORD *)v20 + 8 * v12; (_QWORD *)i != v102; ++v102 )
        {
          if ( v102 )
            *v102 = 0;
        }
LABEL_9:
        *(_DWORD *)(v20 + 8) = v120;
      }
      sub_1D5ABA0(*(__int64 **)(v6 + 80), v10);
      v22 = (__int64 *)sub_13CF970(a2);
      if ( (unsigned __int8)sub_1D61F00(v6, *v22, v15) )
      {
        v74 = sub_13CF970(a2);
        if ( (unsigned __int8)sub_1D61F00(v6, *(_QWORD *)(v74 + 24), v15) )
          return 1;
      }
      v25 = *(_QWORD *)(v6 + 56);
      *(_QWORD *)v25 = v147;
      *(_QWORD *)(v25 + 8) = v142;
      *(_BYTE *)(v25 + 16) = v139;
      *(_QWORD *)(v25 + 40) = v128;
      *(_QWORD *)(v25 + 24) = v135;
      *(_QWORD *)(v25 + 32) = v131;
      *(_QWORD *)(v25 + 48) = v124;
      v26 = *(_QWORD *)v6;
      v27 = *(unsigned int *)(*(_QWORD *)v6 + 8LL);
      if ( v12 < v27 )
        goto LABEL_12;
      if ( v12 > v27 )
      {
        if ( v12 > *(unsigned int *)(v26 + 12) )
        {
          sub_16CD150(*(_QWORD *)v6, (const void *)(v26 + 16), v12, 8, v23, v24);
          v27 = *(unsigned int *)(v26 + 8);
        }
        v105 = (_QWORD *)(*(_QWORD *)v26 + 8 * v27);
        for ( j = *(_QWORD *)v26 + 8 * v12; (_QWORD *)j != v105; ++v105 )
        {
          if ( v105 )
            *v105 = 0;
        }
LABEL_12:
        *(_DWORD *)(v26 + 8) = v120;
      }
LABEL_13:
      sub_1D5ABA0(*(__int64 **)(v6 + 80), v10);
      return 0;
    case 15:
    case 23:
      v42 = (__int64 *)sub_13CF970(a2);
      v43 = v42[3];
      if ( *(_BYTE *)(v43 + 16) != 13 )
        return 0;
      v44 = *(_DWORD *)(v43 + 32);
      if ( v44 > 0x40 )
        return 0;
      v45 = (__int64)(*(_QWORD *)(v43 + 24) << (64 - (unsigned __int8)v44)) >> (64 - (unsigned __int8)v44);
      if ( a3 == 23 )
        v45 = 1LL << v45;
      return sub_1D62240((__int64)a1, *v42, v45, a4);
    case 32:
      v57 = (__int64 *)(sub_13CF970(a2) + 24);
      v58 = sub_16348C0(a2) | 4;
      v126 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      if ( v126 == 1 )
      {
        v121 = 0;
        goto LABEL_102;
      }
      v117 = 0;
      v59 = 1;
      v118 = -1;
      v121 = 0;
      do
      {
        v60 = v58;
        v61 = v58 & 0xFFFFFFFFFFFFFFF8LL;
        v62 = v61;
        v63 = a1[3];
        v64 = (v60 >> 2) & 1;
        if ( v64 )
        {
          v75 = v61;
          if ( v61 )
            goto LABEL_68;
        }
        else if ( v61 )
        {
          v144 = v61;
          v65 = sub_15A9930(a1[3], v61);
          v66 = sub_13CF970(a2);
          v67 = v144;
          v68 = *(_QWORD *)(v66 + 24LL * v59);
          v69 = *(_DWORD *)(v68 + 32) <= 0x40u;
          v70 = *(_QWORD **)(v68 + 24);
          if ( !v69 )
            v70 = (_QWORD *)*v70;
          ++v59;
          v121 += *(_QWORD *)(v65 + 8LL * (unsigned int)v70 + 16);
LABEL_57:
          v62 = sub_1643D30(v67, *v57);
          goto LABEL_58;
        }
        v146 = v61;
        v82 = sub_1643D30(0, *v57);
        v61 = v146;
        v75 = v82;
LABEL_68:
        v137 = v61;
        v145 = (unsigned int)sub_15A9FE0(v63, v75);
        v76 = v145 * ((v145 + ((unsigned __int64)(sub_127FA20(v63, v75) + 7) >> 3) - 1) / v145);
        v77 = sub_13CF970(a2);
        v67 = v137;
        v78 = *(_QWORD *)(v77 + 24LL * v59);
        if ( *(_BYTE *)(v78 + 16) == 13 )
        {
          v79 = *(_DWORD *)(v78 + 32);
          v80 = *(__int64 **)(v78 + 24);
          if ( v79 > 0x40 )
            v81 = *v80;
          else
            v81 = (__int64)((_QWORD)v80 << (64 - (unsigned __int8)v79)) >> (64 - (unsigned __int8)v79);
          v121 += v76 * v81;
        }
        else if ( v76 )
        {
          if ( v118 != -1 )
            return 0;
          v118 = v59;
          v117 = v76;
        }
        ++v59;
        if ( !(_BYTE)v64 || !v137 )
          goto LABEL_57;
LABEL_58:
        v71 = *(_BYTE *)(v62 + 8);
        if ( ((v71 - 14) & 0xFD) != 0 )
        {
          v58 = 0;
          if ( v71 == 13 )
            v58 = v62;
        }
        else
        {
          v58 = *(_QWORD *)(v62 + 24) | 4LL;
        }
        v57 += 3;
      }
      while ( v126 != v59 );
      v6 = (__int64)a1;
      v5 = (_QWORD *)a2;
      v83 = a1[7];
      v84 = *(_QWORD *)(v83 + 8);
      v85 = v84 + v121;
      if ( v118 == -1 )
      {
        *(_QWORD *)(v83 + 8) = v85;
        if ( !v121
          || (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)a1[1] + 736LL))(
               a1[1],
               a1[3],
               a1[7],
               a1[4],
               *((unsigned int *)a1 + 10),
               0) )
        {
LABEL_102:
          v104 = (__int64 *)sub_13CF970((__int64)v5);
          if ( (unsigned __int8)sub_1D61F00(v6, *v104, a4 + 1) )
            return 1;
          goto LABEL_103;
        }
        if ( byte_4FC20A0 )
        {
          if ( *(_BYTE *)(a2 + 16) == 56 )
          {
            v109 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1[1] + 960LL))(a1[1]);
            if ( a4 == 0 && v121 > 0 )
            {
              if ( v109 )
              {
                v110 = sub_13CF970(a2);
                v111 = *(_QWORD *)v110;
                v112 = *(_BYTE *)(*(_QWORD *)v110 + 16LL);
                if ( v112 <= 0x17u )
                {
                  if ( v112 == 17 || v112 <= 3u )
                  {
                    v113 = *(_QWORD *)(sub_15F2060(a2) + 80);
                    if ( v113 )
                      v113 -= 24;
                    goto LABEL_126;
                  }
                }
                else if ( (unsigned int)v112 - 60 > 0xC && v112 != 56 )
                {
                  v113 = *(_QWORD *)(v111 + 40);
LABEL_126:
                  if ( *(_QWORD *)(a2 + 40) != v113 )
                  {
                    v114 = (unsigned int)*(unsigned __int8 *)(sub_157EBA0(v113) + 16) - 34;
                    if ( (unsigned int)v114 > 0x36 || (v115 = 0x40018000000001LL, !_bittest64(&v115, v114)) )
                    {
                      v116 = (__int64 *)a1[11];
                      *v116 = a2;
                      v116[1] = v121;
                    }
                  }
                }
              }
            }
          }
        }
LABEL_103:
        v28 = 0;
        *(_QWORD *)(*(_QWORD *)(v6 + 56) + 8LL) -= v121;
        return v28;
      }
      v86 = *(_BYTE *)(v83 + 16);
      v150 = *(_QWORD *)v83;
      v138 = *(_QWORD *)(v83 + 32);
      v141 = *(_QWORD *)(v83 + 24);
      v87 = *(_QWORD *)(v83 + 48);
      v88 = *(unsigned int *)(*a1 + 8);
      *(_QWORD *)(v83 + 8) = v85;
      v134 = *(_QWORD *)(v83 + 40);
      v127 = v88;
      v89 = (__int64 *)sub_13CF970(a2);
      v91 = sub_1D61F00(v6, *v89, a4 + 1);
      if ( (_BYTE)v91 )
        goto LABEL_87;
      v93 = *(_QWORD *)(v6 + 56);
      if ( *(_BYTE *)(v93 + 16) )
      {
LABEL_91:
        *(_QWORD *)(v93 + 8) = v84;
        *(_QWORD *)v93 = v150;
        *(_BYTE *)(v93 + 16) = v86;
        *(_QWORD *)(v93 + 24) = v141;
        *(_QWORD *)(v93 + 32) = v138;
        *(_QWORD *)(v93 + 40) = v134;
        *(_QWORD *)(v93 + 48) = v87;
        sub_1D61E90(*(_QWORD *)v6, v127, v150, v90, v91, v92);
        return 0;
      }
      else
      {
        *(_BYTE *)(v93 + 16) = 1;
        v94 = *(_QWORD *)(v6 + 56);
        *(_QWORD *)(v94 + 32) = *(_QWORD *)sub_13CF970(a2);
LABEL_87:
        v95 = sub_13CF970(a2);
        v96 = 24LL * v118;
        if ( (unsigned __int8)sub_1D62240(v6, *(_QWORD *)(v95 + v96), v117, a4) )
          return 1;
        v99 = *(_QWORD *)(v6 + 56);
        *(_QWORD *)v99 = v150;
        *(_BYTE *)(v99 + 16) = v86;
        *(_QWORD *)(v99 + 32) = v138;
        *(_QWORD *)(v99 + 8) = v84;
        *(_QWORD *)(v99 + 24) = v141;
        *(_QWORD *)(v99 + 40) = v134;
        *(_QWORD *)(v99 + 48) = v87;
        sub_1D61E90(*(_QWORD *)v6, v127, v141, v87, v97, v98);
        v100 = *(_QWORD *)(v6 + 56);
        if ( !*(_BYTE *)(v100 + 16) )
        {
          *(_BYTE *)(v100 + 16) = 1;
          v119 = *(_QWORD *)(v6 + 56);
          *(_QWORD *)(v119 + 32) = *(_QWORD *)sub_13CF970(a2);
          *(_QWORD *)(*(_QWORD *)(v6 + 56) + 8LL) += v121;
          v101 = sub_13CF970(a2);
          if ( !(unsigned __int8)sub_1D62240(v6, *(_QWORD *)(v101 + v96), v117, a4) )
          {
            v93 = *(_QWORD *)(v6 + 56);
            goto LABEL_91;
          }
          return 1;
        }
        return 0;
      }
    case 37:
    case 38:
      if ( *(_BYTE *)(a2 + 16) <= 0x17u )
        return 0;
      v30 = sub_1D61430((__int64 *)a2, a1[8], a1[1], a1[9]);
      if ( !v30 )
        return 0;
      v31 = a1[10];
      v10 = 0;
      v32 = *(unsigned int *)(v31 + 8);
      if ( (_DWORD)v32 )
        v10 = *(_QWORD *)(*(_QWORD *)v31 + 8 * v32 - 8);
      v33 = (_QWORD *)a1[1];
      v151[0] = 0;
      v148 = sub_1D5EF60(v33, (_QWORD *)a2);
      v34 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, unsigned int *, _QWORD, _QWORD, _QWORD))v30)(
              a2,
              *(_QWORD *)(v6 + 80),
              *(_QWORD *)(v6 + 72),
              v151,
              0,
              0,
              *(_QWORD *)(v6 + 8));
      if ( a5 )
        *a5 = 1;
      v35 = *(__int64 **)(v6 + 56);
      v36 = v35[6];
      v143 = *v35;
      v140 = v35[1];
      v136 = *((_BYTE *)v35 + 16);
      v132 = v35[3];
      v129 = v35[4];
      v125 = v35[5];
      v37 = *(_DWORD *)(*(_QWORD *)v6 + 8LL);
      v28 = sub_1D61F00(v6, v34, a4);
      if ( (_BYTE)v28 )
      {
        v39 = *(_DWORD *)(*(_QWORD *)v6 + 8LL) - v37 + (v148 ^ 1);
        if ( v39 >= v151[0] )
        {
          if ( v39 > v151[0] )
            return v28;
          v149 = v28;
          v40 = sub_1D5E480(*(_QWORD *)(v6 + 8), *(_QWORD *)(v6 + 24), v34);
          v28 = v149;
          if ( v40 )
            return v28;
        }
      }
      v41 = *(_QWORD *)(v6 + 56);
      *(_QWORD *)v41 = v143;
      *(_QWORD *)(v41 + 8) = v140;
      *(_BYTE *)(v41 + 16) = v136;
      *(_QWORD *)(v41 + 24) = v132;
      *(_QWORD *)(v41 + 40) = v125;
      *(_QWORD *)(v41 + 32) = v129;
      *(_QWORD *)(v41 + 48) = v36;
      sub_1D61E90(*(_QWORD *)v6, v37, v129, v132, v28, v38);
      goto LABEL_13;
    case 45:
      goto LABEL_39;
    case 46:
      v46 = *(_QWORD *)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
        v46 = **(_QWORD **)(v46 + 16);
      v47 = 8 * sub_15A9520(a1[3], *(_DWORD *)(v46 + 8) >> 8);
      if ( v47 == 32 )
      {
        v48 = 5;
      }
      else if ( v47 > 0x20 )
      {
        if ( v47 == 64 )
        {
          v48 = 6;
        }
        else
        {
          if ( v47 != 128 )
            goto LABEL_115;
          v48 = 7;
        }
      }
      else
      {
        if ( v47 != 8 )
        {
          v48 = 4;
          if ( v47 == 16 )
            goto LABEL_38;
LABEL_115:
          v107 = (__int64 ***)sub_13CF970(a2);
          if ( (unsigned __int8)sub_1D5D7E0(a1[3], **v107, 0) || v108 )
            return 0;
LABEL_39:
          v50 = (__int64 **)sub_13CF970(a2);
          v51 = a4;
          v52 = *v50;
          return sub_1D61F00(v6, (__int64)v52, v51);
        }
        v48 = 3;
      }
LABEL_38:
      v49 = (__int64 ***)sub_13CF970(a2);
      if ( v48 != (unsigned __int8)sub_1D5D7E0(a1[3], **v49, 0) )
        return 0;
      goto LABEL_39;
    case 47:
      v72 = (__int64 **)sub_13CF970(a2);
      v52 = *v72;
      v73 = **v72;
      if ( (*(_BYTE *)(v73 + 8) & 0xFB) != 0xB || v73 == *v5 )
        return 0;
      v51 = a4;
      return sub_1D61F00(v6, (__int64)v52, v51);
    case 48:
      v53 = **(_QWORD **)sub_13CF970(a2);
      if ( *(_BYTE *)(v53 + 8) == 16 )
        v53 = **(_QWORD **)(v53 + 16);
      v54 = *(_QWORD *)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
        v54 = **(_QWORD **)(v54 + 16);
      v55 = a1[1];
      v56 = *(__int64 (**)())(*(_QWORD *)v55 + 576LL);
      if ( v56 == sub_1D12D90
        || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))v56)(
              v55,
              *(_DWORD *)(v53 + 8) >> 8,
              *(_DWORD *)(v54 + 8) >> 8) )
      {
        return 0;
      }
      goto LABEL_39;
    default:
      return 0;
  }
}
