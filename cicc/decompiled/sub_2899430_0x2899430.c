// Function: sub_2899430
// Address: 0x2899430
//
__int64 *__fastcall sub_2899430(
        __int64 a1,
        __int64 *a2,
        __int64 *a3,
        __int64 *a4,
        __int64 a5,
        char a6,
        char a7,
        int a8)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // rdx
  int v14; // edx
  unsigned __int8 v15; // al
  __int64 v16; // rbx
  unsigned int v17; // eax
  __int64 v18; // r13
  int v19; // esi
  _DWORD *v20; // r14
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // rdi
  _BYTE *v24; // r10
  __int64 (__fastcall *v25)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // r14
  __int64 v29; // rax
  _BYTE *v30; // r13
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdi
  unsigned __int8 *v34; // r14
  __int64 (__fastcall *v35)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v36; // r15
  _BYTE *v37; // rcx
  _BYTE *v38; // rax
  __int64 v39; // rax
  __int64 v41; // r15
  __int64 v42; // rbx
  unsigned int v43; // eax
  __int64 v44; // r13
  _DWORD *v45; // r14
  __int64 v46; // r15
  __int64 v47; // rax
  __int64 v48; // rdi
  _BYTE *v49; // r10
  __int64 (__fastcall *v50)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v51; // rax
  _BYTE *v52; // r12
  __int64 v53; // r14
  __int64 v54; // rax
  _BYTE *v55; // r13
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rdi
  unsigned __int8 *v59; // r14
  __int64 (__fastcall *v60)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v61; // r15
  __int64 v62; // rdx
  _BYTE *v63; // rax
  int v64; // esi
  __int64 v65; // rax
  _QWORD *v66; // rax
  unsigned int *v67; // r14
  __int64 v68; // r13
  __int64 v69; // rdx
  unsigned int v70; // esi
  _QWORD *v71; // rax
  unsigned int *v72; // r14
  __int64 v73; // r13
  __int64 v74; // rdx
  unsigned int v75; // esi
  _QWORD *v76; // rax
  __int64 v77; // r9
  unsigned int *v78; // r14
  __int64 v79; // r13
  __int64 v80; // rdx
  unsigned int v81; // esi
  _QWORD *v82; // rax
  __int64 v83; // r9
  unsigned int *v84; // r14
  __int64 v85; // r13
  __int64 v86; // rdx
  unsigned int v87; // esi
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // r12
  int v92; // esi
  _DWORD *v93; // r13
  __int64 v94; // r15
  __int64 v95; // rax
  __int64 v96; // rdi
  _BYTE *v97; // r14
  __int64 (__fastcall *v98)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  _QWORD *v99; // rax
  unsigned int *v100; // r12
  __int64 v101; // r13
  __int64 v102; // rdx
  unsigned int v103; // esi
  __int64 v104; // [rsp-10h] [rbp-1C0h]
  __int64 v105; // [rsp-10h] [rbp-1C0h]
  __int64 v106; // [rsp+0h] [rbp-1B0h]
  int v107; // [rsp+8h] [rbp-1A8h]
  char v108; // [rsp+10h] [rbp-1A0h]
  __int64 v109; // [rsp+10h] [rbp-1A0h]
  char v112; // [rsp+22h] [rbp-18Eh]
  char v113; // [rsp+23h] [rbp-18Dh]
  unsigned int v114; // [rsp+24h] [rbp-18Ch]
  unsigned int v115; // [rsp+24h] [rbp-18Ch]
  int v116; // [rsp+28h] [rbp-188h]
  unsigned int v117; // [rsp+2Ch] [rbp-184h]
  unsigned int v118; // [rsp+2Ch] [rbp-184h]
  __int64 v119; // [rsp+30h] [rbp-180h]
  __int64 v120; // [rsp+30h] [rbp-180h]
  _BYTE *v121; // [rsp+48h] [rbp-168h]
  _BYTE *v122; // [rsp+48h] [rbp-168h]
  __int64 v123; // [rsp+48h] [rbp-168h]
  __int64 v124; // [rsp+48h] [rbp-168h]
  _BYTE *v125; // [rsp+48h] [rbp-168h]
  _BYTE *v126; // [rsp+48h] [rbp-168h]
  unsigned int v127; // [rsp+54h] [rbp-15Ch]
  unsigned int v128; // [rsp+54h] [rbp-15Ch]
  int v132; // [rsp+A4h] [rbp-10Ch]
  int v133; // [rsp+A4h] [rbp-10Ch]
  _BYTE *v134; // [rsp+B0h] [rbp-100h]
  _BYTE *v135; // [rsp+B0h] [rbp-100h]
  __int64 v136; // [rsp+B8h] [rbp-F8h]
  __int64 v137; // [rsp+B8h] [rbp-F8h]
  unsigned int v138; // [rsp+CCh] [rbp-E4h] BYREF
  _QWORD v139[4]; // [rsp+D0h] [rbp-E0h] BYREF
  char v140; // [rsp+F0h] [rbp-C0h]
  char v141; // [rsp+F1h] [rbp-BFh]
  _QWORD v142[4]; // [rsp+100h] [rbp-B0h] BYREF
  __int16 v143; // [rsp+120h] [rbp-90h]
  unsigned __int64 v144; // [rsp+130h] [rbp-80h] BYREF
  __int64 v145; // [rsp+138h] [rbp-78h]
  _BYTE v146[16]; // [rsp+140h] [rbp-70h] BYREF
  __int16 v147; // [rsp+150h] [rbp-60h]

  v142[0] = sub_DFB1B0(*(_QWORD *)(a1 + 16));
  v9 = *a2;
  v142[1] = v10;
  v11 = *(_QWORD *)(*(_QWORD *)v9 + 8LL);
  v12 = *(_QWORD *)(v11 + 24);
  v144 = sub_BCAE30(v12);
  v145 = v13;
  v14 = 1;
  if ( (unsigned int)(v142[0] / v144) )
    v14 = v142[0] / v144;
  v107 = v14;
  if ( *((_BYTE *)a2 + 160) )
  {
    v114 = *(_DWORD *)(v11 + 32);
    v117 = *((_DWORD *)a2 + 2);
  }
  else
  {
    v114 = *((_DWORD *)a2 + 2);
    v117 = *(_DWORD *)(v11 + 32);
  }
  if ( *((_BYTE *)a3 + 160) )
    v116 = *((_DWORD *)a3 + 2);
  else
    v116 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)*a3 + 8LL) + 32LL);
  v15 = *(_BYTE *)(v12 + 8);
  v113 = 1;
  if ( v15 > 3u && v15 != 5 )
    v113 = (v15 & 0xFD) == 4;
  *(_DWORD *)(a5 + 104) = a8;
  v138 = 0;
  if ( !*((_BYTE *)a3 + 160) )
  {
    if ( !v114 )
    {
      v117 = 0;
      goto LABEL_46;
    }
    v109 = v114;
    v41 = a5;
    v120 = 0;
    while ( 1 )
    {
      v128 = 0;
      v112 = **(_BYTE **)(*a2 + 8 * v120);
      if ( v117 )
        break;
LABEL_123:
      if ( v109 == ++v120 )
      {
        v117 = v138;
        goto LABEL_46;
      }
    }
    v133 = v107;
    v42 = v41;
    while ( 1 )
    {
      v115 = v128 + v133;
      if ( v117 < v128 + v133 )
      {
        v43 = v133;
        do
          v43 >>= 1;
        while ( v43 + v128 > v117 );
        v115 = v43 + v128;
        v133 = v43;
      }
      if ( v116 )
        break;
      v135 = 0;
LABEL_108:
      *(_QWORD *)(*a2 + 8 * v120) = sub_28988E0(*(_QWORD *)(*a2 + 8 * v120), v128, (__int64)v135, v42);
      if ( v117 <= v115 )
      {
        v41 = v42;
        goto LABEL_123;
      }
      v128 = v115;
    }
    v137 = 0;
    v135 = 0;
    while ( 1 )
    {
      v64 = v137;
      v65 = *a4;
      if ( *((_BYTE *)a4 + 160) )
      {
        v44 = *(_QWORD *)(v65 + 8LL * v128);
      }
      else
      {
        v44 = *(_QWORD *)(v65 + 8 * v137);
        v64 = v128;
      }
      v141 = 1;
      v139[0] = "block";
      v140 = 3;
      sub_9B9680((__int64 *)&v144, v64, v133, 0);
      v45 = (_DWORD *)v144;
      v46 = (unsigned int)v145;
      v47 = sub_ACADE0(*(__int64 ***)(v44 + 8));
      v48 = *(_QWORD *)(v42 + 80);
      v49 = (_BYTE *)v47;
      v50 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v48 + 112LL);
      if ( v50 != sub_9B6630 )
        break;
      if ( *(_BYTE *)v44 <= 0x15u && *v49 <= 0x15u )
      {
        v122 = v49;
        v51 = sub_AD5CE0(v44, (__int64)v49, v45, v46, 0);
        v49 = v122;
        v52 = (_BYTE *)v51;
        goto LABEL_62;
      }
LABEL_98:
      v124 = (__int64)v49;
      v143 = 257;
      v82 = sub_BD2C40(112, unk_3F1FE60);
      v52 = v82;
      if ( v82 )
      {
        sub_B4E9E0((__int64)v82, v44, v124, v45, v46, (__int64)v142, 0, 0);
        v83 = v105;
      }
      (*(void (__fastcall **)(_QWORD, _BYTE *, _QWORD *, _QWORD, _QWORD, __int64))(**(_QWORD **)(v42 + 88) + 16LL))(
        *(_QWORD *)(v42 + 88),
        v52,
        v139,
        *(_QWORD *)(v42 + 56),
        *(_QWORD *)(v42 + 64),
        v83);
      v84 = *(unsigned int **)v42;
      v85 = *(_QWORD *)v42 + 16LL * *(unsigned int *)(v42 + 8);
      if ( *(_QWORD *)v42 != v85 )
      {
        do
        {
          v86 = *((_QWORD *)v84 + 1);
          v87 = *v84;
          v84 += 4;
          sub_B99FD0((__int64)v52, v87, v86);
        }
        while ( (unsigned int *)v85 != v84 );
      }
LABEL_63:
      if ( (_BYTE *)v144 != v146 )
        _libc_free(v144);
      v143 = 257;
      if ( a7 )
      {
        v53 = v120;
        v54 = 8 * v137;
      }
      else
      {
        v54 = 8 * v120;
        v53 = v137;
      }
      v55 = *(_BYTE **)(*a3 + v54);
      v56 = sub_BCB2E0(*(_QWORD **)(v42 + 72));
      v57 = sub_ACD640(v56, v53, 0);
      v58 = *(_QWORD *)(v42 + 80);
      v59 = (unsigned __int8 *)v57;
      v60 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v58 + 96LL);
      if ( v60 == sub_948070 )
      {
        if ( *v55 > 0x15u || *v59 > 0x15u )
        {
LABEL_88:
          v147 = 257;
          v71 = sub_BD2C40(72, 2u);
          v61 = (__int64)v71;
          if ( v71 )
            sub_B4DE80((__int64)v71, (__int64)v55, (__int64)v59, (__int64)&v144, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v42 + 88) + 16LL))(
            *(_QWORD *)(v42 + 88),
            v61,
            v142,
            *(_QWORD *)(v42 + 56),
            *(_QWORD *)(v42 + 64));
          v72 = *(unsigned int **)v42;
          v73 = *(_QWORD *)v42 + 16LL * *(unsigned int *)(v42 + 8);
          if ( *(_QWORD *)v42 != v73 )
          {
            do
            {
              v74 = *((_QWORD *)v72 + 1);
              v75 = *v72;
              v72 += 4;
              sub_B99FD0(v61, v75, v74);
            }
            while ( (unsigned int *)v73 != v72 );
          }
          goto LABEL_72;
        }
        v61 = sub_AD5840((__int64)v55, v59, 0);
      }
      else
      {
        v61 = v60(v58, v55, v59);
      }
      if ( !v61 )
        goto LABEL_88;
LABEL_72:
      v147 = 259;
      v144 = (unsigned __int64)"splat";
      v62 = sub_B37A60((unsigned int **)v42, v133, v61, (__int64 *)&v144);
      if ( !(_DWORD)v137 )
      {
        v63 = 0;
        if ( v112 != 14 )
          v63 = v135;
        v135 = v63;
      }
      ++v137;
      v135 = (_BYTE *)sub_2898E80(a1, v135, v62, v52, v113, (unsigned int **)v42, (a8 & 0x20) != 0, &v138);
      if ( v116 == v137 )
        goto LABEL_108;
    }
    v126 = v49;
    v89 = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64))v50)(v48, v44, v49, v45, v46);
    v49 = v126;
    v52 = (_BYTE *)v89;
LABEL_62:
    if ( v52 )
      goto LABEL_63;
    goto LABEL_98;
  }
  if ( v117 )
  {
    v106 = v117;
    v119 = 0;
    while ( 1 )
    {
      v127 = 0;
      v108 = **(_BYTE **)(*a2 + 8 * v119);
      if ( !v114 )
        goto LABEL_44;
      v132 = v107;
      v16 = a5;
      while ( 1 )
      {
        v17 = v132;
        v118 = v127 + v132;
        if ( v127 + v132 > v114 )
        {
          do
            v17 >>= 1;
          while ( v17 + v127 > v114 );
          v118 = v17 + v127;
          v132 = v17;
        }
        v134 = 0;
        if ( a6 )
        {
          v90 = *a2;
          if ( *((_BYTE *)a2 + 160) )
          {
            v91 = *(_QWORD *)(v90 + 8 * v119);
            v92 = v127;
          }
          else
          {
            v92 = v119;
            v91 = *(_QWORD *)(v90 + 8LL * v127);
          }
          v141 = 1;
          v139[0] = "block";
          v140 = 3;
          sub_9B9680((__int64 *)&v144, v92, v132, 0);
          v93 = (_DWORD *)v144;
          v94 = (unsigned int)v145;
          v95 = sub_ACADE0(*(__int64 ***)(v91 + 8));
          v96 = *(_QWORD *)(v16 + 80);
          v97 = (_BYTE *)v95;
          v98 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v96 + 112LL);
          if ( v98 == sub_9B6630 )
          {
            if ( *(_BYTE *)v91 <= 0x15u && *v97 <= 0x15u )
            {
              v134 = (_BYTE *)sub_AD5CE0(v91, (__int64)v97, v93, v94, 0);
              goto LABEL_118;
            }
            goto LABEL_125;
          }
          v134 = (_BYTE *)((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64))v98)(
                            v96,
                            v91,
                            v97,
                            v93,
                            v94);
LABEL_118:
          if ( !v134 )
          {
LABEL_125:
            v143 = 257;
            v99 = sub_BD2C40(112, unk_3F1FE60);
            v134 = v99;
            if ( v99 )
              sub_B4E9E0((__int64)v99, v91, (__int64)v97, v93, v94, (__int64)v142, 0, 0);
            (*(void (__fastcall **)(_QWORD, _BYTE *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v16 + 88) + 16LL))(
              *(_QWORD *)(v16 + 88),
              v134,
              v139,
              *(_QWORD *)(v16 + 56),
              *(_QWORD *)(v16 + 64));
            v100 = *(unsigned int **)v16;
            v101 = *(_QWORD *)v16 + 16LL * *(unsigned int *)(v16 + 8);
            if ( *(_QWORD *)v16 != v101 )
            {
              do
              {
                v102 = *((_QWORD *)v100 + 1);
                v103 = *v100;
                v100 += 4;
                sub_B99FD0((__int64)v134, v103, v102);
              }
              while ( (unsigned int *)v101 != v100 );
            }
          }
          if ( (_BYTE *)v144 != v146 )
            _libc_free(v144);
        }
        if ( v116 )
        {
          v136 = 0;
          while ( 1 )
          {
            v39 = *a3;
            if ( *((_BYTE *)a3 + 160) )
            {
              v18 = *(_QWORD *)(v39 + 8 * v136);
              v19 = v127;
            }
            else
            {
              v19 = v136;
              v18 = *(_QWORD *)(v39 + 8LL * v127);
            }
            v141 = 1;
            v139[0] = "block";
            v140 = 3;
            sub_9B9680((__int64 *)&v144, v19, v132, 0);
            v20 = (_DWORD *)v144;
            v21 = (unsigned int)v145;
            v22 = sub_ACADE0(*(__int64 ***)(v18 + 8));
            v23 = *(_QWORD *)(v16 + 80);
            v24 = (_BYTE *)v22;
            v25 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v23 + 112LL);
            if ( v25 != sub_9B6630 )
              break;
            if ( *(_BYTE *)v18 <= 0x15u && *v24 <= 0x15u )
            {
              v121 = v24;
              v26 = sub_AD5CE0(v18, (__int64)v24, v20, v21, 0);
              v24 = v121;
              v27 = v26;
              goto LABEL_26;
            }
LABEL_93:
            v123 = (__int64)v24;
            v143 = 257;
            v76 = sub_BD2C40(112, unk_3F1FE60);
            v27 = (__int64)v76;
            if ( v76 )
            {
              sub_B4E9E0((__int64)v76, v18, v123, v20, v21, (__int64)v142, 0, 0);
              v77 = v104;
            }
            (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD, __int64))(**(_QWORD **)(v16 + 88) + 16LL))(
              *(_QWORD *)(v16 + 88),
              v27,
              v139,
              *(_QWORD *)(v16 + 56),
              *(_QWORD *)(v16 + 64),
              v77);
            v78 = *(unsigned int **)v16;
            v79 = *(_QWORD *)v16 + 16LL * *(unsigned int *)(v16 + 8);
            if ( *(_QWORD *)v16 != v79 )
            {
              do
              {
                v80 = *((_QWORD *)v78 + 1);
                v81 = *v78;
                v78 += 4;
                sub_B99FD0(v27, v81, v80);
              }
              while ( (unsigned int *)v79 != v78 );
            }
LABEL_27:
            if ( (_BYTE *)v144 != v146 )
              _libc_free(v144);
            v143 = 257;
            if ( a7 )
            {
              v28 = v119;
              v29 = 8 * v136;
            }
            else
            {
              v29 = 8 * v119;
              v28 = v136;
            }
            v30 = *(_BYTE **)(*a4 + v29);
            v31 = sub_BCB2E0(*(_QWORD **)(v16 + 72));
            v32 = sub_ACD640(v31, v28, 0);
            v33 = *(_QWORD *)(v16 + 80);
            v34 = (unsigned __int8 *)v32;
            v35 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v33 + 96LL);
            if ( v35 == sub_948070 )
            {
              if ( *v30 > 0x15u || *v34 > 0x15u )
              {
LABEL_83:
                v147 = 257;
                v66 = sub_BD2C40(72, 2u);
                v36 = (__int64)v66;
                if ( v66 )
                  sub_B4DE80((__int64)v66, (__int64)v30, (__int64)v34, (__int64)&v144, 0, 0);
                (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v16 + 88) + 16LL))(
                  *(_QWORD *)(v16 + 88),
                  v36,
                  v142,
                  *(_QWORD *)(v16 + 56),
                  *(_QWORD *)(v16 + 64));
                v67 = *(unsigned int **)v16;
                v68 = *(_QWORD *)v16 + 16LL * *(unsigned int *)(v16 + 8);
                if ( *(_QWORD *)v16 != v68 )
                {
                  do
                  {
                    v69 = *((_QWORD *)v67 + 1);
                    v70 = *v67;
                    v67 += 4;
                    sub_B99FD0(v36, v70, v69);
                  }
                  while ( (unsigned int *)v68 != v67 );
                }
                goto LABEL_36;
              }
              v36 = sub_AD5840((__int64)v30, v34, 0);
            }
            else
            {
              v36 = v35(v33, v30, v34);
            }
            if ( !v36 )
              goto LABEL_83;
LABEL_36:
            v147 = 259;
            v144 = (unsigned __int64)"splat";
            v37 = (_BYTE *)sub_B37A60((unsigned int **)v16, v132, v36, (__int64 *)&v144);
            if ( !(_DWORD)v136 )
            {
              v38 = 0;
              if ( v108 != 14 )
                v38 = v134;
              v134 = v38;
            }
            ++v136;
            v134 = (_BYTE *)sub_2898E80(a1, v134, v27, v37, v113, (unsigned int **)v16, (a8 & 0x20) != 0, &v138);
            if ( v116 == v136 )
              goto LABEL_110;
          }
          v125 = v24;
          v88 = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64))v25)(v23, v18, v24, v20, v21);
          v24 = v125;
          v27 = v88;
LABEL_26:
          if ( v27 )
            goto LABEL_27;
          goto LABEL_93;
        }
LABEL_110:
        *(_QWORD *)(*a2 + 8 * v119) = sub_28988E0(*(_QWORD *)(*a2 + 8 * v119), v127, (__int64)v134, v16);
        if ( v118 >= v114 )
          break;
        v127 = v118;
      }
      a5 = v16;
LABEL_44:
      if ( v106 == ++v119 )
      {
        v117 = v138;
        break;
      }
    }
  }
LABEL_46:
  *((_DWORD *)a2 + 38) += v117;
  return a2;
}
