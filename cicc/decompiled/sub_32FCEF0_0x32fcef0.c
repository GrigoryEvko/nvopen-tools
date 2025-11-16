// Function: sub_32FCEF0
// Address: 0x32fcef0
//
__int64 __fastcall sub_32FCEF0(__int64 *a1, __int64 a2)
{
  _BOOL4 v2; // ecx
  __int64 v5; // rax
  __int64 *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned int v11; // r13d
  unsigned __int16 *v12; // rdx
  unsigned __int16 v13; // di
  unsigned int v14; // edx
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r12
  __int64 v23; // rax
  __int64 v24; // rax
  bool v25; // r10
  int v26; // eax
  int v27; // edx
  __int64 v28; // rdx
  __int64 v29; // rcx
  unsigned int v30; // ecx
  __int64 v31; // rsi
  __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // rax
  int v39; // edx
  __int64 v40; // rax
  __int16 v41; // dx
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // r8
  bool v47; // r10
  __int64 v48; // rax
  unsigned __int16 v49; // bx
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned int v52; // r12d
  bool v53; // al
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  unsigned __int16 v57; // ax
  __int64 v58; // rdx
  __int64 v59; // r12
  __int64 (__fastcall *v60)(__int64, __int64, __int64, unsigned int); // rbx
  _BOOL8 v61; // r8
  __int64 v62; // rax
  __int16 v63; // dx
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  int v67; // eax
  unsigned __int64 v68; // rdi
  int v69; // eax
  int v70; // eax
  int v71; // r9d
  int v72; // ebx
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rbx
  __int128 v76; // rax
  int v77; // r9d
  unsigned int v78; // edx
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // r12
  __int128 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  unsigned int v86; // eax
  unsigned int v87; // eax
  unsigned int v88; // eax
  unsigned int v89; // eax
  unsigned int v90; // ebx
  __int64 v91; // rax
  int v92; // r12d
  int v93; // eax
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  unsigned int v97; // eax
  __int128 v98; // [rsp-30h] [rbp-150h]
  __int64 v99; // [rsp+0h] [rbp-120h]
  __int64 v100; // [rsp+8h] [rbp-118h]
  unsigned int v101; // [rsp+10h] [rbp-110h]
  bool v102; // [rsp+17h] [rbp-109h]
  bool v103; // [rsp+17h] [rbp-109h]
  _BOOL4 v104; // [rsp+18h] [rbp-108h]
  bool v105; // [rsp+18h] [rbp-108h]
  unsigned int v106; // [rsp+20h] [rbp-100h]
  bool v107; // [rsp+20h] [rbp-100h]
  bool v108; // [rsp+20h] [rbp-100h]
  __int64 v109; // [rsp+28h] [rbp-F8h]
  __int64 v110; // [rsp+28h] [rbp-F8h]
  bool v111; // [rsp+30h] [rbp-F0h]
  bool v112; // [rsp+30h] [rbp-F0h]
  bool v113; // [rsp+30h] [rbp-F0h]
  unsigned int v114; // [rsp+30h] [rbp-F0h]
  bool v115; // [rsp+30h] [rbp-F0h]
  __int128 v116; // [rsp+30h] [rbp-F0h]
  unsigned __int16 v117; // [rsp+46h] [rbp-DAh]
  __int64 v118; // [rsp+48h] [rbp-D8h]
  __int64 v119; // [rsp+48h] [rbp-D8h]
  unsigned int v120; // [rsp+50h] [rbp-D0h]
  __int64 v121; // [rsp+50h] [rbp-D0h]
  __int64 v122; // [rsp+50h] [rbp-D0h]
  bool v123; // [rsp+50h] [rbp-D0h]
  unsigned int v124; // [rsp+50h] [rbp-D0h]
  __int128 v125; // [rsp+50h] [rbp-D0h]
  __int64 v126; // [rsp+60h] [rbp-C0h] BYREF
  int v127; // [rsp+68h] [rbp-B8h]
  unsigned int v128; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v129; // [rsp+78h] [rbp-A8h]
  __int64 v130; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v131; // [rsp+88h] [rbp-98h]
  __int64 v132; // [rsp+90h] [rbp-90h] BYREF
  __int64 v133; // [rsp+98h] [rbp-88h]
  __int64 v134; // [rsp+A0h] [rbp-80h]
  __int64 v135; // [rsp+A8h] [rbp-78h]
  unsigned __int64 v136; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v137; // [rsp+B8h] [rbp-68h]
  char v138; // [rsp+C0h] [rbp-60h]
  unsigned __int64 v139; // [rsp+D0h] [rbp-50h] BYREF
  unsigned int v140; // [rsp+D8h] [rbp-48h]
  char v141; // [rsp+DCh] [rbp-44h]
  __int64 v142; // [rsp+E0h] [rbp-40h]

  v2 = 1;
  v5 = *(_QWORD *)(a2 + 56);
  v111 = 0;
  if ( v5 && !*(_QWORD *)(v5 + 32) )
  {
    v24 = *(_QWORD *)(v5 + 16);
    v111 = *(_DWORD *)(v24 + 24) == 305;
    v2 = *(_DWORD *)(v24 + 24) != 305;
  }
  v6 = *(__int64 **)(a2 + 40);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *v6;
  v9 = v6[5];
  v10 = v6[6];
  v11 = *(_DWORD *)(v6[10] + 96);
  v12 = *(unsigned __int16 **)(a2 + 48);
  v13 = *v12;
  v126 = v7;
  v117 = v13;
  v118 = *((_QWORD *)v12 + 1);
  v120 = *((_DWORD *)v6 + 2);
  v109 = v6[5];
  v106 = *((_DWORD *)v6 + 12);
  if ( v7 )
  {
    v99 = v9;
    v100 = v10;
    v104 = v2;
    sub_B96E90((__int64)&v126, v7, 1);
    v9 = v99;
    v10 = v100;
    v2 = v104;
  }
  v14 = *((_DWORD *)a1 + 6);
  v15 = a1[1];
  v127 = *(_DWORD *)(a2 + 72);
  v140 = v14;
  v16 = *a1;
  v139 = (unsigned __int64)a1;
  v142 = v16;
  *((_QWORD *)&v98 + 1) = v10;
  *(_QWORD *)&v98 = v9;
  v141 = 0;
  v17 = sub_348D3E0(v15, v117, v118, v8, v120, v11, v98, v2, (__int64)&v139, (__int64)&v126);
  v19 = v17;
  v20 = v17;
  if ( !v17 )
  {
    v25 = v11 == 17 || v11 == 22;
    if ( !v25 )
      goto LABEL_24;
    v26 = *(_DWORD *)(v8 + 24);
    v27 = *(_DWORD *)(v109 + 24);
    if ( v26 == 186 )
    {
      if ( ((v27 - 190) & 0xFFFFFFFD) == 0 )
      {
        v32 = *(_QWORD *)(v109 + 40);
        v33 = *(_QWORD *)(v8 + 40);
        if ( *(_QWORD *)v33 == *(_QWORD *)v32 && *(_DWORD *)(v33 + 8) == *(_DWORD *)(v32 + 8) )
        {
          v30 = v120;
          v31 = v8;
          v25 = 0;
          goto LABEL_34;
        }
      }
      if ( v27 == 186 )
        goto LABEL_24;
    }
    else if ( v27 == 186 )
    {
      if ( ((v26 - 190) & 0xFFFFFFFD) == 0 )
      {
        v28 = *(_QWORD *)(v8 + 40);
        v29 = *(_QWORD *)(v109 + 40);
        if ( *(_QWORD *)v29 == *(_QWORD *)v28 && *(_DWORD *)(v29 + 8) == *(_DWORD *)(v28 + 8) )
        {
          v30 = v106;
          v25 = 0;
          v31 = v109;
          v109 = v8;
          v106 = v120;
          goto LABEL_34;
        }
      }
LABEL_30:
      if ( (unsigned int)(v26 - 193) > 1 )
        goto LABEL_24;
      v35 = *(_QWORD *)(v8 + 40);
      if ( *(_QWORD *)v35 != v109 || *(_DWORD *)(v35 + 8) != v106 )
        goto LABEL_24;
      v31 = v109;
      v30 = v106;
      v109 = v8;
      v106 = v120;
      goto LABEL_34;
    }
    if ( (unsigned int)(v27 - 193) > 1 )
      goto LABEL_30;
    v34 = *(_QWORD *)(v109 + 40);
    if ( *(_QWORD *)v34 != v8 || *(_DWORD *)(v34 + 8) != v120 )
      goto LABEL_30;
    v30 = v120;
    v31 = v8;
LABEL_34:
    v36 = *(_QWORD *)(v109 + 56);
    if ( v36 )
    {
      v37 = 1;
      do
      {
        if ( v106 == *(_DWORD *)(v36 + 8) )
        {
          if ( !v37 )
            goto LABEL_24;
          v36 = *(_QWORD *)(v36 + 32);
          if ( !v36 )
            goto LABEL_42;
          if ( *(_DWORD *)(v36 + 8) == v106 )
            goto LABEL_24;
          v37 = 0;
        }
        v36 = *(_QWORD *)(v36 + 32);
      }
      while ( v36 );
      if ( v37 == 1 )
        goto LABEL_24;
LABEL_42:
      if ( v25 )
      {
        v40 = *(_QWORD *)(v8 + 48) + 16LL * v120;
        v41 = *(_WORD *)v40;
        v42 = *(_QWORD *)(v40 + 8);
        LOWORD(v128) = v41;
        v129 = v42;
        goto LABEL_52;
      }
      v38 = *(_QWORD *)(v31 + 56);
      if ( v38 )
      {
        v39 = 1;
        do
        {
          if ( v30 == *(_DWORD *)(v38 + 8) )
          {
            if ( !v39 )
              goto LABEL_24;
            v38 = *(_QWORD *)(v38 + 32);
            if ( !v38 )
              goto LABEL_86;
            if ( v30 == *(_DWORD *)(v38 + 8) )
              goto LABEL_24;
            v39 = 0;
          }
          v38 = *(_QWORD *)(v38 + 32);
        }
        while ( v38 );
        if ( v39 == 1 )
          goto LABEL_24;
LABEL_86:
        v62 = *(_QWORD *)(v8 + 48) + 16LL * v120;
        v63 = *(_WORD *)v62;
        v129 = *(_QWORD *)(v62 + 8);
        v64 = *(_QWORD *)(v31 + 40);
        LOWORD(v128) = v63;
        v65 = sub_33DFBC0(*(_QWORD *)(v64 + 40), *(_QWORD *)(v64 + 48), 0, 0);
        v20 = 0;
        v25 = 0;
        if ( v65 )
        {
          v66 = *(_QWORD *)(v65 + 96);
          v137 = *(_DWORD *)(v66 + 32);
          if ( v137 > 0x40 )
          {
            sub_C43780((__int64)&v136, (const void **)(v66 + 24));
            v25 = 0;
            v20 = 0;
          }
          else
          {
            v136 = *(_QWORD *)(v66 + 24);
          }
          v138 = 1;
          goto LABEL_53;
        }
LABEL_52:
        v138 = 0;
LABEL_53:
        v112 = v25;
        v121 = v20;
        v43 = *(_QWORD *)(v109 + 40);
        v44 = *(_QWORD *)(v43 + 48);
        v45 = sub_33DFBC0(*(_QWORD *)(v43 + 40), v44, 0, 0);
        v46 = v121;
        v47 = v112;
        if ( v45 )
        {
          v48 = *(_QWORD *)(v45 + 96);
          v140 = *(_DWORD *)(v48 + 32);
          if ( v140 > 0x40 )
          {
            v44 = v48 + 24;
            sub_C43780((__int64)&v139, (const void **)(v48 + 24));
            v47 = v112;
            v46 = v121;
          }
          else
          {
            v139 = *(_QWORD *)(v48 + 24);
          }
          LOBYTE(v142) = 1;
        }
        else
        {
          LOBYTE(v142) = 0;
        }
        v49 = v128;
        if ( (_WORD)v128 )
        {
          if ( (unsigned __int16)(v128 - 17) > 0xD3u )
          {
LABEL_59:
            v46 = v129;
            goto LABEL_60;
          }
          v49 = word_4456580[(unsigned __int16)v128 - 1];
        }
        else
        {
          v123 = v47;
          v53 = sub_30070B0((__int64)&v128);
          v47 = v123;
          if ( !v53 )
            goto LABEL_59;
          v57 = sub_3009970((__int64)&v128, v44, v54, v55, v56);
          v47 = v123;
          v49 = v57;
          v46 = v58;
        }
LABEL_60:
        LOWORD(v132) = v49;
        v133 = v46;
        if ( v49 )
        {
          if ( v49 == 1 || (unsigned __int16)(v49 - 504) <= 7u )
            BUG();
          v122 = *(_QWORD *)&byte_444C4A0[16 * v49 - 16];
        }
        else
        {
          v113 = v47;
          v50 = sub_3007260((__int64)&v132);
          v47 = v113;
          v134 = v50;
          v135 = v51;
          LODWORD(v122) = v50;
        }
        if ( !(_BYTE)v142 )
          goto LABEL_66;
        v52 = v140;
        if ( !v47 && !v138 )
          goto LABEL_65;
        if ( v140 > 0x40 )
        {
          v115 = v47;
          v67 = sub_C444A0((__int64)&v139);
          v47 = v115;
          if ( v52 - v67 > 0x40 )
          {
            LOBYTE(v142) = 0;
            v68 = v139;
            goto LABEL_97;
          }
          v68 = v139;
          if ( (unsigned __int64)(unsigned int)v122 <= *(_QWORD *)v139 )
          {
            LOBYTE(v142) = 0;
            goto LABEL_97;
          }
          v114 = *(_DWORD *)(v109 + 24);
          if ( v47 )
          {
            v59 = a1[1];
            v60 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int))(*(_QWORD *)v59 + 448LL);
            goto LABEL_102;
          }
        }
        else
        {
          if ( (unsigned int)v122 <= v139 )
            goto LABEL_66;
          v114 = *(_DWORD *)(v109 + 24);
          if ( v47 )
          {
            v59 = a1[1];
            v60 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int))(*(_QWORD *)v59 + 448LL);
            goto LABEL_77;
          }
        }
        sub_9865C0((__int64)&v130, (__int64)&v136);
        sub_987160((__int64)&v130, (__int64)&v136, v83, v84, v85);
        v86 = v131;
        v131 = 0;
        LODWORD(v133) = v86;
        v132 = v130;
        if ( v86 > 0x40 )
          v87 = sub_C44630((__int64)&v132);
        else
          v87 = sub_39FAC40(v130);
        v102 = sub_D94970((__int64)&v139, (_QWORD *)v87);
        sub_969240(&v132);
        sub_969240(&v130);
        if ( v137 > 0x40 )
          v88 = sub_C44630((__int64)&v136);
        else
          v88 = sub_39FAC40(v136);
        v101 = v88;
        sub_9865C0((__int64)&v130, (__int64)&v139);
        sub_C46A40((__int64)&v130, v101);
        v89 = v131;
        v131 = 0;
        LODWORD(v133) = v89;
        v132 = v130;
        v103 = sub_D94970((__int64)&v132, (_QWORD *)(unsigned int)v122) && v102;
        sub_969240(&v132);
        sub_969240(&v130);
        v47 = v103;
        if ( v114 == 190 )
        {
          sub_9865C0((__int64)&v130, (__int64)&v136);
          sub_987160((__int64)&v130, (__int64)&v136, v94, v95, v96);
          v97 = v131;
          v131 = 0;
          LODWORD(v133) = v97;
          v132 = v130;
          v108 = sub_1002450((__int64)&v132) && v103;
          sub_969240(&v132);
          sub_969240(&v130);
          v47 = v108;
          goto LABEL_126;
        }
        v90 = v137;
        if ( v137 > 0x40 )
        {
          v92 = sub_C445E0((__int64)&v136);
          if ( v92 )
          {
            v93 = sub_C444A0((__int64)&v136);
            v47 = v103;
            if ( v90 != v93 + v92 )
              v47 = 0;
LABEL_126:
            v59 = a1[1];
            v60 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int))(*(_QWORD *)v59 + 448LL);
            if ( v140 <= 0x40 )
            {
LABEL_77:
              v61 = 0;
              if ( v139 )
                v61 = (v139 & (v139 - 1)) == 0;
              goto LABEL_79;
            }
LABEL_102:
            v107 = v47;
            v69 = sub_C44630((__int64)&v139);
            v47 = v107;
            v61 = v69 == 1;
LABEL_79:
            if ( v60 != sub_2FE3080 )
            {
              v105 = v47;
              v70 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _BOOL8, unsigned __int64 *, unsigned __int64 *))v60)(
                      v59,
                      v128,
                      v129,
                      v114,
                      v61,
                      &v139,
                      &v136);
              v72 = v70;
              if ( v70 != v114 && v105 )
              {
                *(_QWORD *)&v116 = sub_3406EB0(
                                     *a1,
                                     v70,
                                     (unsigned int)&v126,
                                     v128,
                                     v129,
                                     v71,
                                     *(_OWORD *)*(_QWORD *)(v109 + 40),
                                     *(_OWORD *)(*(_QWORD *)(v109 + 40) + 40LL));
                *((_QWORD *)&v116 + 1) = v73;
                if ( ((v72 - 190) & 0xFFFFFFFD) != 0 )
                {
                  v91 = *(_QWORD *)(v109 + 40);
                  v80 = *(_QWORD *)v91;
                  v79 = *(unsigned int *)(v91 + 8);
                }
                else
                {
                  LODWORD(v74) = v139;
                  if ( v72 == 190 )
                  {
                    if ( v140 > 0x40 )
                      v74 = *(_QWORD *)v139;
                    sub_109DDE0((__int64)&v132, v122, v122 - v74);
                  }
                  else
                  {
                    if ( v140 > 0x40 )
                      v74 = *(_QWORD *)v139;
                    sub_F0A5D0((__int64)&v132, v122, v122 - v74);
                  }
                  v75 = *a1;
                  *(_QWORD *)&v76 = sub_34007B0(*a1, (unsigned int)&v132, (unsigned int)&v126, v128, v129, 0, 0);
                  v110 = sub_3406EB0(
                           v75,
                           186,
                           (unsigned int)&v126,
                           v128,
                           v129,
                           v77,
                           *(_OWORD *)*(_QWORD *)(v109 + 40),
                           v76);
                  v124 = v78;
                  sub_969240(&v132);
                  v79 = v124;
                  v80 = v110;
                }
                v81 = *a1;
                *(_QWORD *)&v125 = v80;
                *((_QWORD *)&v125 + 1) = v79;
                *(_QWORD *)&v82 = sub_33ED040(*a1, v11);
                v21 = sub_340F900(v81, 208, (unsigned int)&v126, v117, v118, DWORD2(v125), v125, v116, v82);
                if ( (_BYTE)v142 )
                {
                  LOBYTE(v142) = 0;
                  sub_969240((__int64 *)&v139);
                }
                if ( v138 )
                {
                  v138 = 0;
                  sub_969240((__int64 *)&v136);
                }
                goto LABEL_7;
              }
            }
            if ( !(_BYTE)v142 )
              goto LABEL_66;
            v52 = v140;
LABEL_65:
            LOBYTE(v142) = 0;
            if ( v52 <= 0x40 )
            {
LABEL_66:
              if ( v138 )
              {
                v138 = 0;
                if ( v137 > 0x40 )
                {
                  if ( v136 )
                    j_j___libc_free_0_0(v136);
                }
              }
              goto LABEL_24;
            }
            v68 = v139;
LABEL_97:
            if ( v68 )
              j_j___libc_free_0_0(v68);
            goto LABEL_66;
          }
        }
        else if ( v136 )
        {
          if ( (v136 & (v136 + 1)) != 0 )
            v47 = 0;
          goto LABEL_126;
        }
        v47 = 0;
        goto LABEL_126;
      }
    }
LABEL_24:
    v21 = 0;
    goto LABEL_7;
  }
  if ( !v111 || *(_DWORD *)(v17 + 24) == 208 )
    goto LABEL_6;
  v119 = v17;
  v23 = sub_32FC610(a1, v17, v18, v17, v17, v18);
  v19 = v119;
  v21 = v23;
  if ( a2 == v23 )
    goto LABEL_24;
  if ( !v23 )
LABEL_6:
    v21 = v19;
LABEL_7:
  if ( v126 )
    sub_B91220((__int64)&v126, v126);
  return v21;
}
