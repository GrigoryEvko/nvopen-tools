// Function: sub_2C5D9B0
// Address: 0x2c5d9b0
//
__int64 __fastcall sub_2C5D9B0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  unsigned int v5; // r11d
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r13
  _BYTE *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rcx
  bool v15; // r11
  bool v16; // zf
  unsigned __int64 v17; // r8
  _BYTE *v18; // rcx
  int v19; // r15d
  _BYTE *v20; // r9
  __int64 v21; // rax
  int v22; // edx
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // r9
  bool v26; // r11
  int v27; // edx
  int v28; // r8d
  int v29; // edx
  unsigned __int64 v30; // rax
  int v31; // edx
  unsigned int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r15
  __int64 v36; // rax
  int v37; // r15d
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r8
  unsigned __int8 *v41; // r15
  __int64 v42; // rbx
  _BYTE *v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned __int8 v46; // r11
  __int64 v47; // rcx
  __int64 v48; // rax
  int v49; // r15d
  unsigned __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rcx
  unsigned __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // r8
  int v58; // edx
  int v59; // esi
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rdx
  signed __int64 v63; // rcx
  __int64 v64; // rcx
  unsigned __int64 v65; // rdx
  bool v66; // al
  unsigned __int64 v67; // rdx
  bool v68; // sf
  bool v69; // of
  __int64 v70; // rax
  __int64 v71; // r12
  __int64 *v72; // rax
  char v73; // al
  __int64 v74; // rsi
  __int64 v75; // rdx
  __int64 v76; // r12
  __int64 i; // r13
  __int64 v78; // rax
  __int64 v79; // rcx
  __int64 v80; // rax
  __int64 v81; // rsi
  __int64 v82; // rax
  int v83; // edx
  __int64 v84; // rax
  unsigned __int64 v85; // rcx
  __int128 v86; // [rsp-2A0h] [rbp-2A0h]
  __int128 v87; // [rsp-2A0h] [rbp-2A0h]
  unsigned __int8 v88; // [rsp-280h] [rbp-280h]
  __int64 v89; // [rsp-280h] [rbp-280h]
  __int64 v90; // [rsp-280h] [rbp-280h]
  int v91; // [rsp-278h] [rbp-278h]
  int v92; // [rsp-278h] [rbp-278h]
  unsigned __int8 v93; // [rsp-26Ah] [rbp-26Ah]
  unsigned __int8 v94; // [rsp-26Ah] [rbp-26Ah]
  unsigned __int8 v95; // [rsp-269h] [rbp-269h]
  char v96; // [rsp-268h] [rbp-268h]
  bool v97; // [rsp-260h] [rbp-260h]
  unsigned __int8 v98; // [rsp-260h] [rbp-260h]
  unsigned int v99; // [rsp-260h] [rbp-260h]
  bool v100; // [rsp-248h] [rbp-248h]
  __int64 v101; // [rsp-248h] [rbp-248h]
  bool v102; // [rsp-248h] [rbp-248h]
  unsigned __int64 v103; // [rsp-248h] [rbp-248h]
  int v104; // [rsp-240h] [rbp-240h]
  bool v105; // [rsp-240h] [rbp-240h]
  int v106; // [rsp-23Ch] [rbp-23Ch]
  unsigned __int8 *v107; // [rsp-238h] [rbp-238h]
  unsigned __int8 *v108; // [rsp-230h] [rbp-230h]
  __int64 v109; // [rsp-228h] [rbp-228h]
  bool v110; // [rsp-220h] [rbp-220h]
  int v111; // [rsp-220h] [rbp-220h]
  signed __int64 v112; // [rsp-220h] [rbp-220h]
  unsigned __int8 v113; // [rsp-220h] [rbp-220h]
  unsigned __int8 v114; // [rsp-220h] [rbp-220h]
  unsigned __int8 v115; // [rsp-220h] [rbp-220h]
  unsigned __int8 v116; // [rsp-220h] [rbp-220h]
  unsigned __int64 v117; // [rsp-220h] [rbp-220h]
  __int64 v118; // [rsp-218h] [rbp-218h]
  __int64 v119; // [rsp-210h] [rbp-210h]
  _QWORD v120[2]; // [rsp-1F8h] [rbp-1F8h] BYREF
  char *v121; // [rsp-1E8h] [rbp-1E8h] BYREF
  __int64 v122; // [rsp-1E0h] [rbp-1E0h]
  _BYTE v123[32]; // [rsp-1D8h] [rbp-1D8h] BYREF
  _BYTE *v124; // [rsp-1B8h] [rbp-1B8h] BYREF
  __int64 v125; // [rsp-1B0h] [rbp-1B0h]
  _BYTE v126[48]; // [rsp-1A8h] [rbp-1A8h] BYREF
  _BYTE v127[24]; // [rsp-178h] [rbp-178h] BYREF
  __int64 *v128; // [rsp-160h] [rbp-160h]
  __int64 v129; // [rsp-150h] [rbp-150h] BYREF
  __int64 *v130; // [rsp-130h] [rbp-130h]
  __int64 v131; // [rsp-120h] [rbp-120h] BYREF
  __m128i v132; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 v133; // [rsp-C8h] [rbp-C8h]
  _QWORD *v134; // [rsp-C0h] [rbp-C0h]
  __int64 v135; // [rsp-B8h] [rbp-B8h]
  _QWORD v136[3]; // [rsp-B0h] [rbp-B0h] BYREF
  __int16 v137; // [rsp-98h] [rbp-98h]
  __int64 *v138; // [rsp-90h] [rbp-90h]
  __int64 v139; // [rsp-80h] [rbp-80h] BYREF

  if ( *a2 != 85 )
    return 0;
  v3 = (__int64)a2;
  v4 = *((_QWORD *)a2 - 4);
  if ( v4 )
  {
    if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *((_QWORD *)a2 + 10) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
    {
      v6 = a1;
      if ( sub_B5A760(*(_DWORD *)(v4 + 36)) )
      {
        v7 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v8 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
        if ( sub_9B7DA0((char *)v7, 0xFFFFFFFF, 0) && sub_9B7DA0((char *)v8, 0xFFFFFFFF, 0) )
        {
          v108 = (unsigned __int8 *)sub_9B7920((char *)v7);
          v107 = (unsigned __int8 *)sub_9B7920((char *)v8);
          if ( v107 != 0 && v108 != 0 )
          {
            v9 = (_BYTE *)sub_9B7920(*(char **)&a2[32 * (2LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
            if ( v9 )
            {
              if ( *v9 <= 0x15u && sub_AD7930(v9, 0xFFFFFFFFLL, v10, v11, v12) )
              {
                v13 = *((_QWORD *)a2 - 4);
                if ( !v13 || *(_BYTE *)v13 || *(_QWORD *)(v13 + 24) != *((_QWORD *)a2 + 10) )
                  BUG();
                v91 = *(_DWORD *)(v13 + 36);
                v95 = sub_B5B050(v91);
                if ( v95 )
                {
                  v14 = *((_QWORD *)a2 + 1);
                  v15 = 0;
                  v125 = 0xC00000000LL;
                  v16 = *(_BYTE *)(v14 + 8) == 17;
                  v109 = v14;
                  v124 = v126;
                  if ( v16 )
                  {
                    v17 = *(unsigned int *)(v14 + 32);
                    v18 = v126;
                    v19 = v17;
                    if ( v17 )
                    {
                      v20 = v126;
                      if ( v17 > 0xC )
                      {
                        v117 = v17;
                        sub_C8D5F0((__int64)&v124, v126, v17, 4u, v17, (__int64)v126);
                        v15 = v107 == 0 || v108 == 0;
                        v17 = v117;
                        v20 = &v124[4 * (unsigned int)v125];
                      }
                      v110 = v15;
                      memset(v20, 0, 4 * v17);
                      v18 = v124;
                      v15 = v110;
                      LODWORD(v125) = v19 + v125;
                      v17 = (unsigned int)v125;
                    }
                  }
                  else
                  {
                    v18 = v126;
                    v17 = 0;
                  }
                  v100 = v15;
                  v21 = sub_DFBC30(
                          *(__int64 **)(a1 + 152),
                          0,
                          v109,
                          (__int64)v18,
                          v17,
                          *(unsigned int *)(a1 + 192),
                          0,
                          0,
                          0,
                          0,
                          0);
                  v111 = v22;
                  v23 = v21;
                  v24 = sub_DFD330(*(__int64 **)(a1 + 152));
                  v26 = v100;
                  v28 = v27;
                  v29 = 1;
                  if ( v111 != 1 )
                    v29 = v28;
                  v69 = __OFADD__(v23, v24);
                  v30 = v23 + v24;
                  v106 = v29;
                  if ( v69 )
                  {
                    v30 = 0x8000000000000000LL;
                    if ( v23 > 0 )
                      v30 = 0x7FFFFFFFFFFFFFFFLL;
                  }
                  v112 = v30;
                  v31 = *a2;
                  v121 = v123;
                  v122 = 0x400000000LL;
                  if ( v31 == 40 )
                  {
                    v32 = sub_B491D0((__int64)a2);
                    v26 = v100;
                    v101 = -32 - 32LL * v32;
                  }
                  else
                  {
                    v101 = -32;
                    if ( v31 != 85 )
                    {
                      v101 = -96;
                      if ( v31 != 34 )
                        BUG();
                    }
                  }
                  if ( (a2[7] & 0x80u) != 0 )
                  {
                    v97 = v26;
                    v33 = sub_BD2BC0((__int64)a2);
                    v26 = v97;
                    v35 = v33 + v34;
                    if ( (a2[7] & 0x80u) == 0 )
                    {
                      if ( !(unsigned int)(v35 >> 4) )
                        goto LABEL_39;
                    }
                    else
                    {
                      v36 = sub_BD2BC0((__int64)a2);
                      v26 = v97;
                      if ( !(unsigned int)((v35 - v36) >> 4) )
                        goto LABEL_39;
                      if ( (a2[7] & 0x80u) != 0 )
                      {
                        v37 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
                        if ( (a2[7] & 0x80u) == 0 )
                          BUG();
                        v38 = sub_BD2BC0((__int64)a2);
                        v26 = v97;
                        v101 -= 32LL * (unsigned int)(*(_DWORD *)(v38 + v39 - 4) - v37);
                        goto LABEL_39;
                      }
                    }
                    BUG();
                  }
LABEL_39:
                  v40 = (unsigned int)v122;
                  v41 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
                  if ( v41 != &a2[v101] )
                  {
                    do
                    {
                      v42 = *(_QWORD *)(*(_QWORD *)v41 + 8LL);
                      if ( v40 + 1 > (unsigned __int64)HIDWORD(v122) )
                      {
                        v105 = v26;
                        sub_C8D5F0((__int64)&v121, v123, v40 + 1, 8u, v40, v25);
                        v40 = (unsigned int)v122;
                        v26 = v105;
                      }
                      v41 += 32;
                      *(_QWORD *)&v121[8 * v40] = v42;
                      v40 = (unsigned int)(v122 + 1);
                      LODWORD(v122) = v122 + 1;
                    }
                    while ( &a2[v101] != v41 );
                    v6 = a1;
                    v3 = (__int64)a2;
                  }
                  v102 = v26;
                  *((_QWORD *)&v86 + 1) = 1;
                  *(_QWORD *)&v86 = 0;
                  sub_DF8CB0((__int64)v127, v91, v109, v121, v40, 0, 0, v86);
                  v43 = v127;
                  v44 = sub_DFD690(*(_QWORD *)(v6 + 152), (__int64)v127);
                  v46 = v102;
                  v47 = v44;
                  v48 = 2 * v112;
                  if ( !is_mul_ok(2u, v112) )
                  {
                    v48 = 0x7FFFFFFFFFFFFFFFLL;
                    v43 = (_BYTE *)0x8000000000000000LL;
                    if ( v112 <= 0 )
                      v48 = 0x8000000000000000LL;
                  }
                  v49 = v45;
                  if ( (_DWORD)v45 != 1 )
                    v49 = v106 == 1;
                  v69 = __OFADD__(v47, v48);
                  v50 = v47 + v48;
                  if ( v69 )
                  {
                    v45 = 0x7FFFFFFFFFFFFFFFLL;
                    v50 = 0x8000000000000000LL;
                    if ( v47 > 0 )
                      v50 = 0x7FFFFFFFFFFFFFFFLL;
                  }
                  v103 = v50;
                  v51 = *(_QWORD *)(v3 - 32);
                  v98 = v46;
                  if ( !v51 || *(_BYTE *)v51 || (v52 = *(_QWORD *)(v3 + 80), *(_QWORD *)(v51 + 24) != v52) )
                    BUG();
                  v53 = sub_B5A790(*(_DWORD *)(v51 + 36), (__int64)v43, v45, v52);
                  v104 = v53;
                  v54 = HIDWORD(v53);
                  if ( BYTE4(v53) )
                  {
                    v55 = v109;
                    if ( (unsigned int)*(unsigned __int8 *)(v109 + 8) - 17 <= 1 )
                      v55 = **(_QWORD **)(v109 + 16);
                    v56 = sub_DFD800(*(_QWORD *)(v6 + 152), v53, v55, *(_DWORD *)(v6 + 192), 0, 0, 0, 0, 0, 0);
                    v96 = 0;
                    v5 = v98;
                    v99 = 0;
                    v57 = v56;
                    v59 = v58;
                  }
                  else
                  {
                    v78 = *(_QWORD *)(v3 - 32);
                    if ( !v78 || *(_BYTE *)v78 || (v79 = *(_QWORD *)(v3 + 80), *(_QWORD *)(v78 + 24) != v79) )
                      BUG();
                    v80 = sub_B5A9F0(*(_DWORD *)(v78 + 36), (__int64)v43, v54, v79);
                    v5 = v98;
                    v99 = v80;
                    v96 = BYTE4(v80);
                    if ( !BYTE4(v80) )
                      goto LABEL_85;
                    v81 = v109;
                    if ( (unsigned int)*(unsigned __int8 *)(v109 + 8) - 17 <= 1 )
                      v81 = **(_QWORD **)(v109 + 16);
                    *((_QWORD *)&v87 + 1) = 1;
                    *(_QWORD *)&v87 = 0;
                    v88 = v5;
                    sub_DF8CB0((__int64)&v132, v80, v81, v121, (unsigned int)v122, 0, 0, v87);
                    v82 = sub_DFD690(*(_QWORD *)(v6 + 152), (__int64)&v132);
                    v5 = v88;
                    v57 = v82;
                    v59 = v83;
                    if ( v138 != &v139 )
                    {
                      v93 = v88;
                      v89 = v82;
                      v92 = v83;
                      _libc_free((unsigned __int64)v138);
                      v5 = v93;
                      v57 = v89;
                      v59 = v92;
                    }
                    if ( v134 != v136 )
                    {
                      v94 = v5;
                      v90 = v57;
                      _libc_free((unsigned __int64)v134);
                      v57 = v90;
                      v5 = v94;
                    }
                  }
                  v60 = *(_QWORD *)(v8 + 16);
                  v61 = v112;
                  if ( v60 )
                    v61 = v112 * (*(_QWORD *)(v60 + 8) != 0);
                  v62 = *(_QWORD *)(v7 + 16);
                  if ( v62 )
                    v63 = v112 * (*(_QWORD *)(v62 + 8) != 0);
                  else
                    v63 = v112;
                  if ( v106 == 1 )
                    v59 = 1;
                  v69 = __OFADD__(v61, v63);
                  v64 = v61 + v63;
                  if ( v69 )
                  {
                    v64 = 0x7FFFFFFFFFFFFFFFLL;
                    if ( v61 <= 0 )
                      v64 = 0x8000000000000000LL;
                  }
                  v65 = v112 + v57;
                  if ( __OFADD__(v112, v57) )
                  {
                    v65 = 0x7FFFFFFFFFFFFFFFLL;
                    if ( v112 <= 0 )
                      v65 = 0x8000000000000000LL;
                  }
                  v66 = v95;
                  if ( v106 != 1 )
                  {
                    v106 = v59;
                    v66 = v59 != 0;
                  }
                  v69 = __OFADD__(v64, v65);
                  v67 = v64 + v65;
                  if ( v69 )
                  {
                    v67 = 0x7FFFFFFFFFFFFFFFLL;
                    if ( v64 <= 0 )
                      v67 = 0x8000000000000000LL;
                  }
                  v69 = __OFSUB__(v49, v106);
                  v68 = v49 - v106 < 0;
                  if ( v49 == v106 )
                  {
                    v69 = __OFSUB__(v103, v67);
                    v68 = (__int64)(v103 - v67) < 0;
                  }
                  if ( !v66 && v68 == v69 )
                  {
                    v70 = *(_QWORD *)(v7 + 8);
                    BYTE4(v118) = *(_BYTE *)(v70 + 8) == 18;
                    LODWORD(v118) = *(_DWORD *)(v70 + 32);
                    v71 = *(_QWORD *)(v3 + 32 * (3LL - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
                    if ( v96 )
                    {
                      v72 = (__int64 *)sub_BD5C60(v3);
                      v132.m128i_i64[0] = sub_B612D0(v72, v99);
                      v73 = sub_A73ED0(&v132, 67);
                    }
                    else
                    {
                      v73 = sub_991600(v104, v3, 0, *(_QWORD *)(v6 + 176), *(_QWORD *)(v6 + 160), 0, 1, 0);
                    }
                    if ( v73 )
                      goto LABEL_73;
                    v84 = *(_QWORD *)(v6 + 176);
                    v85 = *(_QWORD *)(v6 + 184);
                    v134 = *(_QWORD **)(v6 + 160);
                    v137 = 257;
                    v132 = (__m128i)v85;
                    v133 = 0;
                    v135 = v84;
                    v136[0] = v3;
                    v136[1] = 0;
                    v136[2] = 0;
                    v5 = sub_9B6260(v71, &v132, 0);
                    if ( (_BYTE)v5 )
                    {
LABEL_73:
                      if ( v96 )
                      {
                        v74 = v109;
                        HIDWORD(v119) = 0;
                        LOWORD(v135) = 257;
                        v120[0] = v108;
                        v120[1] = v107;
                        if ( (unsigned int)*(unsigned __int8 *)(v109 + 8) - 17 <= 1 )
                          v74 = **(_QWORD **)(v109 + 16);
                        v75 = sub_B35180(v6 + 8, v74, v99, (__int64)v120, 2u, v119, (__int64)&v132);
                      }
                      else
                      {
                        LOWORD(v135) = 257;
                        v75 = sub_2C51350((__int64 *)(v6 + 8), v104, v108, v107, v120[0], 0, (__int64)&v132, 0);
                      }
                      LOWORD(v135) = 257;
                      v76 = sub_B37620((unsigned int **)(v6 + 8), v118, v75, v132.m128i_i64);
                      sub_BD84D0(v3, v76);
                      if ( *(_BYTE *)v76 > 0x1Cu )
                      {
                        sub_BD6B90((unsigned __int8 *)v76, (unsigned __int8 *)v3);
                        for ( i = *(_QWORD *)(v76 + 16); i; i = *(_QWORD *)(i + 8) )
                          sub_F15FC0(v6 + 200, *(_QWORD *)(i + 24));
                        if ( *(_BYTE *)v76 > 0x1Cu )
                          sub_F15FC0(v6 + 200, v76);
                      }
                      if ( *(_BYTE *)v3 > 0x1Cu )
                        sub_F15FC0(v6 + 200, v3);
                      v5 = v95;
                    }
                  }
LABEL_85:
                  if ( v130 != &v131 )
                  {
                    v113 = v5;
                    _libc_free((unsigned __int64)v130);
                    v5 = v113;
                  }
                  if ( v128 != &v129 )
                  {
                    v114 = v5;
                    _libc_free((unsigned __int64)v128);
                    v5 = v114;
                  }
                  if ( v121 != v123 )
                  {
                    v115 = v5;
                    _libc_free((unsigned __int64)v121);
                    v5 = v115;
                  }
                  if ( v124 != v126 )
                  {
                    v116 = v5;
                    _libc_free((unsigned __int64)v124);
                    return v116;
                  }
                  return v5;
                }
              }
            }
          }
        }
      }
    }
  }
  return 0;
}
