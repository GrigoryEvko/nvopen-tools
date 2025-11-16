// Function: sub_34AB5C0
// Address: 0x34ab5c0
//
void __fastcall sub_34AB5C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rax
  char *v17; // rcx
  char *v18; // rax
  char *v19; // rdi
  signed __int64 v20; // rcx
  __int64 v21; // rdx
  char v22; // cl
  __int64 v23; // rax
  _DWORD *v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  char v29; // cl
  char *v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rcx
  __int64 v36; // r9
  char v37; // cl
  char v38; // cl
  char v39; // dl
  char v40; // dl
  __int64 v41; // rax
  int v42; // eax
  unsigned int v43; // r15d
  int v44; // edx
  __int64 v45; // rsi
  int v46; // edx
  unsigned int v47; // ecx
  int *v48; // rax
  int v49; // r8d
  __int64 v50; // rax
  __int64 (__fastcall *v51)(__int64); // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 *v55; // r14
  __int64 v56; // rax
  __int64 (__fastcall *v57)(__int64); // rax
  _QWORD *v58; // rax
  __int64 v59; // rdi
  char v60; // r15
  __int64 v61; // r14
  __int64 v62; // rbx
  __int64 v63; // r12
  __int64 v64; // r8
  int v65; // ecx
  unsigned int v66; // esi
  __int64 v67; // rax
  int v68; // r10d
  const __m128i *v69; // rsi
  __int64 v70; // r9
  __int64 v71; // rax
  char v72; // al
  __int8 v73; // dl
  __int64 v74; // rcx
  __int64 v75; // rdx
  _QWORD *v76; // rax
  _QWORD *v77; // rdx
  __int64 v78; // r11
  __int64 v79; // rbx
  int *v80; // rdi
  int *v81; // rax
  __int64 v82; // rax
  int v83; // eax
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r9
  unsigned int v92; // r14d
  unsigned __int64 v93; // rbx
  __int64 v94; // rax
  __int64 v95; // rax
  int v96; // edx
  unsigned int v97; // ecx
  __int64 v98; // r11
  __int64 v99; // rdi
  __int64 v100; // rax
  __int64 v101; // r10
  __int64 v102; // r11
  unsigned int v103; // esi
  int v104; // eax
  int v105; // edi
  int v106; // r11d
  __int64 v107; // [rsp+18h] [rbp-388h]
  int v108; // [rsp+20h] [rbp-380h]
  __int64 v109; // [rsp+28h] [rbp-378h]
  __int64 v110; // [rsp+30h] [rbp-370h]
  const __m128i *v111; // [rsp+30h] [rbp-370h]
  __int64 v113; // [rsp+38h] [rbp-368h]
  __int64 v114; // [rsp+38h] [rbp-368h]
  _QWORD *v116; // [rsp+40h] [rbp-360h]
  __int64 v117; // [rsp+40h] [rbp-360h]
  __int64 v119; // [rsp+50h] [rbp-350h] BYREF
  __int64 v120; // [rsp+58h] [rbp-348h]
  char v121; // [rsp+60h] [rbp-340h]
  __int64 v122; // [rsp+70h] [rbp-330h] BYREF
  __int64 v123; // [rsp+78h] [rbp-328h]
  char v124; // [rsp+80h] [rbp-320h]
  int v125; // [rsp+90h] [rbp-310h] BYREF
  __int64 v126; // [rsp+98h] [rbp-308h]
  __int64 v127; // [rsp+B0h] [rbp-2F0h]
  __int64 v128[2]; // [rsp+B8h] [rbp-2E8h] BYREF
  __int64 v129; // [rsp+C8h] [rbp-2D8h]
  __int64 v130; // [rsp+D0h] [rbp-2D0h]
  char *v131; // [rsp+E0h] [rbp-2C0h] BYREF
  int v132; // [rsp+E8h] [rbp-2B8h]
  char v133; // [rsp+F0h] [rbp-2B0h] BYREF
  char v134; // [rsp+100h] [rbp-2A0h]
  unsigned int *v135; // [rsp+110h] [rbp-290h] BYREF
  unsigned __int64 v136; // [rsp+118h] [rbp-288h] BYREF
  unsigned int v137; // [rsp+120h] [rbp-280h] BYREF
  _BYTE v138[64]; // [rsp+128h] [rbp-278h] BYREF
  unsigned int v139; // [rsp+168h] [rbp-238h]
  __int64 v140; // [rsp+170h] [rbp-230h]
  unsigned __int64 v141; // [rsp+178h] [rbp-228h]
  char *v142; // [rsp+180h] [rbp-220h] BYREF
  unsigned __int64 v143; // [rsp+188h] [rbp-218h]
  char v144; // [rsp+190h] [rbp-210h] BYREF
  _BYTE v145[88]; // [rsp+198h] [rbp-208h] BYREF
  __m128i v146[4]; // [rsp+1F0h] [rbp-1B0h] BYREF
  char *v147; // [rsp+230h] [rbp-170h]
  char v148; // [rsp+240h] [rbp-160h] BYREF
  char v149[224]; // [rsp+260h] [rbp-140h] BYREF
  char *v150; // [rsp+340h] [rbp-60h]
  char v151; // [rsp+350h] [rbp-50h] BYREF

  if ( (unsigned __int16)(*(_WORD *)(a2 + 68) - 14) > 1u )
    return;
  v6 = a2;
  v9 = sub_2E89170(a2);
  v10 = sub_2E891C0(a2);
  v11 = sub_B10CD0(a2 + 56);
  v15 = *(unsigned __int8 *)(v11 - 16);
  if ( (v15 & 2) != 0 )
  {
    if ( *(_DWORD *)(v11 - 24) != 2 )
    {
LABEL_4:
      v16 = 0;
      goto LABEL_5;
    }
    v41 = *(_QWORD *)(v11 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(v11 - 16) >> 6) & 0xF) != 2 )
      goto LABEL_4;
    v15 = 8LL * (((unsigned __int8)v15 >> 2) & 0xF);
    v41 = v11 - 16 - v15;
  }
  v16 = *(_QWORD *)(v41 + 8);
LABEL_5:
  v127 = v9;
  if ( v10 )
  {
    v110 = v16;
    sub_AF47B0((__int64)v128, *(unsigned __int64 **)(v10 + 16), *(unsigned __int64 **)(v10 + 24));
    v16 = v110;
  }
  else
  {
    LOBYTE(v129) = 0;
  }
  v130 = v16;
  sub_34A1780((__int64)&v131, a3, v12, v15, v13, v14, v127, v128[0], v128[1], v129, v16);
  if ( !*(_WORD *)(v9 + 20) )
    goto LABEL_9;
  if ( !v134 )
    goto LABEL_9;
  v111 = (const __m128i *)sub_349D6E0(a4, *(_QWORD *)&v131[8 * v132 - 8]);
  if ( a2 == v111[3].m128i_i64[0] )
    goto LABEL_9;
  v17 = *(char **)(a2 + 32);
  if ( *(_WORD *)(a2 + 68) == 14 )
  {
    v18 = *(char **)(a2 + 32);
    if ( *v17 )
    {
LABEL_90:
      v19 = v17 + 40;
      goto LABEL_91;
    }
  }
  else
  {
    v18 = v17 + 80;
    if ( v17[80] )
      goto LABEL_10;
  }
  v43 = *((_DWORD *)v18 + 2);
  if ( !v43 || (v44 = *(_DWORD *)(a6 + 24), v45 = *(_QWORD *)(a6 + 8), !v44) )
  {
LABEL_97:
    if ( *(_QWORD *)(a1 + 480) || !sub_2E31AB0(*(_QWORD *)(v6 + 24)) )
    {
      sub_2E891C0(v6);
      goto LABEL_99;
    }
    goto LABEL_9;
  }
  v46 = v44 - 1;
  v47 = v46 & (37 * v43);
  v48 = (int *)(v45 + 16LL * v47);
  v49 = *v48;
  if ( v43 != *v48 )
  {
    v104 = 1;
    while ( v49 != -1 )
    {
      v105 = v104 + 1;
      v47 = v46 & (v104 + v47);
      v48 = (int *)(v45 + 16LL * v47);
      v49 = *v48;
      if ( v43 == *v48 )
        goto LABEL_72;
      v104 = v105;
    }
    goto LABEL_97;
  }
LABEL_72:
  v113 = *((_QWORD *)v48 + 1);
  if ( !v113 )
    goto LABEL_97;
  v50 = sub_2E891C0(v6);
  v54 = *(_QWORD *)(v50 + 24) - *(_QWORD *)(v50 + 16);
  if ( !(unsigned int)(v54 >> 3) )
  {
    if ( *(_WORD *)(v113 + 68) == 20 )
    {
      v84 = *(_QWORD *)(v113 + 32);
      v54 = v84 + 40;
      v117 = v84 + 40;
      goto LABEL_127;
    }
    v55 = *(__int64 **)(a1 + 16);
    v56 = *v55;
    v51 = *(__int64 (__fastcall **)(__int64))(*v55 + 520);
    if ( v51 != sub_2DCA430 )
    {
      ((void (__fastcall *)(__int64 *, _QWORD, __int64))v51)(&v122, *(_QWORD *)(a1 + 16), v113);
      v54 = v123;
      v84 = v122;
      v117 = v123;
      if ( v124 )
        goto LABEL_127;
      v56 = *v55;
    }
    v57 = *(__int64 (__fastcall **)(__int64))(v56 + 528);
    if ( v57 != sub_2E77FE0 )
    {
      ((void (__fastcall *)(__int64 *, __int64 *, __int64))v57)(&v119, v55, v113);
      if ( v121 )
      {
        v54 = v120;
        v84 = v119;
        v117 = v120;
LABEL_127:
        if ( v43 == *(_DWORD *)(v84 + 8) )
        {
          sub_34A41A0((__int64)v146, a3 + 8, (_QWORD *)0x4000000100000000LL, 0x4000000200000000uLL, v52, v53);
          sub_34A19F0((__int64)&v135, (__int64)v146, v85, v86, v87, v88);
          sub_34A19F0((__int64)&v142, (__int64)v149, v89, v90, (__int64)&v142, v91);
          v107 = v6;
          while ( !sub_34A1A50((__int64)&v135, (__int64)&v142) )
          {
            v92 = v139;
            v93 = v140 + v139;
            v94 = sub_349D6E0(a4, __ROL8__(v93, 32));
            if ( *(_DWORD *)(v94 + 56) == 3 )
            {
              v98 = *(unsigned int *)(v94 + 72);
              v99 = *(_QWORD *)(v94 + 64);
              v125 = 1;
              v126 = v43;
              v100 = sub_349EDA0(v99, v99 + 32 * v98, (__int64)&v125);
              if ( v102 != v100 )
              {
                v108 = *(_DWORD *)(v117 + 8);
                if ( v108 == *(_DWORD *)(sub_34A0140(*(_QWORD *)(v101 + 48)) + 8) )
                {
                  v6 = v107;
                  if ( (_BYTE *)v143 != v145 )
                    _libc_free(v143);
                  if ( (_BYTE *)v136 != v138 )
                    _libc_free(v136);
                  sub_34A03D0((__int64)v146);
                  goto LABEL_9;
                }
              }
            }
            if ( v93 >= v141 )
            {
              v95 = v136 + 16LL * v137 - 16;
              v96 = *(_DWORD *)(v95 + 12) + 1;
              *(_DWORD *)(v95 + 12) = v96;
              v97 = v137;
              if ( v96 == *(_DWORD *)(v136 + 16LL * v137 - 8) )
              {
                v103 = v135[48];
                if ( v103 )
                {
                  sub_F03D40((__int64 *)&v136, v103);
                  v97 = v137;
                }
              }
              if ( v97 && *(_DWORD *)(v136 + 12) < *(_DWORD *)(v136 + 8) )
              {
                v139 = 0;
                v140 = *(_QWORD *)sub_34A2590((__int64)&v135);
                v141 = *(_QWORD *)sub_34A25B0((__int64)&v135);
              }
              else
              {
                v139 = -1;
                v140 = 0;
                v141 = 0;
              }
            }
            else
            {
              v139 = v92 + 1;
            }
          }
          v6 = v107;
          if ( (_BYTE *)v143 != v145 )
            _libc_free(v143);
          if ( (_BYTE *)v136 != v138 )
            _libc_free(v136);
          sub_34A03D0((__int64)v146);
        }
      }
    }
  }
  v146[0].m128i_i64[0] = v113;
  if ( !a5[5] )
    goto LABEL_99;
  v58 = sub_349D550((__int64)a5, (unsigned __int64 *)v146);
  v51 = (__int64 (__fastcall *)(__int64))v54;
  v59 = (__int64)v58;
  if ( v58 == (_QWORD *)v54 )
    goto LABEL_99;
  v114 = v6;
  v109 = a3;
  v60 = *(_BYTE *)(a4 + 56) & 1;
  v61 = v54;
  v62 = v111->m128i_i64[0];
  v63 = v111[4].m128i_i64[0];
  while ( 1 )
  {
    v70 = *(unsigned int *)(v59 + 40);
    if ( v60 )
    {
      v64 = a4 + 64;
      v65 = 3;
    }
    else
    {
      v71 = *(unsigned int *)(a4 + 72);
      v64 = *(_QWORD *)(a4 + 64);
      if ( !(_DWORD)v71 )
        goto LABEL_111;
      v65 = v71 - 1;
    }
    v66 = v65 & (37 * v70);
    v67 = v64 + 32LL * v66;
    v68 = *(_DWORD *)v67;
    if ( (_DWORD)v70 == *(_DWORD *)v67 )
      goto LABEL_82;
    v83 = 1;
    while ( v68 != -1 )
    {
      v106 = v83 + 1;
      v66 = v65 & (v83 + v66);
      v67 = v64 + 32LL * v66;
      v68 = *(_DWORD *)v67;
      if ( *(_DWORD *)v67 == (_DWORD)v70 )
        goto LABEL_82;
      v83 = v106;
    }
    if ( v60 )
    {
      v82 = 128;
      goto LABEL_112;
    }
    v71 = *(unsigned int *)(a4 + 72);
LABEL_111:
    v82 = 32 * v71;
LABEL_112:
    v67 = v64 + v82;
LABEL_82:
    v69 = (const __m128i *)(*(_QWORD *)(v67 + 8) + 384LL * *(unsigned int *)(v59 + 44));
    if ( v69->m128i_i64[0] == v62 )
    {
      v73 = v111[1].m128i_i8[8];
      if ( v73 == v69[1].m128i_i8[8]
        && (!v73 || v111->m128i_i64[1] == v69->m128i_i64[1] && v111[1].m128i_i64[0] == v69[1].m128i_i64[0]) )
      {
        v74 = v69[2].m128i_i64[0];
        if ( v111[2].m128i_i64[0] == v74 && *(_QWORD *)(v63 + 8) == *(_QWORD *)(v69[4].m128i_i64[0] + 8) )
        {
          v75 = v69[2].m128i_i64[1];
          if ( v111[2].m128i_i64[1] == v75 )
            break;
        }
      }
    }
    v59 = sub_220EEE0(v59);
    if ( v61 == v59 )
    {
      v6 = v114;
      a3 = v109;
      goto LABEL_99;
    }
  }
  a3 = v109;
  v6 = v114;
  sub_34A9810(v109, v69, v75, v74, v64, v70);
  v76 = sub_349D550((__int64)a5, (unsigned __int64 *)v146);
  v116 = v77;
  if ( v76 == *(_QWORD **)(v78 + 24) && v77 == a5 + 1 )
  {
    sub_349E8A0(a5[2]);
    v54 = (__int64)v116;
    a5[2] = 0;
    a5[5] = 0;
    a5[3] = v116;
    a5[4] = v116;
  }
  else
  {
    v54 = (__int64)a5;
    if ( v116 != v76 )
    {
      v79 = (__int64)v76;
      do
      {
        v80 = (int *)v79;
        v79 = sub_220EF30(v79);
        v81 = sub_220F330(v80, a5 + 1);
        j_j___libc_free_0((unsigned __int64)v81);
        --a5[5];
      }
      while ( v116 != (_QWORD *)v79 );
      v6 = v114;
    }
  }
LABEL_99:
  sub_34A9810(a3, v111, v54, (__int64)v51, v52, v53);
LABEL_9:
  v17 = *(char **)(v6 + 32);
  v18 = v17 + 80;
  if ( *(_WORD *)(v6 + 68) == 14 )
    goto LABEL_90;
LABEL_10:
  v19 = &v17[40 * (*(_DWORD *)(v6 + 40) & 0xFFFFFF)];
  v20 = 0xCCCCCCCCCCCCCCCDLL * ((v19 - v18) >> 3);
  v21 = v20 >> 2;
  if ( v20 >> 2 <= 0 )
  {
LABEL_43:
    if ( v20 == 2 )
      goto LABEL_48;
    if ( v20 == 3 )
    {
      v39 = *v18;
      if ( *v18 )
      {
        if ( (unsigned __int8)(v39 - 1) > 2u && v39 != 7 )
          goto LABEL_13;
      }
      else if ( !*((_DWORD *)v18 + 2) )
      {
        goto LABEL_13;
      }
      v18 += 40;
LABEL_48:
      v40 = *v18;
      if ( !*v18 )
      {
        if ( !*((_DWORD *)v18 + 2) )
          goto LABEL_13;
        goto LABEL_50;
      }
      if ( (unsigned __int8)(v40 - 1) <= 2u || v40 == 7 )
      {
LABEL_50:
        v17 = v18 + 40;
        goto LABEL_91;
      }
LABEL_13:
      if ( v19 == v18 )
        goto LABEL_29;
      goto LABEL_14;
    }
    if ( v20 != 1 )
      goto LABEL_29;
    v17 = v18;
LABEL_91:
    v72 = *v17;
    if ( *v17 )
    {
      if ( (unsigned __int8)(v72 - 1) > 2u && v72 != 7 )
        goto LABEL_94;
    }
    else if ( !*((_DWORD *)v17 + 2) )
    {
LABEL_94:
      v18 = v17;
      goto LABEL_13;
    }
LABEL_29:
    sub_349F140((__int64)v146, v6);
    sub_34A9810(a3, v146, v31, v32, v33, v34);
    sub_34A0610(&v135, (_QWORD *)a4, (__int64)v146);
    v142 = &v144;
    v143 = 0x200000000LL;
    if ( (_DWORD)v136 )
      sub_349DD80((__int64)&v142, (__int64)&v135, (unsigned int)v136, v35, (__int64)&v142, v36);
    sub_34AADC0(a3, (__int64)&v142, v146, v35, (__int64)&v142, v36);
    if ( v142 != &v144 )
      _libc_free((unsigned __int64)v142);
    if ( v135 != &v137 )
      _libc_free((unsigned __int64)v135);
    goto LABEL_16;
  }
  while ( 2 )
  {
    v22 = *v18;
    if ( *v18 )
    {
      if ( (unsigned __int8)(v22 - 1) > 2u && v22 != 7 )
        goto LABEL_13;
    }
    else if ( !*((_DWORD *)v18 + 2) )
    {
      goto LABEL_13;
    }
    v29 = v18[40];
    v30 = v18 + 40;
    if ( !v29 )
    {
      if ( !*((_DWORD *)v18 + 12) )
        goto LABEL_28;
LABEL_37:
      v37 = v18[80];
      v30 = v18 + 80;
      if ( v37 )
      {
        if ( (unsigned __int8)(v37 - 1) <= 2u )
          goto LABEL_39;
        if ( v37 != 7 )
          goto LABEL_28;
        v38 = v18[120];
        v30 = v18 + 120;
        if ( v38 )
          goto LABEL_54;
LABEL_40:
        if ( !*((_DWORD *)v18 + 32) )
          goto LABEL_28;
      }
      else
      {
        if ( !*((_DWORD *)v18 + 22) )
          goto LABEL_28;
LABEL_39:
        v38 = v18[120];
        v30 = v18 + 120;
        if ( !v38 )
          goto LABEL_40;
LABEL_54:
        if ( (unsigned __int8)(v38 - 1) > 2u && v38 != 7 )
          goto LABEL_28;
      }
      v18 += 160;
      if ( !--v21 )
      {
        v20 = 0xCCCCCCCCCCCCCCCDLL * ((v19 - v18) >> 3);
        goto LABEL_43;
      }
      continue;
    }
    break;
  }
  if ( (unsigned __int8)(v29 - 1) <= 2u || v29 == 7 )
    goto LABEL_37;
LABEL_28:
  if ( v19 == v30 )
    goto LABEL_29;
LABEL_14:
  v23 = *(_QWORD *)(v6 + 48);
  v24 = (_DWORD *)(v23 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v23 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v42 = v23 & 7;
    if ( v42 )
    {
      if ( v42 != 3 || !*v24 )
        goto LABEL_15;
    }
    else
    {
      *(_QWORD *)(v6 + 48) = v24;
    }
    BUG();
  }
LABEL_15:
  sub_349F140((__int64)v146, v6);
  sub_34A9810(a3, v146, v25, v26, v27, v28);
LABEL_16:
  if ( v150 != &v151 )
    _libc_free((unsigned __int64)v150);
  if ( v147 != &v148 )
    _libc_free((unsigned __int64)v147);
  if ( v134 )
  {
    if ( v131 != &v133 )
      _libc_free((unsigned __int64)v131);
  }
}
