// Function: sub_17E9890
// Address: 0x17e9890
//
void __fastcall sub_17E9890(__int64 *a1, __int64 a2, _QWORD *a3, __int64 a4, unsigned __int64 a5, int a6)
{
  _QWORD *v6; // r10
  _QWORD *v7; // r15
  unsigned __int64 v8; // r14
  __int64 v9; // rbx
  _QWORD *v10; // r8
  _QWORD *v11; // r13
  __int64 v12; // rcx
  _QWORD *v13; // rbx
  unsigned __int64 v14; // rax
  unsigned int *v16; // rdx
  __int64 v17; // rdx
  unsigned int *v18; // rsi
  __int64 v19; // rax
  unsigned __int64 v20; // rdi
  __int64 v21; // r11
  char *v22; // rax
  size_t v23; // rdx
  __int64 v24; // r11
  _QWORD *v25; // r8
  _QWORD *v26; // r10
  void **v27; // r9
  _QWORD *v28; // r8
  _QWORD *v29; // r10
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // rdi
  int v33; // eax
  bool v34; // al
  int v35; // eax
  bool v36; // al
  int v37; // eax
  bool v38; // al
  _QWORD *v39; // r10
  unsigned __int64 v40; // rsi
  unsigned int *v41; // rdx
  unsigned int *v42; // rcx
  __int64 v43; // rax
  __int64 v44; // r15
  unsigned __int64 v45; // r8
  unsigned __int64 v46; // rcx
  void **v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rax
  __m128i v50; // xmm1
  __m128i v51; // xmm2
  __int64 v52; // rsi
  unsigned int v53; // ecx
  const __m128i *v54; // r13
  __m128i *v55; // rbx
  __m128i *v56; // r15
  __m128i *v57; // rdi
  __int64 *v58; // r13
  __int64 v59; // rax
  __int64 v60; // rax
  __m128i *v61; // rbx
  const __m128i *v62; // r12
  const __m128i *v63; // rbx
  const __m128i *v64; // rdi
  __int64 v65; // rax
  unsigned int v66; // [rsp+8h] [rbp-4E8h]
  __int64 v67; // [rsp+10h] [rbp-4E0h]
  __int64 v68; // [rsp+10h] [rbp-4E0h]
  __int64 v69; // [rsp+10h] [rbp-4E0h]
  _QWORD *v70; // [rsp+18h] [rbp-4D8h]
  _QWORD *v71; // [rsp+18h] [rbp-4D8h]
  _QWORD *v72; // [rsp+18h] [rbp-4D8h]
  _QWORD *v73; // [rsp+18h] [rbp-4D8h]
  unsigned int v74; // [rsp+18h] [rbp-4D8h]
  unsigned int v75; // [rsp+18h] [rbp-4D8h]
  int v76; // [rsp+20h] [rbp-4D0h]
  _QWORD *v77; // [rsp+28h] [rbp-4C8h]
  _QWORD *v78; // [rsp+28h] [rbp-4C8h]
  _QWORD *v79; // [rsp+28h] [rbp-4C8h]
  _QWORD *v80; // [rsp+28h] [rbp-4C8h]
  _QWORD *v81; // [rsp+28h] [rbp-4C8h]
  _QWORD *v82; // [rsp+28h] [rbp-4C8h]
  _QWORD *v83; // [rsp+28h] [rbp-4C8h]
  _QWORD *v85; // [rsp+30h] [rbp-4C0h]
  __int64 v86; // [rsp+30h] [rbp-4C0h]
  __int64 v87; // [rsp+30h] [rbp-4C0h]
  __int64 v88; // [rsp+30h] [rbp-4C0h]
  _QWORD *v89; // [rsp+30h] [rbp-4C0h]
  _QWORD *v90; // [rsp+30h] [rbp-4C0h]
  _QWORD *v91; // [rsp+30h] [rbp-4C0h]
  _QWORD *v92; // [rsp+30h] [rbp-4C0h]
  size_t v93; // [rsp+30h] [rbp-4C0h]
  _QWORD *v94; // [rsp+30h] [rbp-4C0h]
  _QWORD *v95; // [rsp+30h] [rbp-4C0h]
  _QWORD *v96; // [rsp+38h] [rbp-4B8h]
  int v97; // [rsp+44h] [rbp-4ACh] BYREF
  __int64 v98; // [rsp+48h] [rbp-4A8h] BYREF
  __int64 *v99[2]; // [rsp+50h] [rbp-4A0h] BYREF
  __int64 *v100; // [rsp+60h] [rbp-490h]
  unsigned int *v101; // [rsp+70h] [rbp-480h] BYREF
  __int64 v102; // [rsp+78h] [rbp-478h]
  _BYTE v103[16]; // [rsp+80h] [rbp-470h] BYREF
  __m128i *v104; // [rsp+90h] [rbp-460h] BYREF
  size_t v105; // [rsp+98h] [rbp-458h]
  __m128i si128; // [rsp+A0h] [rbp-450h] BYREF
  _BYTE *v107; // [rsp+B0h] [rbp-440h] BYREF
  size_t v108; // [rsp+B8h] [rbp-438h]
  _QWORD v109[2]; // [rsp+C0h] [rbp-430h] BYREF
  void *v110; // [rsp+D0h] [rbp-420h] BYREF
  __m128i *v111; // [rsp+D8h] [rbp-418h]
  __int64 v112; // [rsp+E0h] [rbp-410h]
  __m128i *v113; // [rsp+E8h] [rbp-408h]
  int v114; // [rsp+F0h] [rbp-400h]
  _BYTE **v115; // [rsp+F8h] [rbp-3F8h]
  __m128i *v116; // [rsp+100h] [rbp-3F0h] BYREF
  size_t v117; // [rsp+108h] [rbp-3E8h]
  _BYTE v118[24]; // [rsp+110h] [rbp-3E0h] BYREF
  __m128i **v119; // [rsp+128h] [rbp-3C8h]
  __int64 v120; // [rsp+130h] [rbp-3C0h]
  __m128i v121; // [rsp+138h] [rbp-3B8h]
  __int64 v122; // [rsp+148h] [rbp-3A8h]
  char v123; // [rsp+150h] [rbp-3A0h]
  __m128i *v124; // [rsp+158h] [rbp-398h] BYREF
  __int64 v125; // [rsp+160h] [rbp-390h]
  _BYTE v126[352]; // [rsp+168h] [rbp-388h] BYREF
  char v127; // [rsp+2C8h] [rbp-228h]
  int v128; // [rsp+2CCh] [rbp-224h]
  __int64 v129; // [rsp+2D0h] [rbp-220h]
  void *v130; // [rsp+2E0h] [rbp-210h] BYREF
  void *v131; // [rsp+2E8h] [rbp-208h]
  __int64 v132; // [rsp+2F0h] [rbp-200h]
  void *dest[2]; // [rsp+2F8h] [rbp-1F8h] BYREF
  __m128i **v134; // [rsp+308h] [rbp-1E8h]
  __int64 v135; // [rsp+310h] [rbp-1E0h]
  __m128i v136; // [rsp+318h] [rbp-1D8h] BYREF
  __int64 v137; // [rsp+328h] [rbp-1C8h]
  char v138; // [rsp+330h] [rbp-1C0h]
  const __m128i *v139; // [rsp+338h] [rbp-1B8h]
  unsigned int v140; // [rsp+340h] [rbp-1B0h]
  char v141; // [rsp+348h] [rbp-1A8h] BYREF
  char v142; // [rsp+4A8h] [rbp-48h]
  int v143; // [rsp+4ACh] [rbp-44h]
  __int64 v144; // [rsp+4B0h] [rbp-40h]

  v6 = a3;
  v7 = a3;
  v8 = 1;
  v9 = a2;
  v98 = *a1;
  if ( a5 > 0xFFFFFFFE )
    v8 = a5 / 0xFFFFFFFF + 1;
  v10 = &a3[a4];
  v101 = (unsigned int *)v103;
  v102 = 0x400000000LL;
  if ( v10 == a3 )
  {
    v17 = 0;
    v18 = (unsigned int *)v103;
  }
  else
  {
    v11 = a3 + 1;
    v12 = 0;
    v13 = v10;
    v14 = *a3 / v8;
    v16 = (unsigned int *)v103;
    while ( 1 )
    {
      v16[v12] = v14;
      v12 = (unsigned int)(v102 + 1);
      LODWORD(v102) = v102 + 1;
      if ( v13 == v11 )
        break;
      v14 = *v11 / v8;
      if ( HIDWORD(v102) <= (unsigned int)v12 )
      {
        v76 = *v11 / v8;
        sub_16CD150((__int64)&v101, v103, 0, 4, (int)v10, a6);
        v12 = (unsigned int)v102;
        LODWORD(v14) = v76;
      }
      v16 = v101;
      ++v11;
    }
    v10 = v13;
    v6 = a3;
    v9 = a2;
    v17 = (unsigned int)v12;
    v7 = a3;
    v18 = v101;
  }
  v85 = v6;
  v96 = v10;
  v19 = sub_161BD30(&v98, v18, v17);
  sub_1625C10(v9, 2, v19);
  if ( !byte_4FA4EC0 )
  {
LABEL_11:
    v20 = (unsigned __int64)v101;
    if ( v101 == (unsigned int *)v103 )
      return;
    goto LABEL_12;
  }
  if ( *(_BYTE *)(v9 + 16) != 26
    || (*(_DWORD *)(v9 + 20) & 0xFFFFFFF) != 3
    || (v21 = *(_QWORD *)(v9 - 72), *(_BYTE *)(v21 + 16) != 75) )
  {
    si128.m128i_i8[0] = 0;
    v104 = &si128;
    v105 = 0;
    goto LABEL_18;
  }
  v117 = 0;
  v116 = (__m128i *)v118;
  v134 = &v116;
  v118[0] = 0;
  LODWORD(dest[1]) = 1;
  dest[0] = 0;
  v132 = 0;
  v131 = 0;
  v130 = &unk_49EFBE0;
  v70 = v85;
  v86 = v21;
  v22 = sub_15FF290(*(_WORD *)(v21 + 18) & 0x7FFF);
  v24 = v86;
  v25 = v96;
  v26 = v70;
  if ( v132 - (unsigned __int64)dest[0] < v23 )
  {
    v65 = sub_16E7EE0((__int64)&v130, v22, v23);
    v25 = v96;
    v26 = v70;
    v24 = v86;
    v27 = (void **)v65;
  }
  else
  {
    v27 = &v130;
    if ( v23 )
    {
      v69 = v86;
      v93 = v23;
      memcpy(dest[0], v22, v23);
      v27 = &v130;
      dest[0] = (char *)dest[0] + v93;
      v24 = v69;
      v26 = v70;
      v25 = v96;
    }
  }
  v71 = v26;
  v77 = v25;
  v87 = v24;
  sub_1263B40((__int64)v27, "_");
  sub_154E060(**(_QWORD **)(v87 - 48), (__int64)&v130, 1, 0);
  v28 = v77;
  v29 = v71;
  v30 = *(_QWORD *)(v87 - 24);
  if ( *(_BYTE *)(v30 + 16) == 13 )
  {
    v31 = *(_DWORD *)(v30 + 32);
    v32 = v30 + 24;
    if ( v31 <= 0x40 )
    {
      v34 = *(_QWORD *)(v30 + 24) == 0;
    }
    else
    {
      v66 = *(_DWORD *)(v30 + 32);
      v67 = *(_QWORD *)(v87 - 24);
      v88 = v30 + 24;
      v33 = sub_16A57B0(v32);
      v31 = v66;
      v32 = v88;
      v28 = v77;
      v29 = v71;
      v30 = v67;
      v34 = v66 == v33;
    }
    if ( v34 )
    {
      v82 = v29;
      v94 = v28;
      sub_1263B40((__int64)&v130, "_Zero");
      v28 = v94;
      v29 = v82;
    }
    else
    {
      if ( v31 <= 0x40 )
      {
        v36 = *(_QWORD *)(v30 + 24) == 1;
      }
      else
      {
        v68 = v30;
        v72 = v29;
        v78 = v28;
        v35 = sub_16A57B0(v32);
        v28 = v78;
        v29 = v72;
        v30 = v68;
        v36 = v31 - 1 == v35;
      }
      if ( v36 )
      {
        v83 = v29;
        v95 = v28;
        sub_1263B40((__int64)&v130, "_One");
        v28 = v95;
        v29 = v83;
      }
      else
      {
        if ( v31 <= 0x40 )
        {
          v38 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v31) == *(_QWORD *)(v30 + 24);
        }
        else
        {
          v79 = v29;
          v89 = v28;
          v37 = sub_16A58F0(v32);
          v28 = v89;
          v29 = v79;
          v38 = v31 == v37;
        }
        v80 = v29;
        v90 = v28;
        if ( v38 )
        {
          sub_1263B40((__int64)&v130, "_MinusOne");
          v28 = v90;
          v29 = v80;
        }
        else
        {
          sub_1263B40((__int64)&v130, "_Const");
          v29 = v80;
          v28 = v90;
        }
      }
    }
  }
  if ( dest[0] != v131 )
  {
    v81 = v29;
    v91 = v28;
    sub_16E7BA0((__int64 *)&v130);
    v29 = v81;
    v28 = v91;
  }
  v104 = &si128;
  if ( v116 == (__m128i *)v118 )
  {
    si128 = _mm_load_si128((const __m128i *)v118);
  }
  else
  {
    v104 = v116;
    si128.m128i_i64[0] = *(_QWORD *)v118;
  }
  v73 = v29;
  v92 = v28;
  v105 = v117;
  v117 = 0;
  v116 = (__m128i *)v118;
  v118[0] = 0;
  sub_16E7BC0((__int64 *)&v130);
  sub_2240A30(&v116);
  v39 = v73;
  if ( v105 )
  {
    v40 = 0;
    v41 = &v101[(unsigned int)v102];
    v42 = v101;
    if ( v41 == v101 )
    {
      if ( v92 == v7 )
      {
        LODWORD(v45) = 0;
        v44 = 0;
        v46 = 1;
LABEL_47:
        sub_16AF710(&v97, *v101 / v46, v45);
        v115 = &v107;
        v107 = v109;
        v110 = &unk_49EFBE0;
        v108 = 0;
        LOBYTE(v109[0]) = 0;
        v114 = 1;
        v113 = 0;
        v112 = 0;
        v111 = 0;
        LODWORD(v130) = v97;
        sub_16AF620((int *)&v130, (__int64)&v110);
        if ( (unsigned __int64)(v112 - (_QWORD)v113) <= 0xF )
        {
          v47 = (void **)sub_16E7EE0((__int64)&v110, " (total count : ", 0x10u);
        }
        else
        {
          v47 = &v110;
          *v113++ = _mm_load_si128((const __m128i *)&xmmword_42B6810);
        }
        v48 = sub_16E7A90((__int64)v47, v44);
        sub_1263B40(v48, ")");
        if ( v113 != v111 )
          sub_16E7BA0((__int64 *)&v110);
        sub_143A950(v99, *(__int64 **)(*(_QWORD *)(v9 + 40) + 56LL));
        v49 = sub_15E0530((__int64)v99[0]);
        if ( !sub_1602790(v49) )
        {
          v59 = sub_15E0530((__int64)v99[0]);
          v60 = sub_16033E0(v59);
          if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v60 + 48LL))(v60) )
            goto LABEL_68;
        }
        sub_15CA3B0((__int64)&v130, (__int64)"pgo-instrumentation", (__int64)"pgo-instrumentation", 19, v9);
        sub_15CAB20((__int64)&v130, v104, v105);
        sub_15CAB20((__int64)&v130, " is true with probability : ", 0x1Cu);
        sub_15CAB20((__int64)&v130, v107, v108);
        v50 = _mm_loadu_si128((const __m128i *)dest);
        v51 = _mm_loadu_si128(&v136);
        LODWORD(v117) = (_DWORD)v131;
        *(__m128i *)&v118[8] = v50;
        BYTE4(v117) = BYTE4(v131);
        v121 = v51;
        *(_QWORD *)v118 = v132;
        v119 = v134;
        v116 = (__m128i *)&unk_49ECF68;
        v120 = v135;
        v123 = v138;
        if ( v138 )
          v122 = v137;
        v52 = v140;
        v124 = (__m128i *)v126;
        v53 = v140;
        v125 = 0x400000000LL;
        if ( v140 )
        {
          if ( v140 > 4uLL )
          {
            v75 = v140;
            sub_14B3F20((__int64)&v124, v140);
            v61 = v124;
            v52 = v140;
            v53 = v75;
          }
          else
          {
            v61 = (__m128i *)v126;
          }
          v54 = (const __m128i *)((char *)v139 + 88 * v52);
          if ( v139 != v54 )
          {
            v74 = v53;
            v62 = v139;
            do
            {
              if ( v61 )
              {
                v61->m128i_i64[0] = (__int64)v61[1].m128i_i64;
                sub_17E2330(v61->m128i_i64, v62->m128i_i64[0], v62->m128i_i64[0] + v62->m128i_i64[1]);
                v61[2].m128i_i64[0] = (__int64)v61[3].m128i_i64;
                sub_17E2330(v61[2].m128i_i64, (_BYTE *)v62[2].m128i_i64[0], v62[2].m128i_i64[0] + v62[2].m128i_i64[1]);
                v61[4] = _mm_loadu_si128(v62 + 4);
                v61[5].m128i_i64[0] = v62[5].m128i_i64[0];
              }
              v62 = (const __m128i *)((char *)v62 + 88);
              v61 = (__m128i *)((char *)v61 + 88);
            }
            while ( v54 != v62 );
            v63 = v139;
            LODWORD(v125) = v74;
            v54 = (const __m128i *)((char *)v139 + 88 * v140);
            v127 = v142;
            v128 = v143;
            v129 = v144;
            v116 = (__m128i *)&unk_49ECF98;
            v130 = &unk_49ECF68;
            if ( v139 != v54 )
            {
              do
              {
                v54 = (const __m128i *)((char *)v54 - 88);
                v64 = (const __m128i *)v54[2].m128i_i64[0];
                if ( v64 != &v54[3] )
                  j_j___libc_free_0(v64, v54[3].m128i_i64[0] + 1);
                if ( (const __m128i *)v54->m128i_i64[0] != &v54[1] )
                  j_j___libc_free_0(v54->m128i_i64[0], v54[1].m128i_i64[0] + 1);
              }
              while ( v63 != v54 );
              v54 = v139;
            }
            goto LABEL_57;
          }
          LODWORD(v125) = v53;
        }
        else
        {
          v54 = v139;
        }
        v127 = v142;
        v128 = v143;
        v129 = v144;
        v116 = (__m128i *)&unk_49ECF98;
LABEL_57:
        if ( v54 != (const __m128i *)&v141 )
          _libc_free((unsigned __int64)v54);
        sub_143AA50(v99, (__int64)&v116);
        v55 = v124;
        v116 = (__m128i *)&unk_49ECF68;
        v56 = (__m128i *)((char *)v124 + 88 * (unsigned int)v125);
        if ( v124 != v56 )
        {
          do
          {
            v56 = (__m128i *)((char *)v56 - 88);
            v57 = (__m128i *)v56[2].m128i_i64[0];
            if ( v57 != &v56[3] )
              j_j___libc_free_0(v57, v56[3].m128i_i64[0] + 1);
            if ( (__m128i *)v56->m128i_i64[0] != &v56[1] )
              j_j___libc_free_0(v56->m128i_i64[0], v56[1].m128i_i64[0] + 1);
          }
          while ( v55 != v56 );
          v56 = v124;
        }
        if ( v56 != (__m128i *)v126 )
          _libc_free((unsigned __int64)v56);
LABEL_68:
        v58 = v100;
        if ( v100 )
        {
          sub_1368A00(v100);
          j_j___libc_free_0(v58, 8);
        }
        sub_16E7BC0((__int64 *)&v110);
        if ( v107 != (_BYTE *)v109 )
          j_j___libc_free_0(v107, v109[0] + 1LL);
        if ( v104 != &si128 )
          j_j___libc_free_0(v104, si128.m128i_i64[0] + 1);
        goto LABEL_11;
      }
    }
    else
    {
      do
      {
        v43 = *v42++;
        v40 += v43;
      }
      while ( v41 != v42 );
      if ( v92 == v7 )
      {
        v44 = 0;
LABEL_45:
        if ( v40 > 0xFFFFFFFE )
        {
          v46 = v40 / 0xFFFFFFFF + 1;
          v45 = v40 / v46;
        }
        else
        {
          LODWORD(v45) = v40;
          v46 = 1;
        }
        goto LABEL_47;
      }
    }
    v44 = 0;
    do
      v44 += *v39++;
    while ( v92 != v39 );
    goto LABEL_45;
  }
LABEL_18:
  sub_2240A30(&v104);
  v20 = (unsigned __int64)v101;
  if ( v101 != (unsigned int *)v103 )
LABEL_12:
    _libc_free(v20);
}
