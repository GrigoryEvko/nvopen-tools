// Function: sub_26F1BC0
// Address: 0x26f1bc0
//
__int64 __fastcall sub_26F1BC0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  void **v5; // r13
  __m128i *v6; // rbx
  __m128i *v7; // rdi
  unsigned __int64 *v8; // r14
  __int64 (__fastcall *v9)(__int64); // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rdi
  __int64 v12; // r15
  char *v13; // r13
  size_t v14; // rdx
  __int64 v15; // rsi
  int v16; // eax
  int v17; // edx
  __int64 v18; // r14
  unsigned int v19; // eax
  __int64 v20; // rdi
  _QWORD *v21; // rax
  const void *v22; // rax
  __int64 v23; // rdx
  size_t v24; // r8
  int v25; // eax
  char v26; // al
  _BYTE *v27; // rax
  __int64 v28; // rdx
  __m128i **v29; // r14
  __m128i *v30; // rdi
  __int64 (__fastcall *v31)(__int64); // rax
  __m128i *v32; // rdi
  __int64 v33; // rax
  size_t v34; // r8
  size_t v35; // r11
  unsigned int v36; // ecx
  unsigned int v37; // edi
  size_t *v38; // rax
  size_t v39; // r9
  size_t *v40; // rdx
  int v41; // eax
  size_t *v42; // rdi
  int v43; // ecx
  __int64 v44; // rsi
  size_t v45; // r9
  int i; // r8d
  __int64 v48; // rax
  __m128i *v49; // rax
  unsigned __int64 *v50; // rax
  unsigned __int64 *v51; // rax
  size_t v52; // r13
  _QWORD *v53; // rax
  unsigned __int64 v54; // r9
  unsigned __int64 v55; // rdx
  __m128i si128; // xmm5
  char **v57; // r13
  char **v58; // r14
  __m128i *v59; // rdi
  __int64 (__fastcall *v60)(__int64); // rax
  __int64 v61; // rax
  __int64 v62; // rdi
  char *v63; // rdi
  __int64 v64; // rsi
  __int64 v65; // rcx
  __int64 *v66; // rdx
  __int64 v67; // r10
  __m128i **v68; // r15
  __m128i **v69; // r14
  __m128i *v70; // rdi
  __int64 (__fastcall *v71)(__int64); // rax
  __m128i *v72; // rdi
  int v73; // edx
  int v74; // r11d
  unsigned int v75; // esi
  size_t v76; // r9
  int v77; // ecx
  unsigned int v78; // [rsp+4h] [rbp-2CCh]
  unsigned __int64 v79; // [rsp+8h] [rbp-2C8h]
  unsigned __int64 v80; // [rsp+10h] [rbp-2C0h]
  unsigned __int64 v81; // [rsp+18h] [rbp-2B8h]
  unsigned __int64 v82; // [rsp+20h] [rbp-2B0h]
  unsigned __int64 v83; // [rsp+28h] [rbp-2A8h]
  _QWORD *v85; // [rsp+38h] [rbp-298h]
  size_t v86; // [rsp+38h] [rbp-298h]
  size_t v87; // [rsp+38h] [rbp-298h]
  unsigned __int64 v89; // [rsp+48h] [rbp-288h]
  unsigned __int64 v93; // [rsp+68h] [rbp-268h]
  __m128i v94; // [rsp+80h] [rbp-250h]
  unsigned __int64 v95; // [rsp+90h] [rbp-240h]
  __m128i v96; // [rsp+90h] [rbp-240h]
  size_t n; // [rsp+98h] [rbp-238h]
  size_t na; // [rsp+98h] [rbp-238h]
  int nb; // [rsp+98h] [rbp-238h]
  size_t nc; // [rsp+98h] [rbp-238h]
  size_t nd; // [rsp+98h] [rbp-238h]
  __int64 v102; // [rsp+A0h] [rbp-230h] BYREF
  __int64 v103; // [rsp+A8h] [rbp-228h]
  __int64 v104; // [rsp+B0h] [rbp-220h]
  unsigned int v105; // [rsp+B8h] [rbp-218h]
  __m128i v106; // [rsp+C0h] [rbp-210h] BYREF
  __m128i v107; // [rsp+D0h] [rbp-200h] BYREF
  _QWORD *v108; // [rsp+E0h] [rbp-1F0h] BYREF
  size_t v109; // [rsp+E8h] [rbp-1E8h]
  _QWORD v110[2]; // [rsp+F0h] [rbp-1E0h] BYREF
  __m128i *v111; // [rsp+100h] [rbp-1D0h] BYREF
  size_t v112; // [rsp+108h] [rbp-1C8h]
  __m128i v113; // [rsp+110h] [rbp-1C0h] BYREF
  __m128i v114; // [rsp+120h] [rbp-1B0h] BYREF
  __m128i v115; // [rsp+130h] [rbp-1A0h]
  __m128i v116; // [rsp+140h] [rbp-190h]
  __m128i v117; // [rsp+150h] [rbp-180h]
  __m128i v118; // [rsp+160h] [rbp-170h] BYREF
  __m128i v119[3]; // [rsp+170h] [rbp-160h] BYREF
  unsigned __int64 v120[2]; // [rsp+1A0h] [rbp-130h] BYREF
  __m128i v121; // [rsp+1B0h] [rbp-120h] BYREF
  __int64 (__fastcall *v122)(__int64); // [rsp+1C0h] [rbp-110h]
  __int64 v123; // [rsp+1C8h] [rbp-108h]
  __int64 (__fastcall *v124)(__int64 *); // [rsp+1D0h] [rbp-100h]
  __int64 v125; // [rsp+1D8h] [rbp-F8h]
  __m128i *v126; // [rsp+1E0h] [rbp-F0h] BYREF
  size_t v127; // [rsp+1E8h] [rbp-E8h]
  __m128i v128; // [rsp+1F0h] [rbp-E0h] BYREF
  __int64 (__fastcall *v129)(__int64); // [rsp+200h] [rbp-D0h]
  __int64 v130; // [rsp+208h] [rbp-C8h]
  __int64 (__fastcall *v131)(_QWORD *); // [rsp+210h] [rbp-C0h]
  __int64 v132; // [rsp+218h] [rbp-B8h]
  __m128i v133; // [rsp+220h] [rbp-B0h] BYREF
  __m128i v134; // [rsp+230h] [rbp-A0h] BYREF
  __m128i v135; // [rsp+240h] [rbp-90h] BYREF
  __m128i v136; // [rsp+250h] [rbp-80h] BYREF
  unsigned __int64 v137; // [rsp+260h] [rbp-70h]
  unsigned __int64 v138; // [rsp+268h] [rbp-68h]
  unsigned __int64 v139; // [rsp+270h] [rbp-60h]
  unsigned __int64 v140; // [rsp+278h] [rbp-58h]
  unsigned __int64 v141; // [rsp+280h] [rbp-50h]
  unsigned __int64 v142; // [rsp+288h] [rbp-48h]
  unsigned __int64 v143; // [rsp+290h] [rbp-40h]
  unsigned __int64 v144; // [rsp+298h] [rbp-38h]

  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  sub_BA9680(&v133, a1);
  v80 = v137;
  v114 = _mm_load_si128(&v133);
  v82 = v138;
  v115 = _mm_load_si128(&v134);
  v89 = v139;
  v116 = _mm_load_si128(&v135);
  v95 = v140;
  v117 = _mm_load_si128(&v136);
  v79 = v141;
  v81 = v142;
  v83 = v143;
  v93 = v144;
  while ( *(_OWORD *)&v115 != __PAIR128__(v95, v89)
       || *(_OWORD *)&v114 != __PAIR128__(v82, v80)
       || *(_OWORD *)&v117 != __PAIR128__(v93, v83)
       || *(_OWORD *)&v116 != __PAIR128__(v81, v79) )
  {
    v5 = (void **)v120;
    v121.m128i_i64[1] = 0;
    v6 = (__m128i *)&v126;
    v7 = &v114;
    v121.m128i_i64[0] = (__int64)sub_C11C50;
    v8 = v120;
    v123 = 0;
    v122 = sub_C11C70;
    v125 = 0;
    v124 = sub_C11C90;
    v9 = sub_C11C30;
    if ( ((unsigned __int8)sub_C11C30 & 1) != 0 )
LABEL_4:
      v9 = *(__int64 (__fastcall **)(__int64))((char *)v9 + v7->m128i_i64[0] - 1);
    v10 = v9((__int64)v7);
    if ( !v10 )
    {
      while ( 1 )
      {
        v5 += 2;
        if ( v5 == (void **)&v126 )
          break;
        v11 = v8[3];
        v9 = (__int64 (__fastcall *)(__int64))v8[2];
        v8 = (unsigned __int64 *)v5;
        v7 = (__m128i *)((char *)&v114 + v11);
        if ( ((unsigned __int8)v9 & 1) != 0 )
          goto LABEL_4;
        v10 = v9((__int64)v7);
        if ( v10 )
          goto LABEL_9;
      }
LABEL_145:
      BUG();
    }
LABEL_9:
    v12 = v10;
    if ( (*(_BYTE *)(v10 + 32) & 0xFu) - 7 <= 1 )
    {
      v13 = (char *)sub_BD5D20(v10);
      n = v14;
      v15 = *(_QWORD *)(a5 + 8);
      v16 = *(_DWORD *)(a5 + 24);
      if ( v16 )
      {
        v17 = v16 - 1;
        v18 = 0;
        v19 = (v16 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v20 = *(_QWORD *)(v15 + 8LL * v19);
        if ( v12 != v20 )
        {
          for ( i = 1; ; ++i )
          {
            if ( v20 == -4096 )
              goto LABEL_59;
            v19 = v17 & (i + v19);
            v20 = *(_QWORD *)(v15 + 8LL * v19);
            if ( v12 == v20 )
              break;
          }
          v18 = 0;
        }
      }
      else
      {
LABEL_59:
        v48 = sub_BA8B30(a2, (__int64)v13, n);
        v18 = v48;
        if ( !v48 )
          goto LABEL_32;
        sub_AD0030(v48);
        if ( !*(_QWORD *)(v18 + 16) )
        {
          sub_B30810((_QWORD *)v18);
          goto LABEL_32;
        }
      }
      if ( v13 )
      {
        v106.m128i_i64[0] = (__int64)&v107;
        sub_26F1510(v106.m128i_i64, v13, (__int64)&v13[n]);
      }
      else
      {
        v107.m128i_i8[0] = 0;
        v106 = (__m128i)(unsigned __int64)&v107;
      }
      v126 = (__m128i *)v13;
      v128.m128i_i64[0] = a3;
      v127 = n;
      v128.m128i_i64[1] = a4;
      LOWORD(v129) = 1285;
      sub_CA0F50((__int64 *)&v108, (void **)&v126);
      v21 = (_QWORD *)sub_B326A0(v12);
      if ( !v21 )
        goto LABEL_18;
      v85 = v21;
      v22 = (const void *)sub_AA8810(v21);
      if ( n != v23 )
        goto LABEL_18;
      v24 = (size_t)v85;
      if ( n )
      {
        v25 = memcmp(v22, v13, n);
        v24 = (size_t)v85;
        if ( v25 )
          goto LABEL_18;
      }
      na = v24;
      v33 = sub_BAA410((__int64)a1, v108, v109);
      v34 = na;
      v35 = v33;
      if ( v105 )
      {
        v36 = ((unsigned int)na >> 9) ^ ((unsigned int)na >> 4);
        v37 = (v105 - 1) & v36;
        v38 = (size_t *)(v103 + 16LL * v37);
        v39 = *v38;
        if ( na != *v38 )
        {
          nb = 1;
          v40 = 0;
          while ( v39 != -4096 )
          {
            if ( v39 == -8192 && !v40 )
              v40 = v38;
            v37 = (v105 - 1) & (nb + v37);
            v38 = (size_t *)(v103 + 16LL * v37);
            v39 = *v38;
            if ( v34 == *v38 )
              goto LABEL_18;
            ++nb;
          }
          if ( !v40 )
            v40 = v38;
          ++v102;
          v41 = v104 + 1;
          if ( 4 * ((int)v104 + 1) < 3 * v105 )
          {
            if ( v105 - HIDWORD(v104) - v41 <= v105 >> 3 )
            {
              v78 = v36;
              v86 = v34;
              nc = v35;
              sub_26F19E0((__int64)&v102, v105);
              if ( !v105 )
                goto LABEL_144;
              v42 = 0;
              v34 = v86;
              v35 = nc;
              v41 = v104 + 1;
              v43 = 1;
              v44 = (v105 - 1) & v78;
              v40 = (size_t *)(v103 + 16 * v44);
              v45 = *v40;
              if ( v86 != *v40 )
              {
                while ( v45 != -4096 )
                {
                  if ( v45 == -8192 && !v42 )
                    v42 = v40;
                  LODWORD(v44) = (v105 - 1) & (v43 + v44);
                  v40 = (size_t *)(v103 + 16LL * (unsigned int)v44);
                  v45 = *v40;
                  if ( v86 == *v40 )
                    goto LABEL_119;
                  ++v43;
                }
                goto LABEL_49;
              }
            }
            goto LABEL_119;
          }
LABEL_117:
          v87 = v34;
          nd = v35;
          sub_26F19E0((__int64)&v102, 2 * v105);
          if ( !v105 )
          {
LABEL_144:
            LODWORD(v104) = v104 + 1;
            BUG();
          }
          v34 = v87;
          v35 = nd;
          v75 = (v105 - 1) & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
          v41 = v104 + 1;
          v40 = (size_t *)(v103 + 16LL * v75);
          v76 = *v40;
          if ( v87 != *v40 )
          {
            v77 = 1;
            v42 = 0;
            while ( v76 != -4096 )
            {
              if ( !v42 && v76 == -8192 )
                v42 = v40;
              v75 = (v105 - 1) & (v77 + v75);
              v40 = (size_t *)(v103 + 16LL * v75);
              v76 = *v40;
              if ( v87 == *v40 )
                goto LABEL_119;
              ++v77;
            }
LABEL_49:
            if ( v42 )
              v40 = v42;
          }
LABEL_119:
          LODWORD(v104) = v41;
          if ( *v40 != -4096 )
            --HIDWORD(v104);
          *v40 = v34;
          v40[1] = v35;
        }
LABEL_18:
        LOWORD(v129) = 260;
        v126 = (__m128i *)&v108;
        sub_BD6B50((unsigned __int8 *)v12, (const char **)&v126);
        *(_WORD *)(v12 + 32) = *(_WORD *)(v12 + 32) & 0xBFC0 | 0x4010;
        if ( v18 )
        {
          v126 = (__m128i *)&v108;
          LOWORD(v129) = 260;
          sub_BD6B50((unsigned __int8 *)v18, (const char **)&v126);
          v26 = *(_BYTE *)(v18 + 32) & 0xCF | 0x10;
          *(_BYTE *)(v18 + 32) = v26;
          if ( (v26 & 0xF) != 9 )
            *(_BYTE *)(v18 + 33) |= 0x40u;
        }
        if ( !*(_BYTE *)v12 )
        {
          v27 = (_BYTE *)v106.m128i_i64[0];
          if ( v106.m128i_i64[0] == v106.m128i_i64[0] + v106.m128i_i64[1] )
          {
LABEL_63:
            v119[0].m128i_i8[0] = 0;
            v118 = (__m128i)(unsigned __int64)v119;
            sub_2240E30((__int64)&v118, v106.m128i_i64[1] + 21);
            if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v118.m128i_i64[1]) <= 0x14 )
              goto LABEL_131;
            sub_2241490((unsigned __int64 *)&v118, ".lto_set_conditional ", 0x15u);
            sub_2241490((unsigned __int64 *)&v118, (char *)v106.m128i_i64[0], v106.m128i_u64[1]);
            if ( v118.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
              goto LABEL_131;
            v49 = (__m128i *)sub_2241490((unsigned __int64 *)&v118, ",", 1u);
            v120[0] = (unsigned __int64)&v121;
            if ( (__m128i *)v49->m128i_i64[0] == &v49[1] )
            {
              v121 = _mm_loadu_si128(v49 + 1);
            }
            else
            {
              v120[0] = v49->m128i_i64[0];
              v121.m128i_i64[0] = v49[1].m128i_i64[0];
            }
            v120[1] = v49->m128i_u64[1];
            v49->m128i_i64[0] = (__int64)v49[1].m128i_i64;
            v49->m128i_i64[1] = 0;
            v49[1].m128i_i8[0] = 0;
            v50 = sub_2241490(v120, (char *)v108, v109);
            v126 = &v128;
            if ( (unsigned __int64 *)*v50 == v50 + 2 )
            {
              v128 = _mm_loadu_si128((const __m128i *)v50 + 1);
            }
            else
            {
              v126 = (__m128i *)*v50;
              v128.m128i_i64[0] = v50[2];
            }
            v127 = v50[1];
            *v50 = (unsigned __int64)(v50 + 2);
            v50[1] = 0;
            *((_BYTE *)v50 + 16) = 0;
            if ( v127 == 0x3FFFFFFFFFFFFFFFLL )
              goto LABEL_131;
            v51 = sub_2241490((unsigned __int64 *)&v126, "\n", 1u);
            v111 = &v113;
            if ( (unsigned __int64 *)*v51 == v51 + 2 )
            {
              v113 = _mm_loadu_si128((const __m128i *)v51 + 1);
            }
            else
            {
              v111 = (__m128i *)*v51;
              v113.m128i_i64[0] = v51[2];
            }
            v112 = v51[1];
            *v51 = (unsigned __int64)(v51 + 2);
            v51[1] = 0;
            *((_BYTE *)v51 + 16) = 0;
            if ( v126 != &v128 )
              j_j___libc_free_0((unsigned __int64)v126);
            if ( (__m128i *)v120[0] != &v121 )
              j_j___libc_free_0(v120[0]);
            if ( (__m128i *)v118.m128i_i64[0] != v119 )
              j_j___libc_free_0(v118.m128i_u64[0]);
            if ( v112 > 0x3FFFFFFFFFFFFFFFLL - a1[12] )
LABEL_131:
              sub_4262D8((__int64)"basic_string::append");
            sub_2241490(a1 + 11, v111->m128i_i8, v112);
            v52 = a1[12];
            if ( v52 )
            {
              v53 = (_QWORD *)a1[11];
              if ( *((_BYTE *)v53 + v52 - 1) != 10 )
              {
                v54 = v52 + 1;
                if ( v53 == a1 + 13 )
                  v55 = 15;
                else
                  v55 = a1[13];
                if ( v54 > v55 )
                {
                  sub_2240BB0(a1 + 11, v52, 0, 0, 1u);
                  v54 = v52 + 1;
                  v53 = (_QWORD *)a1[11];
                }
                *((_BYTE *)v53 + v52) = 10;
                a1[12] = v54;
                *(_BYTE *)(a1[11] + v52 + 1) = 0;
              }
            }
            if ( v111 != &v113 )
              j_j___libc_free_0((unsigned __int64)v111);
          }
          else
          {
            while ( 1 )
            {
              if ( (unsigned __int8)((*v27 & 0xDF) - 65) > 0x19u )
              {
                if ( (unsigned __int8)(*v27 - 46) > 0x31u )
                  break;
                v28 = 0x2000000000FFDLL;
                if ( !_bittest64(&v28, (unsigned int)(unsigned __int8)*v27 - 46) )
                  break;
              }
              if ( (_BYTE *)(v106.m128i_i64[0] + v106.m128i_i64[1]) == ++v27 )
                goto LABEL_63;
            }
          }
        }
        if ( v108 != v110 )
          j_j___libc_free_0((unsigned __int64)v108);
        if ( (__m128i *)v106.m128i_i64[0] != &v107 )
          j_j___libc_free_0(v106.m128i_u64[0]);
        goto LABEL_32;
      }
      ++v102;
      goto LABEL_117;
    }
LABEL_32:
    v29 = &v126;
    v128.m128i_i64[1] = 0;
    v130 = 0;
    v30 = &v114;
    v128.m128i_i64[0] = (__int64)sub_C11BA0;
    v132 = 0;
    v129 = sub_C11BD0;
    v131 = sub_C11C00;
    v31 = sub_C11B70;
    if ( ((unsigned __int8)sub_C11B70 & 1) == 0 )
      goto LABEL_34;
LABEL_33:
    v31 = *(__int64 (__fastcall **)(__int64))((char *)v31 + v30->m128i_i64[0] - 1);
LABEL_34:
    while ( !(unsigned __int8)v31((__int64)v30) )
    {
      if ( &v133 == ++v6 )
        goto LABEL_145;
      v32 = v29[3];
      v31 = (__int64 (__fastcall *)(__int64))v29[2];
      v29 = (__m128i **)v6;
      v30 = (__m128i *)((char *)&v114 + (_QWORD)v32);
      if ( ((unsigned __int8)v31 & 1) != 0 )
        goto LABEL_33;
    }
  }
  if ( (_DWORD)v104 )
  {
    sub_BA9600(&v118, (__int64)a1);
    si128 = _mm_load_si128(v119);
    v96 = v119[1];
    v106 = _mm_load_si128(&v118);
    v107 = si128;
    v94 = v119[2];
    while ( 1 )
    {
      if ( *(_OWORD *)&v96 == *(_OWORD *)&v106 && *(_OWORD *)&v107 == *(_OWORD *)&v94 )
        return sub_C7D6A0(v103, 16LL * v105, 8);
      v57 = (char **)&v108;
      v110[1] = 0;
      v58 = (char **)&v108;
      v59 = &v106;
      v110[0] = sub_25AC5E0;
      v60 = sub_25AC5C0;
      if ( ((unsigned __int8)sub_25AC5C0 & 1) != 0 )
LABEL_92:
        v60 = *(__int64 (__fastcall **)(__int64))((char *)v60 + v59->m128i_i64[0] - 1);
      v61 = v60((__int64)v59);
      v62 = v61;
      if ( !v61 )
        break;
LABEL_97:
      v64 = *(_QWORD *)(v61 + 48);
      if ( v64 && v105 )
      {
        v65 = (v105 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
        v66 = (__int64 *)(v103 + 16 * v65);
        v67 = *v66;
        if ( v64 == *v66 )
        {
LABEL_100:
          if ( v66 != (__int64 *)(v103 + 16LL * v105) )
            sub_B2F990(v62, v66[1], (__int64)v66, v65);
        }
        else
        {
          v73 = 1;
          while ( v67 != -4096 )
          {
            v74 = v73 + 1;
            v65 = (v105 - 1) & (v73 + (_DWORD)v65);
            v66 = (__int64 *)(v103 + 16LL * (unsigned int)v65);
            v67 = *v66;
            if ( v64 == *v66 )
              goto LABEL_100;
            v73 = v74;
          }
        }
      }
      v68 = &v111;
      v113.m128i_i64[1] = 0;
      v69 = &v111;
      v70 = &v106;
      v113.m128i_i64[0] = (__int64)sub_25AC590;
      v71 = sub_25AC560;
      if ( ((unsigned __int8)sub_25AC560 & 1) == 0 )
        goto LABEL_104;
LABEL_103:
      v71 = *(__int64 (__fastcall **)(__int64))((char *)v71 + v70->m128i_i64[0] - 1);
LABEL_104:
      while ( !(unsigned __int8)v71((__int64)v70) )
      {
        v68 += 2;
        if ( &v114 == (__m128i *)v68 )
          goto LABEL_145;
        v72 = v69[3];
        v71 = (__int64 (__fastcall *)(__int64))v69[2];
        v69 = v68;
        v70 = (__m128i *)((char *)&v106 + (_QWORD)v72);
        if ( ((unsigned __int8)v71 & 1) != 0 )
          goto LABEL_103;
      }
    }
    while ( 1 )
    {
      v57 += 2;
      if ( &v111 == (__m128i **)v57 )
        goto LABEL_145;
      v63 = v58[3];
      v60 = (__int64 (__fastcall *)(__int64))v58[2];
      v58 = v57;
      v59 = (__m128i *)((char *)&v106 + (_QWORD)v63);
      if ( ((unsigned __int8)v60 & 1) != 0 )
        goto LABEL_92;
      v61 = v60((__int64)v59);
      v62 = v61;
      if ( v61 )
        goto LABEL_97;
    }
  }
  return sub_C7D6A0(v103, 16LL * v105, 8);
}
