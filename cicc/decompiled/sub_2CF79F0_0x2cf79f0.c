// Function: sub_2CF79F0
// Address: 0x2cf79f0
//
_BOOL8 __fastcall sub_2CF79F0(__int64 a1, __int64 a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // rbx
  _BYTE *v10; // rsi
  _BYTE *v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  char v14; // r15
  unsigned __int64 *v15; // rdx
  __int64 v16; // rbx
  unsigned int v17; // eax
  unsigned __int64 v18; // r13
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int8 v21; // r12
  __int64 i; // r13
  unsigned __int64 v23; // r14
  char v24; // al
  unsigned __int64 *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // r8
  unsigned __int8 **v28; // r13
  _BYTE *v29; // rsi
  bool v30; // r14
  __int64 v31; // rax
  _BYTE *v32; // rsi
  __int64 *v33; // r13
  __int64 v34; // r12
  _QWORD *v35; // r14
  __int64 v36; // r8
  __int64 v37; // r11
  unsigned int v38; // ecx
  __int64 *v39; // r12
  __int64 v40; // rsi
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 v43; // rsi
  unsigned __int8 *v44; // rsi
  __int64 *v45; // rax
  __int64 v46; // rsi
  int v47; // edx
  char v48; // dl
  int v49; // eax
  __int64 v50; // rax
  char v51; // bl
  unsigned __int64 v52; // rdi
  __int64 v53; // rdi
  __int64 v54; // rsi
  _QWORD *v56; // rax
  __m128i *v57; // rdx
  __int64 v58; // r12
  __m128i si128; // xmm0
  const char *v60; // rax
  _BYTE *v61; // rdi
  unsigned __int8 *v62; // rsi
  size_t v63; // r13
  unsigned __int64 v64; // rax
  _QWORD *v65; // rax
  __m128i *v66; // rdx
  __int64 v67; // rdi
  __m128i v68; // xmm0
  __int64 v69; // rax
  _WORD *v70; // rdx
  __int64 v71; // rdi
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // r12
  unsigned __int64 v75; // rdx
  const char *v76; // r14
  size_t v77; // r13
  __int64 v78; // rax
  unsigned __int64 *v79; // rdx
  size_t v80; // rdx
  unsigned __int64 *v81; // rsi
  __int64 v82; // r12
  __int64 v83; // rax
  const char *v84; // rax
  size_t v85; // rdx
  void *v86; // rdi
  unsigned __int8 *v87; // rsi
  size_t v88; // r13
  unsigned __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdi
  _BYTE *v92; // rax
  __int64 v93; // rax
  unsigned __int64 *v94; // rdi
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 *v97; // [rsp+0h] [rbp-180h]
  bool v99; // [rsp+10h] [rbp-170h]
  unsigned int v100; // [rsp+10h] [rbp-170h]
  __int64 v101; // [rsp+18h] [rbp-168h]
  __int64 v102; // [rsp+28h] [rbp-158h]
  __int64 v103; // [rsp+50h] [rbp-130h]
  unsigned int v104; // [rsp+58h] [rbp-128h]
  bool v105; // [rsp+5Fh] [rbp-121h]
  bool v106; // [rsp+5Fh] [rbp-121h]
  unsigned __int8 *v107; // [rsp+68h] [rbp-118h] BYREF
  unsigned __int64 v108; // [rsp+70h] [rbp-110h] BYREF
  _BYTE *v109; // [rsp+78h] [rbp-108h]
  _BYTE *v110; // [rsp+80h] [rbp-100h]
  unsigned __int64 v111; // [rsp+90h] [rbp-F0h] BYREF
  _BYTE *v112; // [rsp+98h] [rbp-E8h]
  _BYTE *v113; // [rsp+A0h] [rbp-E0h]
  __int64 v114; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v115; // [rsp+B8h] [rbp-C8h]
  __int64 v116; // [rsp+C0h] [rbp-C0h]
  unsigned int v117; // [rsp+C8h] [rbp-B8h]
  unsigned __int64 v118; // [rsp+D0h] [rbp-B0h] BYREF
  int v119; // [rsp+D8h] [rbp-A8h] BYREF
  unsigned __int64 v120; // [rsp+E0h] [rbp-A0h]
  int *v121; // [rsp+E8h] [rbp-98h]
  int *v122; // [rsp+F0h] [rbp-90h]
  __int64 v123; // [rsp+F8h] [rbp-88h]
  unsigned __int64 *v124; // [rsp+100h] [rbp-80h] BYREF
  __int64 v125; // [rsp+108h] [rbp-78h]
  unsigned __int64 v126; // [rsp+110h] [rbp-70h] BYREF
  _BYTE v127[104]; // [rsp+118h] [rbp-68h] BYREF

  if ( (_BYTE)qword_50146C8 )
  {
    v56 = sub_CB72A0();
    v57 = (__m128i *)v56[4];
    v58 = (__int64)v56;
    if ( v56[3] - (_QWORD)v57 <= 0x14u )
    {
      v58 = sub_CB6200((__int64)v56, "Normalizing function ", 0x15u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42DFE50);
      v57[1].m128i_i32[0] = 1852795252;
      v57[1].m128i_i8[4] = 32;
      *v57 = si128;
      v56[4] += 21LL;
    }
    v60 = sub_BD5D20(a1);
    v61 = *(_BYTE **)(v58 + 32);
    v62 = (unsigned __int8 *)v60;
    v63 = a3;
    v64 = *(_QWORD *)(v58 + 24) - (_QWORD)v61;
    if ( v64 < a3 )
    {
      v93 = sub_CB6200(v58, v62, a3);
      v61 = *(_BYTE **)(v93 + 32);
      v58 = v93;
      if ( *(_QWORD *)(v93 + 24) - (_QWORD)v61 > 4u )
      {
LABEL_97:
        *(_DWORD *)v61 = 774778400;
        v61[4] = 10;
        *(_QWORD *)(v58 + 32) += 5LL;
        goto LABEL_2;
      }
    }
    else
    {
      if ( a3 )
      {
        memcpy(v61, v62, a3);
        v90 = *(_QWORD *)(v58 + 24);
        v61 = (_BYTE *)(v63 + *(_QWORD *)(v58 + 32));
        *(_QWORD *)(v58 + 32) = v61;
        v64 = v90 - (_QWORD)v61;
      }
      if ( v64 > 4 )
        goto LABEL_97;
    }
    sub_CB6200(v58, " ...\n", 5u);
  }
LABEL_2:
  v125 = 0;
  v124 = (unsigned __int64 *)v127;
  v126 = 32;
  sub_2CF5D20(a1, &v124, a3, a4, a5, a6);
  v6 = sub_BA8DC0(*(_QWORD *)(a1 + 40), (__int64)v124, v125);
  if ( v124 != (unsigned __int64 *)v127 )
    _libc_free((unsigned __int64)v124);
  v108 = 0;
  v109 = 0;
  v7 = *(_QWORD *)(a1 + 80);
  v110 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v105 = v6 != 0;
  if ( v7 == a1 + 72 )
  {
    v99 = 0;
    v53 = 0;
    v54 = 0;
    goto LABEL_85;
  }
  do
  {
    if ( !v7 )
      BUG();
    v8 = *(_QWORD *)(v7 + 32);
    v9 = v7 + 24;
    if ( v8 != v7 + 24 )
    {
      while ( 1 )
      {
        if ( !v8 )
          BUG();
        if ( *(_BYTE *)(v8 - 24) != 63 )
          goto LABEL_8;
        v124 = (unsigned __int64 *)(v8 - 24);
        if ( (unsigned __int8)sub_2CF7710(
                                *(_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 - 20) & 0x7FFFFFF) - 24),
                                (__int64)&v114) )
          goto LABEL_8;
        v10 = v109;
        if ( v109 == v110 )
        {
          sub_2CF6420((__int64)&v108, v109, &v124);
LABEL_8:
          v8 = *(_QWORD *)(v8 + 8);
          if ( v9 == v8 )
            break;
        }
        else
        {
          if ( v109 )
          {
            *(_QWORD *)v109 = v124;
            v10 = v109;
          }
          v109 = v10 + 8;
          v8 = *(_QWORD *)(v8 + 8);
          if ( v9 == v8 )
            break;
        }
      }
    }
    v7 = *(_QWORD *)(v7 + 8);
  }
  while ( a1 + 72 != v7 );
  v11 = v109;
  v12 = v108;
  if ( v109 != (_BYTE *)v108 )
  {
    if ( !(_BYTE)qword_50146C8 )
      goto LABEL_19;
    v65 = sub_CB72A0();
    v66 = (__m128i *)v65[4];
    v67 = (__int64)v65;
    if ( v65[3] - (_QWORD)v66 <= 0x14u )
    {
      v67 = sub_CB6200((__int64)v65, "Normalize GEP Index (", 0x15u);
    }
    else
    {
      v68 = _mm_load_si128((const __m128i *)&xmmword_42DFE60);
      v66[1].m128i_i32[0] = 544761188;
      v66[1].m128i_i8[4] = 40;
      *v66 = v68;
      v65[4] += 21LL;
    }
    v69 = sub_CB59D0(v67, (__int64)&v109[-v108] >> 3);
    v70 = *(_WORD **)(v69 + 32);
    v71 = v69;
    if ( *(_QWORD *)(v69 + 24) - (_QWORD)v70 <= 1u )
    {
      v71 = sub_CB6200(v69, (unsigned __int8 *)", ", 2u);
    }
    else
    {
      *v70 = 8236;
      *(_QWORD *)(v69 + 32) += 2LL;
    }
    v72 = sub_CB59F0(v71, v105);
    v73 = *(_QWORD *)(v72 + 32);
    v74 = v72;
    if ( (unsigned __int64)(*(_QWORD *)(v72 + 24) - v73) <= 4 )
    {
      v74 = sub_CB6200(v72, " ) : ", 5u);
    }
    else
    {
      *(_DWORD *)v73 = 975186208;
      *(_BYTE *)(v73 + 4) = 32;
      *(_QWORD *)(v72 + 32) += 5LL;
    }
    v76 = sub_BD5D20(a1);
    v77 = v75;
    if ( !v76 )
    {
      v125 = 0;
      v80 = 0;
      v124 = &v126;
      v81 = &v126;
      LOBYTE(v126) = 0;
      goto LABEL_119;
    }
    v118 = v75;
    v78 = v75;
    v124 = &v126;
    if ( v75 > 0xF )
    {
      v124 = (unsigned __int64 *)sub_22409D0((__int64)&v124, &v118, 0);
      v94 = v124;
      v126 = v118;
    }
    else
    {
      if ( v75 == 1 )
      {
        LOBYTE(v126) = *v76;
        v79 = &v126;
LABEL_108:
        v125 = v78;
        *((_BYTE *)v79 + v78) = 0;
        v80 = v125;
        v81 = v124;
LABEL_119:
        v91 = sub_CB6200(v74, (unsigned __int8 *)v81, v80);
        v92 = *(_BYTE **)(v91 + 32);
        if ( *(_BYTE **)(v91 + 24) == v92 )
        {
          sub_CB6200(v91, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v92 = 10;
          ++*(_QWORD *)(v91 + 32);
        }
        if ( v124 != &v126 )
          j_j___libc_free_0((unsigned __int64)v124);
        v11 = v109;
        v12 = v108;
LABEL_19:
        v119 = 0;
        v13 = (__int64)&v11[-v12] >> 3;
        v121 = &v119;
        v120 = 0;
        v122 = &v119;
        v123 = 0;
        if ( !(_DWORD)v13 )
        {
          v99 = 0;
          v52 = 0;
LABEL_84:
          sub_2CF5F10(v52);
          v53 = v115;
          v54 = 16LL * v117;
          goto LABEL_85;
        }
        v99 = 0;
        v14 = v105;
        v103 = 0;
        v101 = 8LL * (unsigned int)(v13 - 1);
        while ( 2 )
        {
          v15 = &v126;
          v16 = *(_QWORD *)(v12 + v103);
          v124 = &v126;
          v125 = 0x800000001LL;
          v126 = v16;
          v17 = 1;
          while ( 1 )
          {
            v18 = v15[v17 - 1];
            LODWORD(v125) = v17 - 1;
            v21 = sub_BCAC40(*(_QWORD *)(v16 + 72), 8);
            if ( !v21 )
              break;
            for ( i = *(_QWORD *)(v18 + 16); i; i = *(_QWORD *)(i + 8) )
            {
              v23 = *(_QWORD *)(i + 24);
              v24 = *(_BYTE *)v23;
              if ( *(_BYTE *)v23 <= 0x1Cu )
                goto LABEL_27;
              if ( v24 == 63 )
              {
                v41 = (unsigned int)v125;
                v42 = (unsigned int)v125 + 1LL;
                if ( v42 > HIDWORD(v125) )
                {
                  sub_C8D5F0((__int64)&v124, &v126, v42, 8u, v19, v20);
                  v41 = (unsigned int)v125;
                }
                v124[v41] = v23;
                LODWORD(v125) = v125 + 1;
              }
              else if ( v24 != 78 || *(_BYTE *)(*(_QWORD *)(v23 + 8) + 8LL) != 14 )
              {
                goto LABEL_27;
              }
            }
            v25 = v124;
            v17 = v125;
            v15 = v124;
            if ( !(_DWORD)v125 )
              goto LABEL_28;
          }
LABEL_27:
          v25 = v124;
          v21 = 0;
LABEL_28:
          if ( v25 != &v126 )
            _libc_free((unsigned __int64)v25);
          v26 = *(_QWORD *)(v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF));
          v111 = 0;
          v112 = 0;
          v102 = v26;
          v113 = 0;
          v27 = 32 * (1LL - (*(_DWORD *)(v16 + 4) & 0x7FFFFFF));
          if ( v16 == v16 + v27 )
          {
LABEL_47:
            if ( v101 != v103 )
            {
              v12 = v108;
              v103 += 8;
              continue;
            }
            if ( v99 && (v51 = qword_50146C8) != 0 )
            {
              v82 = (__int64)sub_CB72A0();
              v83 = *(_QWORD *)(v82 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v82 + 24) - v83) <= 8 )
              {
                v82 = sub_CB6200(v82, (unsigned __int8 *)"Function ", 9u);
              }
              else
              {
                *(_BYTE *)(v83 + 8) = 32;
                *(_QWORD *)v83 = 0x6E6F6974636E7546LL;
                *(_QWORD *)(v82 + 32) += 9LL;
              }
              v84 = sub_BD5D20(a1);
              v86 = *(void **)(v82 + 32);
              v87 = (unsigned __int8 *)v84;
              v88 = v85;
              v89 = *(_QWORD *)(v82 + 24) - (_QWORD)v86;
              if ( v85 > v89 )
              {
                v95 = sub_CB6200(v82, v87, v85);
                v86 = *(void **)(v95 + 32);
                v82 = v95;
                v89 = *(_QWORD *)(v95 + 24) - (_QWORD)v86;
              }
              else if ( v85 )
              {
                memcpy(v86, v87, v85);
                v96 = *(_QWORD *)(v82 + 24);
                v86 = (void *)(v88 + *(_QWORD *)(v82 + 32));
                *(_QWORD *)(v82 + 32) = v86;
                v89 = v96 - (_QWORD)v86;
              }
              if ( v89 <= 0xE )
              {
                sub_CB6200(v82, " is normalized\n", 0xFu);
              }
              else
              {
                qmemcpy(v86, " is normalized\n", 15);
                *(_QWORD *)(v82 + 32) += 15LL;
              }
              v99 = v51;
              v52 = v120;
            }
            else
            {
              v52 = v120;
            }
            goto LABEL_84;
          }
          break;
        }
        v106 = 0;
        v28 = (unsigned __int8 **)(v16 + v27);
        do
        {
          while ( 1 )
          {
            v107 = *v28;
            v30 = sub_BCAC40(*((_QWORD *)v107 + 1), 64);
            if ( !v30 )
              break;
            v31 = sub_2CF65B0(v107, &v118, v21, v14);
            v124 = (unsigned __int64 *)v31;
            if ( !v31 )
              break;
            v32 = v112;
            if ( v112 == v113 )
            {
              sub_9281F0((__int64)&v111, v112, &v124);
            }
            else
            {
              if ( v112 )
              {
                *(_QWORD *)v112 = v31;
                v32 = v112;
              }
              v112 = v32 + 8;
            }
            v28 += 4;
            v106 = v30;
            if ( (unsigned __int8 **)v16 == v28 )
              goto LABEL_44;
          }
          v29 = v112;
          if ( v112 == v113 )
          {
            sub_9281F0((__int64)&v111, v112, &v107);
          }
          else
          {
            if ( v112 )
            {
              *(_QWORD *)v112 = v107;
              v29 = v112;
            }
            v112 = v29 + 8;
          }
          v28 += 4;
        }
        while ( (unsigned __int8 **)v16 != v28 );
LABEL_44:
        if ( !v106 )
        {
LABEL_45:
          if ( v111 )
            j_j___libc_free_0(v111);
          goto LABEL_47;
        }
        v127[9] = 1;
        v33 = (__int64 *)v111;
        v124 = (unsigned __int64 *)"newGep";
        v127[8] = 3;
        v34 = (__int64)&v112[-v111] >> 3;
        v97 = (__int64 *)v112;
        v35 = sub_BD2C40(88, (int)v34 + 1);
        if ( v35 )
        {
          v36 = v16 + 24;
          v37 = *(_QWORD *)(v102 + 8);
          v38 = (v34 + 1) & 0x7FFFFFF | v104 & 0xE0000000;
          v104 = v38;
          if ( (unsigned int)*(unsigned __int8 *)(v37 + 8) - 17 > 1 && v97 != v33 )
          {
            v45 = v33;
            v46 = *(_QWORD *)(*v33 + 8);
            v47 = *(unsigned __int8 *)(v46 + 8);
            if ( v47 == 17 )
            {
LABEL_79:
              v48 = 0;
            }
            else
            {
              while ( v47 != 18 )
              {
                if ( v97 == ++v45 )
                  goto LABEL_54;
                v46 = *(_QWORD *)(*v45 + 8);
                v47 = *(unsigned __int8 *)(v46 + 8);
                if ( v47 == 17 )
                  goto LABEL_79;
              }
              v48 = 1;
            }
            v49 = *(_DWORD *)(v46 + 32);
            BYTE4(v107) = v48;
            v100 = v38;
            LODWORD(v107) = v49;
            v50 = sub_BCE1B0((__int64 *)v37, (__int64)v107);
            v36 = v16 + 24;
            v38 = v100;
            v37 = v50;
          }
LABEL_54:
          sub_B44260((__int64)v35, v37, 34, v38, v36, 0);
          v35[9] = 0;
          v35[10] = sub_B4DC50(0, (__int64)v33, v34);
          sub_B4D9A0((__int64)v35, v102, v33, v34, (__int64)&v124);
        }
        v39 = v35 + 6;
        sub_BD84D0(v16, (__int64)v35);
        v40 = *(_QWORD *)(v16 + 48);
        v124 = (unsigned __int64 *)v40;
        if ( v40 )
        {
          sub_B96E90((__int64)&v124, v40, 1);
          if ( v39 == (__int64 *)&v124 )
          {
            if ( v124 )
              sub_B91220((__int64)&v124, (__int64)v124);
            goto LABEL_59;
          }
          v43 = v35[6];
          if ( !v43 )
          {
LABEL_72:
            v44 = (unsigned __int8 *)v124;
            v35[6] = v124;
            if ( v44 )
              sub_B976B0((__int64)&v124, v44, (__int64)(v35 + 6));
            goto LABEL_59;
          }
        }
        else if ( v39 == (__int64 *)&v124 || (v43 = v35[6]) == 0 )
        {
LABEL_59:
          v99 = v106;
          goto LABEL_45;
        }
        sub_B91220((__int64)(v35 + 6), v43);
        goto LABEL_72;
      }
      if ( !v75 )
      {
        v79 = &v126;
        goto LABEL_108;
      }
      v94 = &v126;
    }
    memcpy(v94, v76, v77);
    v78 = v118;
    v79 = v124;
    goto LABEL_108;
  }
  v99 = 0;
  v53 = v115;
  v54 = 16LL * v117;
LABEL_85:
  sub_C7D6A0(v53, v54, 8);
  if ( v108 )
    j_j___libc_free_0(v108);
  return v99;
}
