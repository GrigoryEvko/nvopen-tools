// Function: sub_303F880
// Address: 0x303f880
//
__int64 __fastcall sub_303F880(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        _QWORD *a6,
        unsigned __int16 a7,
        int a8,
        __int64 a9,
        char a10,
        __int64 a11,
        __int64 a12,
        unsigned int a13)
{
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int); // rax
  int v16; // eax
  char *v17; // rdx
  _QWORD *v18; // rdi
  __int64 v19; // rax
  __m128i *v20; // rdx
  __m128i si128; // xmm0
  unsigned __int8 v22; // al
  char *v23; // rdx
  unsigned __int64 v24; // rsi
  _QWORD *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r13
  char v29; // bl
  unsigned __int8 *v30; // rax
  size_t v31; // rdx
  unsigned __int64 v32; // r12
  _BYTE *v33; // rax
  char *v34; // rdx
  __int64 v35; // rdx
  unsigned int v36; // r13d
  __int64 v37; // rcx
  unsigned int v38; // r12d
  __int64 v39; // r14
  __int64 v40; // rdx
  int v41; // eax
  bool v42; // zf
  unsigned int v43; // ecx
  unsigned int v44; // eax
  char v45; // al
  char *v46; // rdx
  char v47; // r14
  _QWORD *v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rdx
  char *v51; // rax
  _QWORD *v52; // rdi
  __int64 v53; // rdi
  _BYTE *v54; // rax
  char *v55; // rsi
  __int64 v56; // rbx
  __int64 v57; // rax
  char v58; // al
  size_t v59; // rdx
  unsigned __int64 v60; // r14
  _QWORD *v61; // rdx
  _QWORD *v62; // rdi
  __int64 v63; // rdi
  _BYTE *v64; // rax
  unsigned int v65; // edx
  unsigned int v66; // eax
  _QWORD *v67; // rdx
  unsigned __int64 v68; // r12
  _QWORD *v69; // rdi
  __int64 v70; // rax
  _WORD *v71; // rdx
  __int64 v72; // rax
  char *v73; // rdx
  _QWORD *v74; // r12
  __int64 v75; // rdx
  _BYTE *v76; // rax
  char v78; // al
  char *v79; // rdx
  char v80; // bl
  _QWORD *v81; // rdi
  __int64 v82; // rax
  __int64 v83; // rdx
  char *v84; // rax
  _QWORD *v85; // rbx
  unsigned __int8 *v86; // rax
  size_t v87; // rdx
  unsigned __int64 v88; // r8
  _BYTE *v89; // rax
  unsigned int v90; // edx
  unsigned int v91; // eax
  unsigned __int8 *v92; // rdx
  _QWORD *v93; // rcx
  __int64 v94; // rdx
  _QWORD *v95; // rax
  _QWORD *v96; // rdi
  char *v97; // rdx
  __int64 v98; // rdx
  _QWORD *v99; // rdx
  _QWORD *v100; // r12
  unsigned __int64 v101; // r13
  _WORD *v102; // rdx
  __int64 v103; // rdx
  unsigned __int8 *v104; // rax
  size_t v105; // rdx
  __int64 v106; // rdx
  __int128 v107; // [rsp-10h] [rbp-1F0h]
  char v108; // [rsp+8h] [rbp-1D8h]
  unsigned __int64 v109; // [rsp+8h] [rbp-1D8h]
  int v113; // [rsp+28h] [rbp-1B8h]
  unsigned __int16 v114; // [rsp+2Ch] [rbp-1B4h]
  unsigned __int8 v115; // [rsp+2Fh] [rbp-1B1h]
  __m128i *v117; // [rsp+40h] [rbp-1A0h] BYREF
  size_t v118; // [rsp+48h] [rbp-198h]
  __m128i v119; // [rsp+50h] [rbp-190h] BYREF
  _QWORD v120[3]; // [rsp+60h] [rbp-180h] BYREF
  char *v121; // [rsp+78h] [rbp-168h]
  char *v122; // [rsp+80h] [rbp-160h]
  __int64 v123; // [rsp+88h] [rbp-158h]
  __int64 v124; // [rsp+90h] [rbp-150h]
  __m128i *v125; // [rsp+A0h] [rbp-140h] BYREF
  size_t v126; // [rsp+A8h] [rbp-138h]
  __m128i v127; // [rsp+B0h] [rbp-130h] BYREF
  char v128; // [rsp+C0h] [rbp-120h]

  v15 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*a2 + 32LL);
  if ( v15 == sub_2D42F30 )
  {
    v114 = 2;
    v16 = sub_AE2980(a3, 0)[1];
    if ( v16 != 1 )
    {
      v114 = 3;
      if ( v16 != 2 )
      {
        v114 = 4;
        if ( v16 != 4 )
        {
          v114 = 5;
          if ( v16 != 8 )
          {
            v114 = 6;
            if ( v16 != 16 )
            {
              v114 = 7;
              if ( v16 != 32 )
              {
                v114 = 8;
                if ( v16 != 64 )
                  v114 = 9 * (v16 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v114 = v15((__int64)a2, a3, 0);
  }
  v120[1] = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = a1 + 16;
  v123 = 0x100000000LL;
  *(_BYTE *)(a1 + 16) = 0;
  v120[2] = 0;
  v120[0] = &unk_49DD210;
  v121 = 0;
  v122 = 0;
  v124 = a1;
  sub_CB5980((__int64)v120, 0, 0, 0);
  v17 = v122;
  if ( (unsigned __int64)(v121 - v122) <= 9 )
  {
    v18 = (_QWORD *)sub_CB6200((__int64)v120, "prototype_", 0xAu);
  }
  else
  {
    v18 = v120;
    *(_QWORD *)v122 = 0x7079746F746F7270LL;
    *((_WORD *)v17 + 4) = 24421;
    v122 += 10;
  }
  v19 = sub_CB59D0((__int64)v18, a13);
  v20 = *(__m128i **)(v19 + 32);
  if ( *(_QWORD *)(v19 + 24) - (_QWORD)v20 <= 0x11u )
  {
    sub_CB6200(v19, " : .callprototype ", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_435DC10);
    v20[1].m128i_i16[0] = 8293;
    *v20 = si128;
    *(_QWORD *)(v19 + 32) += 18LL;
  }
  if ( *(_BYTE *)(a4 + 8) != 7 )
  {
    if ( v121 == v122 )
    {
      sub_CB6200((__int64)v120, (unsigned __int8 *)"(", 1u);
      v22 = *(_BYTE *)(a4 + 8);
      if ( v22 <= 3u )
        goto LABEL_66;
    }
    else
    {
      *v122 = 40;
      v22 = *(_BYTE *)(a4 + 8);
      ++v122;
      if ( v22 <= 3u )
        goto LABEL_66;
    }
    if ( v22 != 5 && (v22 & 0xFD) != 4 && v22 != 12 )
    {
      if ( v22 != 14 )
        goto LABEL_21;
      goto LABEL_154;
    }
LABEL_66:
    if ( !(unsigned __int8)sub_302F170(a4) )
    {
      if ( *(_BYTE *)(a4 + 8) == 12 )
      {
        v65 = *(_DWORD *)(a4 + 8) >> 8;
      }
      else
      {
        v104 = (unsigned __int8 *)sub_BCAE30(a4);
        v126 = v105;
        v125 = (__m128i *)v104;
        v65 = sub_CA1930(&v125);
      }
      v66 = 32;
      if ( v65 > 0x20 )
      {
        v66 = 64;
        if ( v65 >= 0x40 )
          v66 = v65;
      }
      v67 = v122;
      v68 = v66;
      if ( (unsigned __int64)(v121 - v122) <= 8 )
      {
        v69 = (_QWORD *)sub_CB6200((__int64)v120, (unsigned __int8 *)".param .b", 9u);
      }
      else
      {
        v122[8] = 98;
        v69 = v120;
        *v67 = 0x2E206D617261702ELL;
        v122 += 9;
      }
      v70 = sub_CB59D0((__int64)v69, v68);
      v71 = *(_WORD **)(v70 + 32);
      if ( *(_QWORD *)(v70 + 24) - (_QWORD)v71 <= 1u )
      {
        sub_CB6200(v70, " _", 2u);
      }
      else
      {
        *v71 = 24352;
        *(_QWORD *)(v70 + 32) += 2LL;
      }
      goto LABEL_29;
    }
    if ( *(_BYTE *)(a4 + 8) != 14 )
    {
LABEL_21:
      if ( !(unsigned __int8)sub_302F170(a4) )
        BUG();
      v23 = v122;
      if ( (unsigned __int64)(v121 - v122) <= 0xD )
      {
        v24 = 0;
        v25 = (_QWORD *)sub_CB6200((__int64)v120, (unsigned __int8 *)".param .align ", 0xEu);
        if ( !HIBYTE(a7) )
          goto LABEL_24;
      }
      else
      {
        *((_DWORD *)v122 + 2) = 1734962273;
        v24 = 0;
        v25 = v120;
        *(_QWORD *)v23 = 0x2E206D617261702ELL;
        *((_WORD *)v23 + 6) = 8302;
        v122 += 14;
        if ( !HIBYTE(a7) )
        {
LABEL_24:
          v26 = sub_CB59D0((__int64)v25, v24);
          v27 = *(_QWORD *)(v26 + 32);
          v28 = v26;
          if ( (unsigned __int64)(*(_QWORD *)(v26 + 24) - v27) <= 6 )
          {
            v28 = sub_CB6200(v26, " .b8 _[", 7u);
          }
          else
          {
            *(_DWORD *)v27 = 945958432;
            *(_WORD *)(v27 + 4) = 24352;
            *(_BYTE *)(v27 + 6) = 91;
            *(_QWORD *)(v26 + 32) += 7LL;
          }
          v29 = sub_AE5020(a3, a4);
          v30 = (unsigned __int8 *)sub_9208B0(a3, a4);
          v126 = v31;
          v125 = (__m128i *)v30;
          v32 = (((unsigned __int64)(v30 + 7) >> 3) + (1LL << v29) - 1) >> v29 << v29;
          if ( (_BYTE)v31 )
          {
            v103 = *(_QWORD *)(v28 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(v28 + 24) - v103) <= 8 )
            {
              sub_CB6200(v28, "vscale x ", 9u);
            }
            else
            {
              *(_BYTE *)(v103 + 8) = 32;
              *(_QWORD *)v103 = 0x7820656C61637376LL;
              *(_QWORD *)(v28 + 32) += 9LL;
            }
          }
          sub_CB59D0(v28, v32);
          v33 = *(_BYTE **)(v28 + 32);
          if ( *(_BYTE **)(v28 + 24) == v33 )
          {
            sub_CB6200(v28, (unsigned __int8 *)"]", 1u);
          }
          else
          {
            *v33 = 93;
            ++*(_QWORD *)(v28 + 32);
          }
LABEL_29:
          if ( (unsigned __int64)(v121 - v122) <= 1 )
          {
            sub_CB6200((__int64)v120, (unsigned __int8 *)") ", 2u);
            v34 = v122;
          }
          else
          {
            *(_WORD *)v122 = 8233;
            v34 = v122 + 2;
            v122 += 2;
          }
          goto LABEL_31;
        }
      }
      v24 = 1LL << a7;
      goto LABEL_24;
    }
LABEL_154:
    v99 = v122;
    if ( (unsigned __int64)(v121 - v122) <= 8 )
    {
      v100 = (_QWORD *)sub_CB6200((__int64)v120, (unsigned __int8 *)".param .b", 9u);
    }
    else
    {
      v122[8] = 98;
      v100 = v120;
      *v99 = 0x2E206D617261702ELL;
      v122 += 9;
    }
    if ( v114 <= 1u || (unsigned __int16)(v114 - 504) <= 7u )
LABEL_187:
      BUG();
    v101 = *(_QWORD *)&byte_444C4A0[16 * v114 - 16];
    if ( byte_444C4A0[16 * v114 - 8] )
    {
      v106 = v100[4];
      if ( (unsigned __int64)(v100[3] - v106) <= 8 )
      {
        sub_CB6200((__int64)v100, "vscale x ", 9u);
      }
      else
      {
        *(_BYTE *)(v106 + 8) = 32;
        *(_QWORD *)v106 = 0x7820656C61637376LL;
        v100[4] += 9LL;
      }
    }
    sub_CB59D0((__int64)v100, v101);
    v102 = (_WORD *)v100[4];
    if ( v100[3] - (_QWORD)v102 <= 1u )
    {
      sub_CB6200((__int64)v100, " _", 2u);
    }
    else
    {
      *v102 = 24352;
      v100[4] += 2LL;
    }
    goto LABEL_29;
  }
  if ( (unsigned __int64)(v121 - v122) <= 1 )
  {
    sub_CB6200((__int64)v120, (unsigned __int8 *)"()", 2u);
    v34 = v122;
  }
  else
  {
    *(_WORD *)v122 = 10536;
    v34 = v122 + 2;
    v122 += 2;
  }
LABEL_31:
  if ( (unsigned __int64)(v121 - v34) <= 2 )
  {
    sub_CB6200((__int64)v120, "_ (", 3u);
  }
  else
  {
    v34[2] = 40;
    *(_WORD *)v34 = 8287;
    v122 += 3;
  }
  if ( a10 )
  {
    v113 = a8;
    if ( !a8 )
    {
      v73 = v122;
      v74 = v120;
LABEL_84:
      if ( v74[3] - (_QWORD)v73 <= 0xEu )
        goto LABEL_163;
      goto LABEL_85;
    }
    v35 = *a5;
  }
  else
  {
    v35 = *a5;
    v113 = -1431655765 * ((a5[1] - *a5) >> 4);
    if ( !v113 )
    {
LABEL_88:
      v55 = v122;
      goto LABEL_89;
    }
  }
  v36 = 0;
  v37 = 0;
  v38 = 1;
  v39 = *(_QWORD *)(v35 + 24);
  while ( 1 )
  {
    v56 = 56LL * v36;
    v57 = v56 + *a6;
    if ( (*(_BYTE *)v57 & 0x20) != 0 )
    {
      v40 = *(_QWORD *)(*a5 + v37 + 40);
      v41 = (*(_WORD *)(v57 + 2) >> 4) & 0x3F;
      v42 = (_BYTE)v41 == 0;
      v43 = v41 - 1;
      v44 = v115;
      if ( !v42 )
        v44 = v43;
      v115 = v44;
      v45 = sub_303F840((__int64)a2, 0, v40, v44, a3);
      v46 = v122;
      v47 = v45;
      if ( (unsigned __int64)(v121 - v122) <= 0xD )
      {
        v48 = (_QWORD *)sub_CB6200((__int64)v120, (unsigned __int8 *)".param .align ", 0xEu);
      }
      else
      {
        *((_DWORD *)v122 + 2) = 1734962273;
        v48 = v120;
        *(_QWORD *)v46 = 0x2E206D617261702ELL;
        *((_WORD *)v46 + 6) = 8302;
        v122 += 14;
      }
      v49 = sub_CB59D0((__int64)v48, 1LL << v47);
      v50 = *(_QWORD *)(v49 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v49 + 24) - v50) <= 4 )
      {
        sub_CB6200(v49, " .b8 ", 5u);
      }
      else
      {
        *(_DWORD *)v50 = 945958432;
        *(_BYTE *)(v50 + 4) = 32;
        *(_QWORD *)(v49 + 32) += 5LL;
      }
      if ( v121 == v122 )
      {
        sub_CB6200((__int64)v120, (unsigned __int8 *)"_", 1u);
        v51 = v122;
      }
      else
      {
        *v122 = 95;
        v51 = ++v122;
      }
      if ( v121 == v51 )
      {
        v52 = (_QWORD *)sub_CB6200((__int64)v120, (unsigned __int8 *)"[", 1u);
      }
      else
      {
        *v51 = 91;
        v52 = v120;
        ++v122;
      }
      v53 = sub_CB59D0((__int64)v52, *(unsigned int *)(*a6 + v56 + 8));
      v54 = *(_BYTE **)(v53 + 32);
      if ( *(_BYTE **)(v53 + 24) == v54 )
      {
        sub_CB6200(v53, (unsigned __int8 *)"]", 1u);
      }
      else
      {
        *v54 = 93;
        ++*(_QWORD *)(v53 + 32);
      }
      goto LABEL_49;
    }
    if ( (unsigned __int8)sub_302F170(v39) )
    {
      v78 = sub_303E700((__int64)a2, a12, v39, v38, a3);
      v79 = v122;
      v80 = v78;
      if ( (unsigned __int64)(v121 - v122) <= 0xD )
      {
        v81 = (_QWORD *)sub_CB6200((__int64)v120, (unsigned __int8 *)".param .align ", 0xEu);
      }
      else
      {
        *((_DWORD *)v122 + 2) = 1734962273;
        v81 = v120;
        *(_QWORD *)v79 = 0x2E206D617261702ELL;
        *((_WORD *)v79 + 6) = 8302;
        v122 += 14;
      }
      v82 = sub_CB59D0((__int64)v81, 1LL << v80);
      v83 = *(_QWORD *)(v82 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v82 + 24) - v83) <= 4 )
      {
        sub_CB6200(v82, " .b8 ", 5u);
      }
      else
      {
        *(_DWORD *)v83 = 945958432;
        *(_BYTE *)(v83 + 4) = 32;
        *(_QWORD *)(v82 + 32) += 5LL;
      }
      if ( v121 == v122 )
      {
        sub_CB6200((__int64)v120, (unsigned __int8 *)"_", 1u);
        v84 = v122;
      }
      else
      {
        *v122 = 95;
        v84 = ++v122;
      }
      if ( v121 == v84 )
      {
        v85 = (_QWORD *)sub_CB6200((__int64)v120, (unsigned __int8 *)"[", 1u);
      }
      else
      {
        *v84 = 91;
        v85 = v120;
        ++v122;
      }
      v108 = sub_AE5020(a3, v39);
      v86 = (unsigned __int8 *)sub_9208B0(a3, v39);
      v126 = v87;
      v125 = (__m128i *)v86;
      v88 = (((unsigned __int64)(v86 + 7) >> 3) + (1LL << v108) - 1) >> v108 << v108;
      if ( (_BYTE)v87 )
      {
        v98 = v85[4];
        if ( (unsigned __int64)(v85[3] - v98) <= 8 )
        {
          v109 = (((unsigned __int64)(v86 + 7) >> 3) + (1LL << v108) - 1) >> v108 << v108;
          sub_CB6200((__int64)v85, "vscale x ", 9u);
          v88 = v109;
        }
        else
        {
          *(_BYTE *)(v98 + 8) = 32;
          *(_QWORD *)v98 = 0x7820656C61637376LL;
          v85[4] += 9LL;
        }
      }
      sub_CB59D0((__int64)v85, v88);
      v89 = (_BYTE *)v85[4];
      if ( (_BYTE *)v85[3] == v89 )
      {
        sub_CB6200((__int64)v85, (unsigned __int8 *)"]", 1u);
      }
      else
      {
        *v89 = 93;
        ++v85[4];
      }
      LOBYTE(v118) = 0;
      *((_QWORD *)&v107 + 1) = v118;
      v117 = 0;
      *(_QWORD *)&v107 = 0;
      v126 = 0x1000000000LL;
      v125 = &v127;
      sub_34B8C80((_DWORD)a2, a3, v39, (unsigned int)&v125, 0, 0, v107);
      if ( (_DWORD)v126 )
        v36 = v36 + v126 - 1;
      if ( v125 != &v127 )
        _libc_free((unsigned __int64)v125);
LABEL_49:
      v55 = v122;
      goto LABEL_50;
    }
    v58 = *(_BYTE *)(v39 + 8);
    if ( v58 == 12 )
    {
      v90 = *(_DWORD *)(v39 + 8);
      v60 = 32;
      v91 = v90 >> 8;
      if ( v90 > 0x20FF )
      {
        if ( v91 < 0x40 )
          v91 = 64;
        v60 = v91;
      }
    }
    else if ( v58 == 14 )
    {
      if ( v114 <= 1u || (unsigned __int16)(v114 - 504) <= 7u )
        goto LABEL_187;
      v92 = *(unsigned __int8 **)&byte_444C4A0[16 * v114 - 16];
      LOBYTE(v126) = byte_444C4A0[16 * v114 - 8];
      v125 = (__m128i *)v92;
      v60 = (unsigned int)sub_CA1930(&v125);
    }
    else
    {
      v125 = (__m128i *)sub_BCAE30(v39);
      v126 = v59;
      v60 = (unsigned int)sub_CA1930(&v125);
    }
    v61 = v122;
    if ( (unsigned __int64)(v121 - v122) <= 8 )
    {
      v62 = (_QWORD *)sub_CB6200((__int64)v120, (unsigned __int8 *)".param .b", 9u);
    }
    else
    {
      v122[8] = 98;
      v62 = v120;
      *v61 = 0x2E206D617261702ELL;
      v122 += 9;
    }
    v63 = sub_CB59D0((__int64)v62, v60);
    v64 = *(_BYTE **)(v63 + 32);
    if ( *(_BYTE **)(v63 + 24) == v64 )
    {
      sub_CB6200(v63, (unsigned __int8 *)" ", 1u);
    }
    else
    {
      *v64 = 32;
      ++*(_QWORD *)(v63 + 32);
    }
    if ( v121 == v122 )
    {
      sub_CB6200((__int64)v120, (unsigned __int8 *)"_", 1u);
      v55 = v122;
    }
    else
    {
      *v122 = 95;
      v55 = ++v122;
    }
LABEL_50:
    ++v36;
    if ( v38 == v113 )
      break;
    v37 = 48LL * v38;
    v39 = *(_QWORD *)(*a5 + v37 + 24);
    if ( (unsigned __int64)(v121 - v55) <= 1 )
    {
      sub_CB6200((__int64)v120, (unsigned __int8 *)", ", 2u);
      v37 = 48LL * v38;
    }
    else
    {
      *(_WORD *)v55 = 8236;
      v122 += 2;
    }
    ++v38;
  }
  if ( !a10 )
    goto LABEL_89;
  if ( v55 == v121 )
  {
    v72 = sub_CB6200((__int64)v120, (unsigned __int8 *)",", 1u);
    v73 = *(char **)(v72 + 32);
    v74 = (_QWORD *)v72;
    goto LABEL_84;
  }
  *v55 = 44;
  v74 = v120;
  v73 = v122 + 1;
  v122 = v73;
  if ( (unsigned __int64)(v121 - v73) > 0xE )
  {
LABEL_85:
    qmemcpy(v73, " .param .align ", 15);
    v74[4] += 15LL;
    goto LABEL_86;
  }
LABEL_163:
  v74 = (_QWORD *)sub_CB6200((__int64)v74, " .param .align ", 0xFu);
LABEL_86:
  sub_C49420(a9, (__int64)v74, 1);
  v75 = v74[4];
  if ( (unsigned __int64)(v74[3] - v75) > 8 )
  {
    *(_BYTE *)(v75 + 8) = 10;
    *(_QWORD *)v75 = 0x5D5B5F2038622E20LL;
    v74[4] += 9LL;
    goto LABEL_88;
  }
  sub_CB6200((__int64)v74, " .b8 _[]\n", 9u);
  v55 = v122;
LABEL_89:
  if ( v55 == v121 )
  {
    sub_CB6200((__int64)v120, (unsigned __int8 *)")", 1u);
  }
  else
  {
    *v55 = 41;
    ++v122;
  }
  if ( (unsigned __int8)sub_307AAA0(a12, a2[67126]) )
  {
    v97 = v122;
    if ( (unsigned __int64)(v121 - v122) <= 9 )
    {
      sub_CB6200((__int64)v120, " .noreturn", 0xAu);
    }
    else
    {
      *(_QWORD *)v122 = 0x757465726F6E2E20LL;
      *((_WORD *)v97 + 4) = 28274;
      v122 += 10;
    }
  }
  if ( *(_DWORD *)(*(_QWORD *)a11 + 24LL) != 46 )
    goto LABEL_93;
  v93 = *(_QWORD **)(*(_QWORD *)a11 + 40LL);
  v94 = *(_QWORD *)(*v93 + 96LL);
  v95 = *(_QWORD **)(v94 + 24);
  if ( *(_DWORD *)(v94 + 32) > 0x40u )
    v95 = (_QWORD *)*v95;
  if ( (_DWORD)v95 != 8285 )
    goto LABEL_93;
  sub_314C660(&v117, *(_QWORD *)(v93[10] + 96LL), 0);
  v125 = &v127;
  if ( v117 == &v119 )
  {
    v127 = _mm_load_si128(&v119);
  }
  else
  {
    v125 = v117;
    v127.m128i_i64[0] = v119.m128i_i64[0];
  }
  v128 = 1;
  v126 = v118;
  if ( v121 == v122 )
  {
    v96 = (_QWORD *)sub_CB6200((__int64)v120, (unsigned __int8 *)" ", 1u);
  }
  else
  {
    *v122 = 32;
    v96 = v120;
    ++v122;
  }
  if ( !v128 )
    abort();
  sub_CB6200((__int64)v96, (unsigned __int8 *)v125, v126);
  if ( v128 && (v128 = 0, v125 != &v127) )
  {
    j_j___libc_free_0((unsigned __int64)v125);
    v76 = v122;
    if ( v121 == v122 )
    {
LABEL_135:
      sub_CB6200((__int64)v120, (unsigned __int8 *)";", 1u);
      goto LABEL_95;
    }
  }
  else
  {
LABEL_93:
    v76 = v122;
    if ( v121 == v122 )
      goto LABEL_135;
  }
  *v76 = 59;
  ++v122;
LABEL_95:
  v120[0] = &unk_49DD210;
  sub_CB5840((__int64)v120);
  return a1;
}
