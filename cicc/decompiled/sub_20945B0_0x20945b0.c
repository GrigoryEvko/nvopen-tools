// Function: sub_20945B0
// Address: 0x20945b0
//
char __fastcall sub_20945B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  char v6; // al
  char v7; // al
  __int16 v8; // r14
  unsigned __int64 v9; // rax
  _BYTE *v10; // rax
  _DWORD *v11; // rdx
  __int64 *v12; // r15
  __int64 *v13; // r14
  __int64 v14; // rsi
  _BYTE *v15; // rax
  unsigned int v16; // r14d
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdi
  _BYTE *v20; // rax
  unsigned __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rdi
  void *v26; // rax
  size_t v27; // rdx
  _BYTE *v28; // rdi
  size_t v29; // r13
  __int64 v30; // rdi
  unsigned __int16 v31; // bx
  _BYTE *v32; // rax
  _BYTE *v33; // rax
  __m128i *v34; // rdx
  __m128i si128; // xmm0
  _QWORD *v36; // rdx
  _DWORD *v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rdx
  _DWORD *v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rdx
  _DWORD *v44; // rdx
  _DWORD *v45; // rdx
  __int64 v46; // rdx
  _BYTE *v47; // rdi
  int v48; // eax
  __int64 v49; // r14
  __int64 v50; // r15
  int v51; // r8d
  __int64 v52; // r9
  _BYTE *v53; // rax
  _BYTE *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rdi
  __int64 v57; // rdi
  _BYTE *v58; // rax
  __int64 v59; // r14
  _BYTE *v60; // rax
  _BYTE *v61; // rax
  _BYTE *v62; // rax
  _BYTE *v63; // rdx
  __int64 v64; // rdi
  unsigned __int8 v65; // r14
  __int64 v66; // rdx
  __int64 v67; // rdi
  __int64 v68; // rdi
  _DWORD *v69; // rdx
  __int64 v70; // rdi
  _BYTE *v71; // rax
  __int64 v72; // r14
  void *v73; // r15
  void *v74; // rax
  unsigned __int64 v75; // r14
  _BYTE *v76; // rdx
  void *v77; // rax
  __int64 *v78; // rsi
  char *v79; // rdx
  char *v80; // rax
  bool v81; // cf
  __int64 v82; // r15
  void *v83; // rax
  __int64 v84; // r14
  __int64 *v85; // rdi
  __int64 v86; // rdi
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  unsigned __int8 v91; // r14
  __int64 v92; // rax
  __int64 v93; // rax
  void *v94; // rax
  __int64 v95; // r14
  __int64 *v96; // rdi
  _BYTE *v97; // rax
  __int64 v98; // rdi
  __int64 v99; // rdi
  _BYTE *v100; // rax
  __int64 v101; // rdi
  __int64 v102; // rdi
  int v103; // r14d
  __int64 v104; // r15
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rsi
  __int64 v108; // rdi
  char *v109; // rax
  size_t v110; // rdx
  void *v111; // rdi
  size_t v112; // r14
  __int64 v113; // r15
  __int64 v114; // rax
  __int64 v115; // r15
  __int64 v116; // rdx
  __int64 v117; // rsi
  __int64 v118; // rdx
  __int64 v119; // rax
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // r14
  __int64 v124; // rax
  char v125; // al
  const char *v126; // r14
  const char *v127; // rsi
  __int64 v128; // r14
  __int64 v129; // rax
  __int64 v130; // rax
  __int64 v131; // rax
  __int64 v132; // rax
  __int64 v133; // rax
  __int64 v134; // rax
  __int64 v135; // r14
  const char *v136; // rsi
  __int64 v137; // rax
  int v139; // [rsp+8h] [rbp-68h]
  unsigned __int64 v140; // [rsp+8h] [rbp-68h]
  char v141[8]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v142; // [rsp+18h] [rbp-58h]
  char *v143; // [rsp+20h] [rbp-50h] BYREF
  size_t v144; // [rsp+28h] [rbp-48h]
  __int64 (__fastcall *v145)(char **, char **, __int64); // [rsp+30h] [rbp-40h]
  void (__fastcall *v146)(char **, __int64); // [rsp+38h] [rbp-38h]

  v4 = a2;
  v6 = *(_BYTE *)(a1 + 80);
  if ( (v6 & 2) != 0 )
  {
    v45 = *(_DWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v45 <= 3u )
    {
      sub_16E7EE0(a2, " nuw", 4u);
    }
    else
    {
      *v45 = 2004184608;
      *(_QWORD *)(a2 + 24) += 4LL;
    }
    v6 = *(_BYTE *)(a1 + 80);
  }
  if ( (v6 & 4) != 0 )
  {
    v44 = *(_DWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v44 <= 3u )
    {
      sub_16E7EE0(a2, " nsw", 4u);
    }
    else
    {
      *v44 = 2004053536;
      *(_QWORD *)(a2 + 24) += 4LL;
    }
    v6 = *(_BYTE *)(a1 + 80);
  }
  if ( (v6 & 8) != 0 )
  {
    v43 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v43) <= 5 )
    {
      sub_16E7EE0(a2, " exact", 6u);
    }
    else
    {
      *(_DWORD *)v43 = 1635280160;
      *(_WORD *)(v43 + 4) = 29795;
      *(_QWORD *)(a2 + 24) += 6LL;
    }
    v6 = *(_BYTE *)(a1 + 80);
  }
  if ( (v6 & 0x10) != 0 )
  {
    v42 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v42) <= 4 )
    {
      sub_16E7EE0(a2, " nnan", 5u);
    }
    else
    {
      *(_DWORD *)v42 = 1634627104;
      *(_BYTE *)(v42 + 4) = 110;
      *(_QWORD *)(a2 + 24) += 5LL;
    }
    v6 = *(_BYTE *)(a1 + 80);
  }
  if ( (v6 & 0x20) != 0 )
  {
    v41 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v41) <= 4 )
    {
      sub_16E7EE0(a2, " ninf", 5u);
    }
    else
    {
      *(_DWORD *)v41 = 1852403232;
      *(_BYTE *)(v41 + 4) = 102;
      *(_QWORD *)(a2 + 24) += 5LL;
    }
    v6 = *(_BYTE *)(a1 + 80);
  }
  if ( (v6 & 0x40) != 0 )
  {
    v40 = *(_DWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v40 <= 3u )
    {
      sub_16E7EE0(a2, " nsz", 4u);
    }
    else
    {
      *v40 = 2054385184;
      *(_QWORD *)(a2 + 24) += 4LL;
    }
    v6 = *(_BYTE *)(a1 + 80);
  }
  if ( v6 < 0 )
  {
    v39 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v39) <= 4 )
    {
      sub_16E7EE0(a2, " arcp", 5u);
    }
    else
    {
      *(_DWORD *)v39 = 1668440352;
      *(_BYTE *)(v39 + 4) = 112;
      *(_QWORD *)(a2 + 24) += 5LL;
    }
  }
  v7 = *(_BYTE *)(a1 + 81);
  if ( (v7 & 2) != 0 )
  {
    v38 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v38) <= 8 )
    {
      sub_16E7EE0(a2, " contract", 9u);
    }
    else
    {
      *(_BYTE *)(v38 + 8) = 116;
      *(_QWORD *)v38 = 0x636172746E6F6320LL;
      *(_QWORD *)(a2 + 24) += 9LL;
    }
    v7 = *(_BYTE *)(a1 + 81);
  }
  if ( (v7 & 4) != 0 )
  {
    v37 = *(_DWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v37 <= 3u )
    {
      sub_16E7EE0(a2, " afn", 4u);
    }
    else
    {
      *v37 = 1852203296;
      *(_QWORD *)(a2 + 24) += 4LL;
    }
    v7 = *(_BYTE *)(a1 + 81);
  }
  if ( (v7 & 8) != 0 )
  {
    v36 = *(_QWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v36 <= 7u )
    {
      sub_16E7EE0(a2, " reassoc", 8u);
    }
    else
    {
      *v36 = 0x636F737361657220LL;
      *(_QWORD *)(a2 + 24) += 8LL;
    }
    v7 = *(_BYTE *)(a1 + 81);
  }
  if ( (v7 & 1) != 0 )
  {
    v34 = *(__m128i **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v34 <= 0x10u )
    {
      sub_16E7EE0(a2, " vector-reduction", 0x11u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4308920);
      v34[1].m128i_i8[0] = 110;
      *v34 = si128;
      *(_QWORD *)(a2 + 24) += 17LL;
    }
  }
  v8 = *(_WORD *)(a1 + 24);
  if ( v8 < 0 )
  {
    v9 = *(_QWORD *)(a1 + 88);
    if ( *(_QWORD *)(a1 + 96) == v9 )
      goto LABEL_23;
    v10 = *(_BYTE **)(a2 + 24);
    if ( *(_BYTE **)(a2 + 16) == v10 )
    {
      sub_16E7EE0(a2, "<", 1u);
      v11 = *(_DWORD **)(a2 + 24);
    }
    else
    {
      *v10 = 60;
      v11 = (_DWORD *)(*(_QWORD *)(a2 + 24) + 1LL);
      *(_QWORD *)(a2 + 24) = v11;
    }
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v11 <= 3u )
    {
      sub_16E7EE0(a2, "Mem:", 4u);
    }
    else
    {
      *v11 = 980247885;
      *(_QWORD *)(a2 + 24) += 4LL;
    }
    v12 = *(__int64 **)(a1 + 88);
    v13 = *(__int64 **)(a1 + 96);
    if ( v12 != v13 )
    {
      while ( 1 )
      {
        v14 = *v12++;
        sub_20941C0(v4, v14, a3);
        if ( v13 == v12 )
          break;
        v15 = *(_BYTE **)(v4 + 24);
        if ( *(_BYTE **)(v4 + 16) == v15 )
        {
          sub_16E7EE0(v4, " ", 1u);
        }
        else
        {
          *v15 = 32;
          ++*(_QWORD *)(v4 + 24);
        }
      }
    }
    goto LABEL_48;
  }
  if ( v8 != 110 )
  {
    if ( v8 == 32 || v8 == 10 )
    {
      v71 = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)v71 >= *(_QWORD *)(a2 + 16) )
      {
        v72 = sub_16E7DE0(a2, 60);
      }
      else
      {
        v72 = a2;
        *(_QWORD *)(a2 + 24) = v71 + 1;
        *v71 = 60;
      }
      sub_16A95F0(*(_QWORD *)(a1 + 88) + 24LL, v72, 1);
      v9 = *(_QWORD *)(v72 + 24);
      if ( v9 >= *(_QWORD *)(v72 + 16) )
      {
        LOBYTE(v9) = sub_16E7DE0(v72, 62);
      }
      else
      {
        *(_QWORD *)(v72 + 24) = v9 + 1;
        *(_BYTE *)v9 = 62;
      }
      goto LABEL_23;
    }
    if ( v8 == 33 || v8 == 11 )
    {
      v73 = *(void **)(*(_QWORD *)(a1 + 88) + 32LL);
      v74 = sub_1698270();
      v75 = *(_QWORD *)(a2 + 16);
      v76 = *(_BYTE **)(a2 + 24);
      if ( v73 == v74 )
      {
        if ( (unsigned __int64)v76 >= v75 )
        {
          v82 = sub_16E7DE0(a2, 60);
        }
        else
        {
          v82 = a2;
          *(_QWORD *)(a2 + 24) = v76 + 1;
          *v76 = 60;
        }
        v94 = sub_16982C0();
        v95 = *(_QWORD *)(a1 + 88);
        v96 = (__int64 *)(v95 + 32);
        if ( *(void **)(v95 + 32) == v94 )
          v96 = (__int64 *)(*(_QWORD *)(v95 + 40) + 8LL);
        sub_169D890(v96);
      }
      else
      {
        v140 = *(_QWORD *)(a2 + 24);
        if ( v73 != sub_1698280() )
        {
          if ( v75 - v140 <= 8 )
          {
            sub_16E7EE0(a2, "<APFloat(", 9u);
          }
          else
          {
            *(_BYTE *)(v140 + 8) = 40;
            *(_QWORD *)v140 = 0x74616F6C4650413CLL;
            *(_QWORD *)(a2 + 24) += 9LL;
          }
          v77 = sub_16982C0();
          v78 = (__int64 *)(*(_QWORD *)(a1 + 88) + 32LL);
          if ( (void *)*v78 == v77 )
            sub_169D930((__int64)&v143, (__int64)v78);
          else
            sub_169D7E0((__int64)&v143, v78);
          sub_16A95F0((__int64)&v143, v4, 0);
          if ( (unsigned int)v144 > 0x40 && v143 )
            j_j___libc_free_0_0(v143);
          v79 = *(char **)(v4 + 24);
          v80 = *(char **)(v4 + 16);
          v81 = v80 == v79;
          v9 = v80 - v79;
          if ( v81 || v9 == 1 )
          {
            LOBYTE(v9) = sub_16E7EE0(v4, ")>", 2u);
          }
          else
          {
            *(_WORD *)v79 = 15913;
            *(_QWORD *)(v4 + 24) += 2LL;
          }
          goto LABEL_23;
        }
        if ( v140 >= v75 )
        {
          v82 = sub_16E7DE0(a2, 60);
        }
        else
        {
          v82 = a2;
          *(_QWORD *)(a2 + 24) = v140 + 1;
          *(_BYTE *)v140 = 60;
        }
        v83 = sub_16982C0();
        v84 = *(_QWORD *)(a1 + 88);
        v85 = (__int64 *)(v84 + 32);
        if ( *(void **)(v84 + 32) == v83 )
          v85 = (__int64 *)(*(_QWORD *)(v84 + 40) + 8LL);
        sub_169D8E0(v85);
      }
      v86 = sub_16E7B70(v82);
      v9 = *(_QWORD *)(v86 + 24);
      if ( v9 >= *(_QWORD *)(v86 + 16) )
      {
        LOBYTE(v9) = sub_16E7DE0(v86, 62);
      }
      else
      {
        *(_QWORD *)(v86 + 24) = v9 + 1;
        *(_BYTE *)v9 = 62;
      }
      goto LABEL_23;
    }
    if ( (unsigned __int16)(v8 - 12) <= 1u || (unsigned __int16)(v8 - 34) <= 1u )
    {
      v59 = *(_QWORD *)(a1 + 96);
      v60 = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)v60 >= *(_QWORD *)(a2 + 16) )
      {
        sub_16E7DE0(a2, 60);
      }
      else
      {
        *(_QWORD *)(a2 + 24) = v60 + 1;
        *v60 = 60;
      }
      sub_15537D0(*(_QWORD *)(a1 + 88), a2, 1, 0);
      v61 = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)v61 >= *(_QWORD *)(a2 + 16) )
      {
        sub_16E7DE0(a2, 62);
      }
      else
      {
        *(_QWORD *)(a2 + 24) = v61 + 1;
        *v61 = 62;
      }
      v62 = *(_BYTE **)(a2 + 16);
      v63 = *(_BYTE **)(a2 + 24);
      if ( v59 <= 0 )
      {
        if ( v63 == v62 )
        {
          v64 = sub_16E7EE0(a2, " ", 1u);
        }
        else
        {
          *v63 = 32;
          v64 = a2;
          ++*(_QWORD *)(a2 + 24);
        }
      }
      else if ( (unsigned __int64)(v62 - v63) <= 2 )
      {
        v64 = sub_16E7EE0(a2, " + ", 3u);
      }
      else
      {
        v63[2] = 32;
        v64 = a2;
        *(_WORD *)v63 = 11040;
        *(_QWORD *)(a2 + 24) += 3LL;
      }
      LOBYTE(v9) = sub_16E7AB0(v64, v59);
      v65 = *(_BYTE *)(a1 + 104);
      if ( v65 )
      {
        v66 = *(_QWORD *)(a2 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v66) <= 4 )
        {
          v67 = sub_16E7EE0(a2, " [TF=", 5u);
        }
        else
        {
          *(_DWORD *)v66 = 1179933472;
          v67 = a2;
          *(_BYTE *)(v66 + 4) = 61;
          *(_QWORD *)(a2 + 24) += 5LL;
        }
        v68 = sub_16E7A90(v67, v65);
        v9 = *(_QWORD *)(v68 + 24);
        if ( v9 >= *(_QWORD *)(v68 + 16) )
        {
          LOBYTE(v9) = sub_16E7DE0(v68, 93);
        }
        else
        {
          *(_QWORD *)(v68 + 24) = v9 + 1;
          *(_BYTE *)v9 = 93;
        }
      }
      goto LABEL_23;
    }
    switch ( v8 )
    {
      case 36:
      case 14:
        v97 = *(_BYTE **)(a2 + 24);
        if ( *(_BYTE **)(a2 + 16) == v97 )
        {
          v98 = sub_16E7EE0(a2, "<", 1u);
        }
        else
        {
          *v97 = 60;
          v98 = a2;
          ++*(_QWORD *)(a2 + 24);
        }
        v99 = sub_16E7AB0(v98, *(int *)(a1 + 84));
        v9 = *(_QWORD *)(v99 + 24);
        if ( *(_QWORD *)(v99 + 16) == v9 )
        {
          LOBYTE(v9) = sub_16E7EE0(v99, ">", 1u);
        }
        else
        {
          *(_BYTE *)v9 = 62;
          ++*(_QWORD *)(v99 + 24);
        }
        goto LABEL_23;
      case 15:
      case 37:
        v100 = *(_BYTE **)(a2 + 24);
        if ( *(_BYTE **)(a2 + 16) == v100 )
        {
          v101 = sub_16E7EE0(a2, "<", 1u);
        }
        else
        {
          *v100 = 60;
          v101 = a2;
          ++*(_QWORD *)(a2 + 24);
        }
        v102 = sub_16E7AB0(v101, *(int *)(a1 + 84));
        v9 = *(_QWORD *)(v102 + 24);
        if ( *(_QWORD *)(v102 + 16) == v9 )
        {
          LOBYTE(v9) = sub_16E7EE0(v102, ">", 1u);
        }
        else
        {
          *(_BYTE *)v9 = 62;
          ++*(_QWORD *)(v102 + 24);
        }
        v91 = *(_BYTE *)(a1 + 88);
        if ( !v91 )
          goto LABEL_23;
        break;
      case 38:
      case 16:
        v103 = *(_DWORD *)(a1 + 96) & 0x7FFFFFFF;
        if ( *(int *)(a1 + 96) < 0 )
        {
          v104 = sub_1263B40(a2, "<");
          (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 88) + 40LL))(*(_QWORD *)(a1 + 88), v104);
        }
        else
        {
          v104 = sub_1263B40(a2, "<");
          sub_155C2B0(*(_QWORD *)(a1 + 88), v104, 0);
        }
        sub_1263B40(v104, ">");
        if ( v103 )
        {
          v105 = sub_1263B40(a2, " + ");
          LOBYTE(v9) = sub_16E7AB0(v105, v103);
        }
        else
        {
          v106 = sub_1263B40(a2, " ");
          LOBYTE(v9) = sub_16E7AB0(v106, 0);
        }
LABEL_222:
        v91 = *(_BYTE *)(a1 + 104);
        if ( !v91 )
          goto LABEL_23;
        break;
      case 42:
        v87 = sub_1263B40(a2, "<");
        v88 = sub_16E7AB0(v87, *(int *)(a1 + 88));
        v89 = sub_1549FC0(v88, 0x2Bu);
        v90 = sub_16E7AB0(v89, *(_QWORD *)(a1 + 96));
        LOBYTE(v9) = sub_1263B40(v90, ">");
        v91 = *(_BYTE *)(a1 + 84);
        if ( !v91 )
          goto LABEL_23;
        break;
      case 5:
        sub_1263B40(a2, "<");
        v107 = *(_QWORD *)(a1 + 88);
        v108 = *(_QWORD *)(v107 + 40);
        if ( v108 )
        {
          v109 = (char *)sub_1649960(v108);
          v111 = *(void **)(v4 + 24);
          v112 = v110;
          if ( v110 > *(_QWORD *)(v4 + 16) - (_QWORD)v111 )
          {
            v113 = sub_16E7EE0(v4, v109, v110);
          }
          else
          {
            v113 = v4;
            if ( v110 )
            {
              memcpy(v111, v109, v110);
              *(_QWORD *)(v4 + 24) += v112;
            }
          }
          sub_1263B40(v113, " ");
          v107 = *(_QWORD *)(a1 + 88);
        }
        v114 = sub_16E7B40(v4, v107);
        LOBYTE(v9) = sub_1263B40(v114, ">");
        goto LABEL_23;
      case 8:
        v115 = sub_1549FC0(a2, 0x20u);
        if ( a3 )
          v116 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a3 + 32) + 16LL) + 112LL))(*(_QWORD *)(*(_QWORD *)(a3 + 32) + 16LL));
        else
          v116 = 0;
        v117 = *(unsigned int *)(a1 + 84);
        sub_1F4AA00((__int64 *)&v143, v117, v116, 0, 0);
        if ( !v145 )
          sub_4263D6(&v143, v117, v118);
        v146(&v143, v115);
        LOBYTE(v9) = (_BYTE)v145;
        if ( v145 )
          LOBYTE(v9) = v145(&v143, &v143, 3);
        goto LABEL_23;
      case 17:
      case 39:
        v121 = sub_1263B40(a2, "'");
        v122 = sub_1263B40(v121, *(const char **)(a1 + 88));
        LOBYTE(v9) = sub_1263B40(v122, "'");
        v91 = *(_BYTE *)(a1 + 96);
        if ( !v91 )
          goto LABEL_23;
        break;
      case 208:
      case 209:
        if ( *(_QWORD *)(a1 + 88) )
        {
          v119 = sub_1263B40(a2, "<");
          v120 = sub_16E7B40(v119, *(_QWORD *)(a1 + 88));
          LOBYTE(v9) = sub_1263B40(v120, ">");
        }
        else
        {
          LOBYTE(v9) = sub_1263B40(a2, "<null>");
        }
        goto LABEL_23;
      case 6:
        v123 = sub_1263B40(a2, ":");
        v124 = *(_QWORD *)(a1 + 96);
        v141[0] = *(_BYTE *)(a1 + 88);
        v142 = v124;
        sub_1F596C0((__int64)&v143, v141);
        sub_16E7EE0(v123, v143, v144);
        LOBYTE(v9) = sub_2240A30(&v143);
        goto LABEL_23;
      case 185:
        sub_1263B40(a2, "<");
        sub_20941C0(a2, *(_QWORD *)(a1 + 104), a3);
        v125 = (*(_BYTE *)(a1 + 27) >> 2) & 3;
        switch ( v125 )
        {
          case 2:
            sub_1263B40(a2, ", sext");
            break;
          case 3:
            sub_1263B40(a2, ", zext");
            break;
          case 1:
            sub_1263B40(a2, ", anyext");
            break;
          default:
            goto LABEL_259;
        }
        v127 = " from ";
LABEL_264:
        v128 = sub_1263B40(v4, v127);
        v129 = *(_QWORD *)(a1 + 96);
        v141[0] = *(_BYTE *)(a1 + 88);
        v142 = v129;
        sub_1F596C0((__int64)&v143, v141);
        sub_16E7EE0(v128, v143, v144);
        sub_2240A30(&v143);
LABEL_259:
        v126 = sub_2094430((*(_WORD *)(a1 + 26) >> 7) & 7);
        if ( *v126 )
        {
          v130 = sub_1263B40(v4, ", ");
          sub_1263B40(v130, v126);
        }
        goto LABEL_261;
      case 186:
        sub_1263B40(a2, "<");
        sub_20941C0(a2, *(_QWORD *)(a1 + 104), a3);
        v127 = ", trunc to ";
        if ( (*(_BYTE *)(a1 + 27) & 4) == 0 )
          goto LABEL_259;
        goto LABEL_264;
      default:
        LOBYTE(v9) = sub_20943E0(a1);
        if ( (_BYTE)v9 )
        {
          sub_1263B40(a2, "<");
          sub_20941C0(a2, *(_QWORD *)(a1 + 104), a3);
LABEL_261:
          LOBYTE(v9) = sub_1263B40(v4, ">");
          goto LABEL_23;
        }
        if ( v8 != 40 && v8 != 18 )
        {
          if ( v8 == 159 )
          {
            v131 = sub_1549FC0(a2, 0x5Bu);
            v132 = sub_16E7A90(v131, *(unsigned int *)(a1 + 84));
            v133 = sub_1263B40(v132, " -> ");
            v134 = sub_16E7A90(v133, *(unsigned int *)(a1 + 88));
            LOBYTE(v9) = sub_1549FC0(v134, 0x5Du);
          }
          goto LABEL_23;
        }
        v135 = *(_QWORD *)(a1 + 96);
        sub_1263B40(a2, "<");
        sub_15537D0(*(_QWORD *)(*(_QWORD *)(a1 + 88) - 48LL), a2, 0, 0);
        sub_1263B40(a2, ", ");
        sub_15537D0(*(_QWORD *)(*(_QWORD *)(a1 + 88) - 24LL), a2, 0, 0);
        sub_1263B40(a2, ">");
        v136 = " + ";
        if ( v135 <= 0 )
          v136 = " ";
        v137 = sub_1263B40(v4, v136);
        LOBYTE(v9) = sub_16E7AB0(v137, v135);
        goto LABEL_222;
    }
    v92 = sub_1263B40(v4, " [TF=");
    v93 = sub_16E7A90(v92, v91);
    LOBYTE(v9) = sub_1549FC0(v93, 0x5Du);
    goto LABEL_23;
  }
  v33 = *(_BYTE **)(a2 + 24);
  if ( *(_BYTE **)(a2 + 16) == v33 )
  {
    sub_16E7EE0(a2, "<", 1u);
  }
  else
  {
    *v33 = 60;
    ++*(_QWORD *)(a2 + 24);
  }
  v47 = *(_BYTE **)(a1 + 40);
  if ( *v47 )
    v48 = word_4308860[(unsigned __int8)(*v47 - 14)];
  else
    v48 = sub_1F58D30((__int64)v47);
  if ( v48 )
  {
    v49 = (unsigned int)(v48 - 1);
    v50 = 0;
    v51 = **(_DWORD **)(a1 + 88);
    while ( 1 )
    {
      if ( v51 >= 0 )
      {
        sub_16E7AB0(a2, v51);
      }
      else
      {
        v54 = *(_BYTE **)(a2 + 24);
        if ( *(_BYTE **)(a2 + 16) == v54 )
        {
          sub_16E7EE0(a2, (char *)"u", 1u);
        }
        else
        {
          *v54 = 117;
          ++*(_QWORD *)(a2 + 24);
        }
      }
      if ( v50 == v49 )
        break;
      v52 = v50 + 1;
      v51 = *(_DWORD *)(*(_QWORD *)(a1 + 88) + 4 * (v50 + 1));
      if ( (_DWORD)v50 != -1 )
      {
        v53 = *(_BYTE **)(a2 + 24);
        if ( *(_BYTE **)(a2 + 16) == v53 )
        {
          v139 = *(_DWORD *)(*(_QWORD *)(a1 + 88) + 4 * (v50 + 1));
          sub_16E7EE0(a2, ",", 1u);
          v51 = v139;
          v52 = v50 + 1;
        }
        else
        {
          *v53 = 44;
          ++*(_QWORD *)(a2 + 24);
        }
      }
      v50 = v52;
    }
  }
LABEL_48:
  v9 = *(_QWORD *)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) != v9 )
  {
    *(_BYTE *)v9 = 62;
    ++*(_QWORD *)(v4 + 24);
    if ( !byte_4FCF040 )
      return v9;
    goto LABEL_24;
  }
  LOBYTE(v9) = sub_16E7EE0(v4, ">", 1u);
LABEL_23:
  if ( !byte_4FCF040 )
    return v9;
LABEL_24:
  v16 = *(_DWORD *)(a1 + 64);
  if ( v16 )
  {
    v55 = *(_QWORD *)(v4 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v4 + 16) - v55) <= 5 )
    {
      v56 = sub_16E7EE0(v4, " [ORD=", 6u);
    }
    else
    {
      *(_DWORD *)v55 = 1380932384;
      v56 = v4;
      *(_WORD *)(v55 + 4) = 15684;
      *(_QWORD *)(v4 + 24) += 6LL;
    }
    v57 = sub_16E7A90(v56, v16);
    v58 = *(_BYTE **)(v57 + 24);
    if ( (unsigned __int64)v58 >= *(_QWORD *)(v57 + 16) )
    {
      sub_16E7DE0(v57, 93);
    }
    else
    {
      *(_QWORD *)(v57 + 24) = v58 + 1;
      *v58 = 93;
    }
  }
  if ( *(_DWORD *)(a1 + 28) != -1 )
  {
    v17 = *(_QWORD *)(v4 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v4 + 16) - v17) <= 4 )
    {
      v18 = sub_16E7EE0(v4, " [ID=", 5u);
    }
    else
    {
      *(_DWORD *)v17 = 1145658144;
      v18 = v4;
      *(_BYTE *)(v17 + 4) = 61;
      *(_QWORD *)(v4 + 24) += 5LL;
    }
    v19 = sub_16E7AB0(v18, *(int *)(a1 + 28));
    v20 = *(_BYTE **)(v19 + 24);
    if ( (unsigned __int64)v20 >= *(_QWORD *)(v19 + 16) )
    {
      sub_16E7DE0(v19, 93);
    }
    else
    {
      *(_QWORD *)(v19 + 24) = v20 + 1;
      *v20 = 93;
    }
  }
  LOWORD(v9) = *(_WORD *)(a1 + 24);
  if ( (unsigned __int16)(v9 - 10) > 1u )
  {
    LOWORD(v9) = v9 - 32;
    if ( (unsigned __int16)v9 > 1u )
    {
      v69 = *(_DWORD **)(v4 + 24);
      if ( *(_QWORD *)(v4 + 16) - (_QWORD)v69 <= 3u )
      {
        v70 = sub_16E7EE0(v4, "# D:", 4u);
      }
      else
      {
        *v69 = 977543203;
        v70 = v4;
        *(_QWORD *)(v4 + 24) += 4LL;
      }
      LOBYTE(v9) = sub_16E7AB0(v70, (*(_BYTE *)(a1 + 26) & 4) != 0);
    }
  }
  if ( a3 )
  {
    v9 = sub_15C70A0(a1 + 72);
    v21 = v9;
    if ( v9 )
    {
      v22 = *(_QWORD *)(v9 - 8LL * *(unsigned int *)(v9 + 8));
      if ( !v22 )
      {
        v46 = *(_QWORD *)(v4 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v4 + 16) - v46) <= 8 )
        {
          sub_16E7EE0(v4, "<unknown>", 9u);
          v28 = *(_BYTE **)(v4 + 24);
        }
        else
        {
          *(_BYTE *)(v46 + 8) = 62;
          *(_QWORD *)v46 = 0x6E776F6E6B6E753CLL;
          v28 = (_BYTE *)(*(_QWORD *)(v4 + 24) + 9LL);
          *(_QWORD *)(v4 + 24) = v28;
        }
        goto LABEL_41;
      }
      v23 = v22;
      if ( *(_BYTE *)v22 == 15 )
      {
        v24 = v22;
      }
      else
      {
        v23 = *(_QWORD *)(v22 - 8LL * *(unsigned int *)(v22 + 8));
        v24 = v23;
        if ( !v23 )
        {
LABEL_40:
          v28 = *(_BYTE **)(v4 + 24);
LABEL_41:
          if ( *(_QWORD *)(v4 + 16) <= (unsigned __int64)v28 )
          {
            v30 = sub_16E7DE0(v4, 58);
          }
          else
          {
            *(_QWORD *)(v4 + 24) = v28 + 1;
            *v28 = 58;
            v30 = v4;
          }
          LOBYTE(v9) = sub_16E7A90(v30, *(unsigned int *)(v21 + 4));
          v31 = *(_WORD *)(v21 + 2);
          if ( v31 )
          {
            v32 = *(_BYTE **)(v4 + 24);
            if ( (unsigned __int64)v32 >= *(_QWORD *)(v4 + 16) )
            {
              v4 = sub_16E7DE0(v4, 58);
            }
            else
            {
              *(_QWORD *)(v4 + 24) = v32 + 1;
              *v32 = 58;
            }
            LOBYTE(v9) = sub_16E7A90(v4, v31);
          }
          return v9;
        }
      }
      v25 = *(_QWORD *)(v24 - 8LL * *(unsigned int *)(v23 + 8));
      if ( v25 )
      {
        v26 = (void *)sub_161E970(v25);
        v28 = *(_BYTE **)(v4 + 24);
        v29 = v27;
        if ( *(_QWORD *)(v4 + 16) - (_QWORD)v28 >= v27 )
        {
          if ( v27 )
          {
            memcpy(v28, v26, v27);
            v28 = (_BYTE *)(v29 + *(_QWORD *)(v4 + 24));
            *(_QWORD *)(v4 + 24) = v28;
          }
          goto LABEL_41;
        }
        sub_16E7EE0(v4, (char *)v26, v27);
      }
      goto LABEL_40;
    }
  }
  return v9;
}
