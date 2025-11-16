// Function: sub_1DD77E0
// Address: 0x1dd77e0
//
unsigned __int64 __fastcall sub_1DD77E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v5; // r13
  __int64 v6; // r12
  _BYTE *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // r14
  char *v13; // rax
  char *v14; // rdx
  __int64 v15; // r15
  char *v16; // rax
  size_t v17; // rdx
  void *v18; // rdi
  _WORD *v19; // rax
  char *v20; // rsi
  void *v21; // rdx
  char *v22; // rsi
  void *v23; // rdx
  char *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 (*v27)(void); // rax
  __int64 (*v28)(void); // rax
  _BYTE *v29; // rax
  __int64 v30; // rax
  void *v31; // rdx
  __int64 *v32; // r12
  __int64 v33; // rsi
  char **v34; // rdi
  __int64 v35; // rdx
  _BYTE *v36; // rax
  __int64 v37; // r15
  int v38; // eax
  __int64 v39; // rcx
  int v40; // r8d
  int v41; // r9d
  __int64 v42; // rdi
  _BYTE *v43; // rax
  _WORD *v44; // rdx
  int v45; // eax
  _BYTE *v46; // rax
  _BYTE *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  unsigned __int16 *v50; // r12
  __int64 v51; // rcx
  __int64 v52; // r8
  int v53; // r9d
  __int64 v54; // rdx
  __int64 v55; // rsi
  _WORD *v56; // rdx
  _BYTE *v57; // rax
  unsigned __int64 v58; // r14
  unsigned __int64 result; // rax
  char v60; // r12
  unsigned int v61; // edx
  unsigned int v62; // r8d
  int v63; // esi
  __int64 v64; // rdi
  unsigned int v65; // eax
  __int64 v66; // rcx
  unsigned __int64 j; // rax
  unsigned int v68; // esi
  __int64 *v69; // rcx
  __int64 v70; // r9
  _BYTE *v71; // rax
  __int64 v72; // rax
  _WORD *v73; // rdx
  _WORD *v74; // rdx
  _BYTE *v75; // rax
  __m128i *v76; // rdx
  __int64 *v77; // r12
  __int64 *v78; // r14
  _WORD *v79; // rdx
  _BYTE *v80; // rax
  __int64 v81; // rax
  char *v82; // rdx
  __int64 v83; // rdi
  char *v84; // rax
  bool v85; // cf
  _BYTE *v86; // rax
  __int64 v87; // rax
  __m128i *v88; // rdx
  __int64 v89; // rdi
  __m128i v90; // xmm0
  __int64 v91; // rdi
  _WORD *v92; // rdx
  __int64 *v93; // r12
  __int64 v94; // rdx
  __int64 v95; // rdi
  double v96; // xmm0_8
  double v97; // xmm1_8
  __int64 v98; // rdi
  _BYTE *v99; // rax
  int *v100; // r14
  _WORD *v101; // rdx
  __int64 v102; // rcx
  int v103; // r8d
  int v104; // r9d
  _BYTE *v105; // rax
  int v106; // ecx
  __m128i *v107; // rdx
  __m128i si128; // xmm0
  __m128i *v109; // rdx
  __int64 v110; // rdx
  __int64 v111; // rax
  int v112; // r10d
  __int64 v113; // [rsp+0h] [rbp-F0h]
  __int64 v114; // [rsp+8h] [rbp-E8h]
  __int64 v118; // [rsp+28h] [rbp-C8h]
  char v120; // [rsp+38h] [rbp-B8h]
  __int64 *v121; // [rsp+38h] [rbp-B8h]
  unsigned __int16 *v122; // [rsp+38h] [rbp-B8h]
  size_t v123; // [rsp+38h] [rbp-B8h]
  __int64 *i; // [rsp+38h] [rbp-B8h]
  _QWORD v125[2]; // [rsp+40h] [rbp-B0h] BYREF
  void (__fastcall *v126)(_QWORD *, _QWORD *, __int64); // [rsp+50h] [rbp-A0h]
  void (__fastcall *v127)(_QWORD *, __int64); // [rsp+58h] [rbp-98h]
  _QWORD v128[2]; // [rsp+60h] [rbp-90h] BYREF
  void (__fastcall *v129)(_QWORD *, _QWORD *, __int64); // [rsp+70h] [rbp-80h]
  void (__fastcall *v130)(_QWORD *, __int64); // [rsp+78h] [rbp-78h]
  _QWORD v131[2]; // [rsp+80h] [rbp-70h] BYREF
  void (__fastcall *v132)(_QWORD *, _QWORD *, __int64); // [rsp+90h] [rbp-60h]
  void (__fastcall *v133)(_QWORD *, __int64); // [rsp+98h] [rbp-58h]
  double *v134; // [rsp+A0h] [rbp-50h] BYREF
  char *v135; // [rsp+A8h] [rbp-48h]
  double v136; // [rsp+B0h] [rbp-40h] BYREF
  __int64 (__fastcall *v137)(int *, __int64, __int64, __int64, __int64, int); // [rsp+B8h] [rbp-38h]

  v5 = a2;
  v6 = *(_QWORD *)(a1 + 56);
  if ( !v6 )
  {
    v109 = *(__m128i **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v109 <= 0x3Fu )
    {
      v111 = sub_16E7EE0(a2, "Can't print out MachineBasicBlock because parent MachineFunction", 0x40u);
      v110 = *(_QWORD *)(v111 + 24);
      v5 = v111;
    }
    else
    {
      *v109 = _mm_load_si128((const __m128i *)&xmmword_42E9C20);
      v109[1] = _mm_load_si128((const __m128i *)&xmmword_42E9C30);
      v109[2] = _mm_load_si128((const __m128i *)&xmmword_42E9C40);
      v109[3] = _mm_load_si128((const __m128i *)&xmmword_42E9C50);
      v110 = *(_QWORD *)(a2 + 24) + 64LL;
      *(_QWORD *)(a2 + 24) = v110;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v110) <= 8 )
      return sub_16E7EE0(v5, " is null\n", 9u);
    *(_BYTE *)(v110 + 8) = 10;
    *(_QWORD *)v110 = 0x6C6C756E20736920LL;
    *(_QWORD *)(v5 + 24) += 9LL;
    return 0x6C6C756E20736920LL;
  }
  if ( a4 )
  {
    v134 = *(double **)(*(_QWORD *)(a4 + 392) + 16LL * *(unsigned int *)(a1 + 48));
    sub_1F10810(&v134, a2);
    v8 = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)v8 >= *(_QWORD *)(a2 + 16) )
    {
      sub_16E7DE0(a2, 9);
    }
    else
    {
      *(_QWORD *)(a2 + 24) = v8 + 1;
      *v8 = 9;
    }
  }
  v9 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v9) <= 2 )
  {
    v10 = sub_16E7EE0(a2, "bb.", 3u);
  }
  else
  {
    *(_BYTE *)(v9 + 2) = 46;
    v10 = a2;
    *(_WORD *)v9 = 25186;
    *(_QWORD *)(a2 + 24) += 3LL;
  }
  sub_16E7AB0(v10, *(int *)(a1 + 48));
  v12 = *(_QWORD *)(a1 + 40);
  if ( v12 )
  {
    v13 = *(char **)(a2 + 16);
    v14 = *(char **)(a2 + 24);
    if ( (*(_BYTE *)(v12 + 23) & 0x20) == 0 )
    {
      if ( (unsigned __int64)(v13 - v14) <= 1 )
      {
        sub_16E7EE0(a2, " (", 2u);
      }
      else
      {
        *(_WORD *)v14 = 10272;
        *(_QWORD *)(a2 + 24) += 2LL;
      }
      v45 = sub_154F480(a3, v12, (__int64)v14, v11);
      if ( v45 == -1 )
      {
        v107 = *(__m128i **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v107 <= 0x10u )
        {
          sub_16E7EE0(a2, "<ir-block badref>", 0x11u);
          v19 = *(_WORD **)(a2 + 24);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_42E9C60);
          v107[1].m128i_i8[0] = 62;
          *v107 = si128;
          v19 = (_WORD *)(*(_QWORD *)(a2 + 24) + 17LL);
          *(_QWORD *)(a2 + 24) = v19;
        }
      }
      else
      {
        LODWORD(v128[0]) = v45;
        LOWORD(v132) = 2563;
        v131[0] = "%ir-block.";
        v131[1] = v128[0];
        sub_16E2FC0((__int64 *)&v134, (__int64)v131);
        sub_16E7EE0(a2, (char *)v134, (size_t)v135);
        if ( v134 != &v136 )
          j_j___libc_free_0(v134, *(_QWORD *)&v136 + 1LL);
        v19 = *(_WORD **)(a2 + 24);
      }
      if ( !*(_BYTE *)(a1 + 181) )
        goto LABEL_20;
      v20 = ", ";
LABEL_16:
      if ( *(_QWORD *)(v5 + 16) - (_QWORD)v19 > 1u )
      {
        *v19 = *(_WORD *)v20;
        v21 = (void *)(*(_QWORD *)(v5 + 24) + 2LL);
        *(_QWORD *)(v5 + 24) = v21;
      }
      else
      {
        sub_16E7EE0(v5, v20, 2u);
        v21 = *(void **)(v5 + 24);
      }
      if ( *(_QWORD *)(v5 + 16) - (_QWORD)v21 <= 0xCu )
      {
        sub_16E7EE0(v5, "address-taken", 0xDu);
        v19 = *(_WORD **)(v5 + 24);
      }
      else
      {
        qmemcpy(v21, "address-taken", 13);
        v19 = (_WORD *)(*(_QWORD *)(v5 + 24) + 13LL);
        *(_QWORD *)(v5 + 24) = v19;
      }
LABEL_20:
      if ( !*(_BYTE *)(a1 + 180) )
      {
LABEL_26:
        if ( !*(_DWORD *)(a1 + 176) )
        {
LABEL_33:
          if ( *(_WORD **)(v5 + 16) == v19 )
          {
            sub_16E7EE0(v5, ")", 1u);
            v19 = *(_WORD **)(v5 + 24);
          }
          else
          {
            *(_BYTE *)v19 = 41;
            v19 = (_WORD *)(*(_QWORD *)(v5 + 24) + 1LL);
            *(_QWORD *)(v5 + 24) = v19;
          }
          goto LABEL_35;
        }
        v24 = ", ";
LABEL_28:
        if ( *(_QWORD *)(v5 + 16) - (_QWORD)v19 > 1u )
        {
          *v19 = *(_WORD *)v24;
          v25 = *(_QWORD *)(v5 + 24) + 2LL;
          *(_QWORD *)(v5 + 24) = v25;
        }
        else
        {
          sub_16E7EE0(v5, v24, 2u);
          v25 = *(_QWORD *)(v5 + 24);
        }
        if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v25) <= 5 )
        {
          v26 = sub_16E7EE0(v5, "align ", 6u);
        }
        else
        {
          *(_DWORD *)v25 = 1734962273;
          *(_WORD *)(v25 + 4) = 8302;
          v26 = v5;
          *(_QWORD *)(v5 + 24) += 6LL;
        }
        sub_16E7A90(v26, *(unsigned int *)(a1 + 176));
        v19 = *(_WORD **)(v5 + 24);
        goto LABEL_33;
      }
      v22 = ", ";
LABEL_22:
      if ( *(_QWORD *)(v5 + 16) - (_QWORD)v19 > 1u )
      {
        *v19 = *(_WORD *)v22;
        v23 = (void *)(*(_QWORD *)(v5 + 24) + 2LL);
        *(_QWORD *)(v5 + 24) = v23;
      }
      else
      {
        sub_16E7EE0(v5, v22, 2u);
        v23 = *(void **)(v5 + 24);
      }
      if ( *(_QWORD *)(v5 + 16) - (_QWORD)v23 <= 0xAu )
      {
        sub_16E7EE0(v5, "landing-pad", 0xBu);
        v19 = *(_WORD **)(v5 + 24);
      }
      else
      {
        qmemcpy(v23, "landing-pad", 11);
        v19 = (_WORD *)(*(_QWORD *)(v5 + 24) + 11LL);
        *(_QWORD *)(v5 + 24) = v19;
      }
      goto LABEL_26;
    }
    if ( v14 == v13 )
    {
      v15 = sub_16E7EE0(a2, ".", 1u);
    }
    else
    {
      *v14 = 46;
      v15 = a2;
      ++*(_QWORD *)(a2 + 24);
    }
    v16 = (char *)sub_1649960(v12);
    v18 = *(void **)(v15 + 24);
    if ( v17 > *(_QWORD *)(v15 + 16) - (_QWORD)v18 )
    {
      sub_16E7EE0(v15, v16, v17);
      v19 = *(_WORD **)(a2 + 24);
      if ( *(_BYTE *)(a1 + 181) )
        goto LABEL_15;
      goto LABEL_69;
    }
    if ( v17 )
    {
      v123 = v17;
      memcpy(v18, v16, v17);
      *(_QWORD *)(v15 + 24) += v123;
    }
  }
  v19 = *(_WORD **)(a2 + 24);
  if ( *(_BYTE *)(a1 + 181) )
  {
LABEL_15:
    v20 = " (";
    goto LABEL_16;
  }
LABEL_69:
  if ( *(_BYTE *)(a1 + 180) )
  {
    v22 = " (";
    goto LABEL_22;
  }
  if ( *(_DWORD *)(a1 + 176) )
  {
    v24 = " (";
    goto LABEL_28;
  }
LABEL_35:
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v19 <= 1u )
  {
    sub_16E7EE0(v5, ":\n", 2u);
  }
  else
  {
    *v19 = 2618;
    *(_QWORD *)(v5 + 24) += 2LL;
  }
  v113 = 0;
  v27 = *(__int64 (**)(void))(**(_QWORD **)(v6 + 16) + 112LL);
  if ( v27 != sub_1D00B10 )
    v113 = v27();
  v118 = 0;
  v114 = *(_QWORD *)(v6 + 40);
  v28 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(a1 + 56) + 16LL) + 40LL);
  if ( v28 != sub_1D00B00 )
    v118 = v28();
  v120 = a5 & (*(_QWORD *)(a1 + 64) != *(_QWORD *)(a1 + 72));
  if ( !v120 )
  {
LABEL_42:
    if ( *(_QWORD *)(a1 + 96) != *(_QWORD *)(a1 + 88) )
      goto LABEL_43;
    goto LABEL_152;
  }
  if ( a4 )
  {
    v75 = *(_BYTE **)(v5 + 24);
    if ( *(_QWORD *)(v5 + 16) <= (unsigned __int64)v75 )
    {
      sub_16E7DE0(v5, 9);
    }
    else
    {
      *(_QWORD *)(v5 + 24) = v75 + 1;
      *v75 = 9;
    }
  }
  v76 = *(__m128i **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v76 <= 0xFu )
  {
    sub_16E7EE0(v5, "; predecessors: ", 0x10u);
  }
  else
  {
    *v76 = _mm_load_si128((const __m128i *)&xmmword_42E9C70);
    *(_QWORD *)(v5 + 24) += 16LL;
  }
  v77 = *(__int64 **)(a1 + 64);
  v78 = *(__int64 **)(a1 + 72);
  if ( v77 != v78 )
  {
    while ( 1 )
    {
      v33 = *v77;
      v34 = (char **)v125;
      sub_1DD5B60(v125, *v77);
      if ( !v126 )
        break;
      v127(v125, v5);
      if ( v126 )
        v126(v125, v125, 3);
      if ( v78 == ++v77 )
        goto LABEL_150;
      if ( *(__int64 **)(a1 + 64) != v77 )
      {
        v79 = *(_WORD **)(v5 + 24);
        if ( *(_QWORD *)(v5 + 16) - (_QWORD)v79 <= 1u )
        {
          sub_16E7EE0(v5, ", ", 2u);
        }
        else
        {
          *v79 = 8236;
          *(_QWORD *)(v5 + 24) += 2LL;
        }
      }
    }
LABEL_229:
    sub_4263D6(v34, v33, v35);
  }
LABEL_150:
  v80 = *(_BYTE **)(v5 + 24);
  if ( (unsigned __int64)v80 >= *(_QWORD *)(v5 + 16) )
  {
    sub_16E7DE0(v5, 10);
    goto LABEL_42;
  }
  *(_QWORD *)(v5 + 24) = v80 + 1;
  *v80 = 10;
  if ( *(_QWORD *)(a1 + 96) != *(_QWORD *)(a1 + 88) )
  {
LABEL_43:
    if ( a4 )
    {
      v29 = *(_BYTE **)(v5 + 24);
      if ( (unsigned __int64)v29 >= *(_QWORD *)(v5 + 16) )
      {
        sub_16E7DE0(v5, 9);
      }
      else
      {
        *(_QWORD *)(v5 + 24) = v29 + 1;
        *v29 = 9;
      }
    }
    v30 = sub_16E8750(v5, 2u);
    v31 = *(void **)(v30 + 24);
    if ( *(_QWORD *)(v30 + 16) - (_QWORD)v31 <= 0xBu )
    {
      sub_16E7EE0(v30, "successors: ", 0xCu);
    }
    else
    {
      qmemcpy(v31, "successors: ", 12);
      *(_QWORD *)(v30 + 24) += 12LL;
    }
    v32 = *(__int64 **)(a1 + 88);
    v121 = *(__int64 **)(a1 + 96);
    if ( v121 != v32 )
    {
      while ( 1 )
      {
        v33 = *v32;
        v34 = (char **)v128;
        sub_1DD5B60(v128, *v32);
        if ( !v129 )
          goto LABEL_229;
        v130(v128, v5);
        if ( v129 )
          v129(v128, v128, 3);
        if ( *(_QWORD *)(a1 + 120) == *(_QWORD *)(a1 + 112) )
          goto LABEL_57;
        v36 = *(_BYTE **)(v5 + 24);
        if ( (unsigned __int64)v36 >= *(_QWORD *)(v5 + 16) )
        {
          v37 = sub_16E7DE0(v5, 40);
        }
        else
        {
          v37 = v5;
          *(_QWORD *)(v5 + 24) = v36 + 1;
          *v36 = 40;
        }
        v38 = sub_1DD75B0((_QWORD *)a1, (__int64)v32);
        v135 = "0x%08x";
        LODWORD(v136) = v38;
        v134 = (double *)&unk_49EFAC8;
        v42 = sub_16E8450(v37, (__int64)&v134, (__int64)&unk_49EFAC8, v39, v40, v41);
        v43 = *(_BYTE **)(v42 + 24);
        if ( (unsigned __int64)v43 < *(_QWORD *)(v42 + 16) )
          break;
        ++v32;
        sub_16E7DE0(v42, 41);
        if ( v121 == v32 )
          goto LABEL_74;
LABEL_58:
        if ( *(__int64 **)(a1 + 88) != v32 )
        {
          v44 = *(_WORD **)(v5 + 24);
          if ( *(_QWORD *)(v5 + 16) - (_QWORD)v44 <= 1u )
          {
            sub_16E7EE0(v5, ", ", 2u);
          }
          else
          {
            *v44 = 8236;
            *(_QWORD *)(v5 + 24) += 2LL;
          }
        }
      }
      *(_QWORD *)(v42 + 24) = v43 + 1;
      *v43 = 41;
LABEL_57:
      if ( v121 == ++v32 )
        goto LABEL_74;
      goto LABEL_58;
    }
LABEL_74:
    if ( *(_QWORD *)(a1 + 120) != *(_QWORD *)(a1 + 112) && a5 )
    {
      v92 = *(_WORD **)(v5 + 24);
      if ( *(_QWORD *)(v5 + 16) - (_QWORD)v92 <= 1u )
      {
        sub_16E7EE0(v5, "; ", 2u);
      }
      else
      {
        *v92 = 8251;
        *(_QWORD *)(v5 + 24) += 2LL;
      }
      v93 = *(__int64 **)(a1 + 88);
      for ( i = *(__int64 **)(a1 + 96); i != v93; ++v93 )
      {
        v100 = (int *)sub_1DD7590(a1, (__int64)v93);
        if ( *(__int64 **)(a1 + 88) != v93 )
        {
          v101 = *(_WORD **)(v5 + 24);
          if ( *(_QWORD *)(v5 + 16) - (_QWORD)v101 <= 1u )
          {
            sub_16E7EE0(v5, ", ", 2u);
          }
          else
          {
            *v101 = 8236;
            *(_QWORD *)(v5 + 24) += 2LL;
          }
        }
        v33 = *v93;
        v34 = (char **)v131;
        sub_1DD5B60(v131, *v93);
        if ( !v132 )
          goto LABEL_229;
        v133(v131, v5);
        v105 = *(_BYTE **)(v5 + 24);
        if ( (unsigned __int64)v105 < *(_QWORD *)(v5 + 16) )
        {
          v94 = (__int64)(v105 + 1);
          v95 = v5;
          *(_QWORD *)(v5 + 24) = v105 + 1;
          *v105 = 40;
        }
        else
        {
          v95 = sub_16E7DE0(v5, 40);
        }
        v96 = (double)*v100 * 4.656612873077393e-10 * 100.0 * 100.0;
        v97 = fabs(v96);
        if ( v97 < 4.503599627370496e15 )
          *(_QWORD *)&v96 = COERCE_UNSIGNED_INT64(v97 + 4.503599627370496e15 - 4.503599627370496e15)
                          | *(_QWORD *)&v96 & 0x8000000000000000LL;
        v136 = v96 / 100.0;
        v135 = "%.2f%%";
        v134 = (double *)&unk_49E8778;
        v98 = sub_16E8450(v95, (__int64)&v134, v94, v102, v103, v104);
        v99 = *(_BYTE **)(v98 + 24);
        if ( (unsigned __int64)v99 >= *(_QWORD *)(v98 + 16) )
        {
          sub_16E7DE0(v98, 41);
        }
        else
        {
          *(_QWORD *)(v98 + 24) = v99 + 1;
          *v99 = 41;
        }
        if ( v132 )
          v132(v131, v131, 3);
      }
    }
    v46 = *(_BYTE **)(v5 + 24);
    if ( (unsigned __int64)v46 >= *(_QWORD *)(v5 + 16) )
    {
      sub_16E7DE0(v5, 10);
    }
    else
    {
      *(_QWORD *)(v5 + 24) = v46 + 1;
      *v46 = 10;
    }
    if ( *(_QWORD *)(a1 + 160) == *(_QWORD *)(a1 + 152) || (**(_BYTE **)(*(_QWORD *)v114 + 352LL) & 4) == 0 )
      goto LABEL_98;
    goto LABEL_80;
  }
LABEL_152:
  if ( *(_QWORD *)(a1 + 160) == *(_QWORD *)(a1 + 152) || (**(_BYTE **)(*(_QWORD *)v114 + 352LL) & 4) == 0 )
  {
    if ( !v120 )
      goto LABEL_100;
    goto LABEL_98;
  }
LABEL_80:
  if ( a4 )
  {
    v47 = *(_BYTE **)(v5 + 24);
    if ( (unsigned __int64)v47 >= *(_QWORD *)(v5 + 16) )
    {
      sub_16E7DE0(v5, 9);
    }
    else
    {
      *(_QWORD *)(v5 + 24) = v47 + 1;
      *v47 = 9;
    }
  }
  v48 = sub_16E8750(v5, 2u);
  v49 = *(_QWORD *)(v48 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v48 + 16) - v49) <= 8 )
  {
    sub_16E7EE0(v48, "liveins: ", 9u);
  }
  else
  {
    *(_BYTE *)(v49 + 8) = 32;
    *(_QWORD *)v49 = 0x3A736E696576696CLL;
    *(_QWORD *)(v48 + 24) += 9LL;
  }
  v122 = *(unsigned __int16 **)(a1 + 160);
  v50 = (unsigned __int16 *)sub_1DD77D0(a1);
  if ( v122 != v50 )
  {
    while ( 1 )
    {
      v33 = *v50;
      v34 = (char **)&v134;
      sub_1F4AA00(&v134, v33, v113, 0, 0);
      if ( v136 == 0.0 )
        goto LABEL_229;
      ((void (__fastcall *)(double **, __int64))v137)(&v134, v5);
      if ( v136 != 0.0 )
        (*(void (__fastcall **)(double **, double **, __int64))&v136)(&v134, &v134, 3);
      if ( *((_DWORD *)v50 + 1) != -1 )
      {
        v54 = *(_QWORD *)(v5 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v54) <= 2 )
        {
          v55 = sub_16E7EE0(v5, ":0x", 3u);
        }
        else
        {
          *(_BYTE *)(v54 + 2) = 120;
          v55 = v5;
          *(_WORD *)v54 = 12346;
          *(_QWORD *)(v5 + 24) += 3LL;
        }
        LODWORD(v134) = *((_DWORD *)v50 + 1);
        v136 = COERCE_DOUBLE(sub_1DB3470);
        v137 = sub_1DB3430;
        sub_1DB3430((int *)&v134, v55, v54, v51, v52, v53);
        if ( v136 != 0.0 )
          (*(void (__fastcall **)(double **, double **, __int64))&v136)(&v134, &v134, 3);
      }
      v50 += 4;
      if ( v50 == v122 )
        break;
      v56 = *(_WORD **)(v5 + 24);
      if ( *(_QWORD *)(v5 + 16) - (_QWORD)v56 <= 1u )
      {
        sub_16E7EE0(v5, ", ", 2u);
      }
      else
      {
        *v56 = 8236;
        *(_QWORD *)(v5 + 24) += 2LL;
      }
    }
  }
LABEL_98:
  v57 = *(_BYTE **)(v5 + 24);
  if ( (unsigned __int64)v57 >= *(_QWORD *)(v5 + 16) )
  {
    sub_16E7DE0(v5, 10);
  }
  else
  {
    *(_QWORD *)(v5 + 24) = v57 + 1;
    *v57 = 10;
  }
LABEL_100:
  v58 = *(_QWORD *)(a1 + 32);
  result = a1 + 24;
  v60 = 0;
  if ( a1 + 24 == v58 )
    goto LABEL_117;
  do
  {
    if ( !a4 )
      goto LABEL_110;
    v61 = *(_DWORD *)(a4 + 384);
    if ( v61 )
    {
      v62 = v61 - 1;
      v63 = 1;
      v64 = *(_QWORD *)(a4 + 368);
      v65 = (v61 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v66 = *(_QWORD *)(v64 + 16LL * v65);
      if ( v66 != v58 )
      {
        while ( v66 != -8 )
        {
          v65 = v62 & (v63 + v65);
          v66 = *(_QWORD *)(v64 + 16LL * v65);
          if ( v66 == v58 )
            goto LABEL_104;
          ++v63;
        }
        v71 = *(_BYTE **)(v5 + 24);
        if ( (unsigned __int64)v71 >= *(_QWORD *)(v5 + 16) )
        {
LABEL_124:
          sub_16E7DE0(v5, 9);
          if ( !v60 )
            goto LABEL_125;
          goto LABEL_111;
        }
        goto LABEL_109;
      }
LABEL_104:
      for ( j = v58; (*(_BYTE *)(j + 46) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
        ;
      v68 = v62 & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
      v69 = (__int64 *)(v64 + 16LL * v68);
      v70 = *v69;
      if ( j != *v69 )
      {
        v106 = 1;
        while ( v70 != -8 )
        {
          v112 = v106 + 1;
          v68 = v62 & (v106 + v68);
          v69 = (__int64 *)(v64 + 16LL * v68);
          v70 = *v69;
          if ( *v69 == j )
            goto LABEL_107;
          v106 = v112;
        }
        v69 = (__int64 *)(v64 + 16LL * v61);
      }
LABEL_107:
      v134 = (double *)v69[1];
      sub_1F10810(&v134, v5);
    }
    v71 = *(_BYTE **)(v5 + 24);
    if ( (unsigned __int64)v71 >= *(_QWORD *)(v5 + 16) )
      goto LABEL_124;
LABEL_109:
    *(_QWORD *)(v5 + 24) = v71 + 1;
    *v71 = 9;
LABEL_110:
    if ( !v60 )
      goto LABEL_125;
LABEL_111:
    if ( (*(_BYTE *)(v58 + 46) & 4) != 0 )
    {
      sub_16E8750(v5, 4u);
      sub_1E181D0(v58, v5, a3, a5, 0, 0, 0, v118);
      result = *(_QWORD *)(v5 + 24);
      goto LABEL_113;
    }
    v72 = sub_16E8750(v5, 2u);
    v73 = *(_WORD **)(v72 + 24);
    if ( *(_QWORD *)(v72 + 16) - (_QWORD)v73 <= 1u )
    {
      sub_16E7EE0(v72, "}\n", 2u);
    }
    else
    {
      *v73 = 2685;
      *(_QWORD *)(v72 + 24) += 2LL;
    }
LABEL_125:
    sub_16E8750(v5, 2u);
    sub_1E181D0(v58, v5, a3, a5, 0, 0, 0, v118);
    if ( (*(_BYTE *)(v58 + 46) & 8) != 0 )
    {
      v74 = *(_WORD **)(v5 + 24);
      if ( *(_QWORD *)(v5 + 16) - (_QWORD)v74 <= 1u )
      {
        sub_16E7EE0(v5, " {", 2u);
        result = *(_QWORD *)(v5 + 24);
      }
      else
      {
        *v74 = 31520;
        result = *(_QWORD *)(v5 + 24) + 2LL;
        *(_QWORD *)(v5 + 24) = result;
      }
      v60 = 1;
LABEL_113:
      if ( *(_QWORD *)(v5 + 16) > result )
        goto LABEL_114;
      goto LABEL_127;
    }
    result = *(_QWORD *)(v5 + 24);
    v60 = 0;
    if ( *(_QWORD *)(v5 + 16) > result )
    {
LABEL_114:
      *(_QWORD *)(v5 + 24) = result + 1;
      *(_BYTE *)result = 10;
      goto LABEL_115;
    }
LABEL_127:
    result = sub_16E7DE0(v5, 10);
LABEL_115:
    v58 = *(_QWORD *)(v58 + 8);
  }
  while ( a1 + 24 != v58 );
  if ( v60 )
  {
    v81 = sub_16E8750(v5, 2u);
    v82 = *(char **)(v81 + 24);
    v83 = v81;
    v84 = *(char **)(v81 + 16);
    v85 = v84 == v82;
    result = v84 - v82;
    if ( v85 || result == 1 )
    {
      result = sub_16E7EE0(v83, "}\n", 2u);
    }
    else
    {
      *(_WORD *)v82 = 2685;
      *(_QWORD *)(v83 + 24) += 2LL;
    }
  }
LABEL_117:
  if ( *(_BYTE *)(a1 + 144) && a5 )
  {
    if ( a4 )
    {
      v86 = *(_BYTE **)(v5 + 24);
      if ( (unsigned __int64)v86 >= *(_QWORD *)(v5 + 16) )
      {
        sub_16E7DE0(v5, 9);
      }
      else
      {
        *(_QWORD *)(v5 + 24) = v86 + 1;
        *v86 = 9;
      }
    }
    v87 = sub_16E8750(v5, 2u);
    v88 = *(__m128i **)(v87 + 24);
    v89 = v87;
    if ( *(_QWORD *)(v87 + 16) - (_QWORD)v88 <= 0x21u )
    {
      v89 = sub_16E7EE0(v87, "; Irreducible loop header weight: ", 0x22u);
    }
    else
    {
      v90 = _mm_load_si128((const __m128i *)&xmmword_42E9C80);
      v88[2].m128i_i16[0] = 8250;
      *v88 = v90;
      v88[1] = _mm_load_si128((const __m128i *)&xmmword_42E9C90);
      *(_QWORD *)(v87 + 24) += 34LL;
    }
    v91 = sub_16E7A90(v89, *(_QWORD *)(a1 + 136));
    result = *(_QWORD *)(v91 + 24);
    if ( result >= *(_QWORD *)(v91 + 16) )
    {
      return sub_16E7DE0(v91, 10);
    }
    else
    {
      *(_QWORD *)(v91 + 24) = result + 1;
      *(_BYTE *)result = 10;
    }
  }
  return result;
}
