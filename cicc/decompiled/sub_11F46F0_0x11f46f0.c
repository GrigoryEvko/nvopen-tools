// Function: sub_11F46F0
// Address: 0x11f46f0
//
__int64 __fastcall sub_11F46F0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned __int64 v8; // rax
  int v9; // edx
  __int64 v10; // rax
  bool v11; // cf
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int8 *v14; // r14
  __int64 v15; // rdx
  char *v16; // rsi
  __int64 v17; // rdx
  char *v18; // rsi
  __int64 v19; // rcx
  __m128i *v20; // rdx
  void **v21; // r9
  __m128i *v22; // rax
  unsigned int v23; // eax
  bool v24; // zf
  __m128i *v25; // rax
  __int64 v26; // rcx
  __m128i *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rax
  double v31; // xmm0_8
  double v32; // xmm0_8
  unsigned __int64 v33; // rax
  __m128i *v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned __int8 v38; // dl
  __int64 v39; // rcx
  __m128i *v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r8
  unsigned __int64 v45; // rcx
  unsigned __int64 v46; // rdx
  unsigned int v47; // edi
  unsigned __int64 v48; // rsi
  unsigned int v49; // eax
  __m128i *v50; // rdi
  unsigned __int64 v51; // rcx
  int v52; // esi
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  char v55; // r11
  __int64 v56; // r10
  __int64 v57; // rcx
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // rcx
  __m128i *v61; // rax
  _BYTE *v62; // rcx
  __m128i *v63; // rdx
  __m128i *v64; // rcx
  __m128i *v65; // rax
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rdi
  unsigned __int64 v68; // rcx
  __m128i *v69; // rax
  __int64 v70; // rcx
  __m128i *v71; // rdx
  __int64 v72; // [rsp+48h] [rbp-218h]
  unsigned __int64 v74; // [rsp+50h] [rbp-210h]
  __int64 v75; // [rsp+58h] [rbp-208h]
  unsigned __int64 v76; // [rsp+58h] [rbp-208h]
  unsigned __int64 v77; // [rsp+58h] [rbp-208h]
  char v78; // [rsp+58h] [rbp-208h]
  double v79; // [rsp+70h] [rbp-1F0h] BYREF
  double v80; // [rsp+78h] [rbp-1E8h] BYREF
  __m128i *v81; // [rsp+80h] [rbp-1E0h] BYREF
  __int64 v82; // [rsp+88h] [rbp-1D8h]
  __m128i v83; // [rsp+90h] [rbp-1D0h] BYREF
  __m128i *v84; // [rsp+A0h] [rbp-1C0h]
  __int64 v85; // [rsp+A8h] [rbp-1B8h]
  __m128i v86; // [rsp+B0h] [rbp-1B0h] BYREF
  __m128i *v87; // [rsp+C0h] [rbp-1A0h] BYREF
  __int64 v88; // [rsp+C8h] [rbp-198h]
  __m128i v89; // [rsp+D0h] [rbp-190h] BYREF
  __m128i *v90; // [rsp+E0h] [rbp-180h] BYREF
  void **v91; // [rsp+E8h] [rbp-178h]
  __m128i v92; // [rsp+F0h] [rbp-170h] BYREF
  _BYTE *v93; // [rsp+100h] [rbp-160h] BYREF
  __m128i *v94; // [rsp+108h] [rbp-158h]
  _BYTE v95[24]; // [rsp+110h] [rbp-150h] BYREF
  __m128i *v96; // [rsp+130h] [rbp-130h] BYREF
  __int64 v97; // [rsp+138h] [rbp-128h]
  __m128i v98; // [rsp+140h] [rbp-120h] BYREF
  __int64 v99; // [rsp+150h] [rbp-110h]
  __m128i v100; // [rsp+158h] [rbp-108h] BYREF
  void *v101; // [rsp+168h] [rbp-F8h]
  __m128i *v102; // [rsp+170h] [rbp-F0h]
  __int64 v103; // [rsp+178h] [rbp-E8h]
  __m128i v104; // [rsp+180h] [rbp-E0h] BYREF
  const char *v105; // [rsp+190h] [rbp-D0h] BYREF
  __int64 v106; // [rsp+198h] [rbp-C8h]
  void ***v107; // [rsp+1A0h] [rbp-C0h] BYREF
  __m128i v108; // [rsp+1A8h] [rbp-B8h] BYREF
  __int64 v109; // [rsp+1B8h] [rbp-A8h] BYREF
  __int64 *v110; // [rsp+1C0h] [rbp-A0h]
  void *v111; // [rsp+1C8h] [rbp-98h] BYREF
  double *v112; // [rsp+1D0h] [rbp-90h]
  void **v113; // [rsp+1D8h] [rbp-88h] BYREF
  __m128i v114; // [rsp+1E0h] [rbp-80h] BYREF
  void *v115; // [rsp+1F0h] [rbp-70h] BYREF
  __m128i *v116; // [rsp+1F8h] [rbp-68h]
  __int64 v117; // [rsp+200h] [rbp-60h]
  __m128i v118; // [rsp+208h] [rbp-58h] BYREF
  _QWORD v119[9]; // [rsp+218h] [rbp-48h] BYREF

  if ( !*(_BYTE *)(a4 + 41) )
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  v8 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8 == a2 + 48 )
  {
    v72 = 0;
    v13 = sub_B46EC0(0, a3);
  }
  else
  {
    if ( !v8 )
      BUG();
    v9 = *(unsigned __int8 *)(v8 - 24);
    v10 = v8 - 24;
    v11 = (unsigned int)(v9 - 30) < 0xB;
    v12 = 0;
    if ( v11 )
      v12 = v10;
    v72 = v12;
    v13 = sub_B46EC0(v12, a3);
  }
  v14 = (unsigned __int8 *)v13;
  v79 = (double)(int)sub_FF0430(*(_QWORD *)(a4 + 16), a2, v13) * 4.656612873077393e-10;
  v16 = (char *)sub_BD5D20((__int64)v14);
  if ( v16 )
  {
    v90 = &v92;
    sub_11F33F0((__int64 *)&v90, v16, (__int64)&v16[v15]);
    if ( v91 )
      goto LABEL_11;
  }
  else
  {
    v92.m128i_i8[0] = 0;
    v90 = &v92;
    v91 = 0;
  }
  v109 = 0x100000000LL;
  v106 = 0;
  v105 = (const char *)&unk_49DD210;
  v107 = 0;
  v110 = (__int64 *)&v90;
  v108 = 0u;
  sub_CB5980((__int64)&v105, 0, 0, 0);
  sub_A5BF40(v14, (__int64)&v105, 0, 0);
  sub_2240CE0(&v90, 0, 1);
  v105 = (const char *)&unk_49DD210;
  sub_CB5840((__int64)&v105);
LABEL_11:
  v18 = (char *)sub_BD5D20(a2);
  if ( v18 )
  {
    v87 = &v89;
    sub_11F33F0((__int64 *)&v87, v18, (__int64)&v18[v17]);
    v19 = v88;
    if ( v88 )
      goto LABEL_13;
  }
  else
  {
    v89.m128i_i8[0] = 0;
    v87 = &v89;
    v88 = 0;
  }
  v109 = 0x100000000LL;
  v106 = 0;
  v107 = 0;
  v108 = 0u;
  v110 = (__int64 *)&v87;
  v105 = (const char *)&unk_49DD210;
  sub_CB5980((__int64)&v105, 0, 0, 0);
  sub_A5BF40((unsigned __int8 *)a2, (__int64)&v105, 0, 0);
  sub_2240CE0(&v87, 0, 1);
  v105 = (const char *)&unk_49DD210;
  sub_CB5840((__int64)&v105);
  v19 = v88;
LABEL_13:
  v20 = v90;
  if ( v90 == &v92 )
  {
    v20 = &v108;
    v108 = _mm_load_si128(&v92);
  }
  else
  {
    v108.m128i_i64[0] = v92.m128i_i64[0];
  }
  v21 = v91;
  v92.m128i_i8[0] = 0;
  v91 = 0;
  v90 = &v92;
  v94 = (__m128i *)&v95[8];
  v93 = &unk_49E64B0;
  v22 = v87;
  if ( v87 == &v89 )
  {
    v22 = (__m128i *)&v95[8];
    *(__m128i *)&v95[8] = _mm_load_si128(&v89);
  }
  else
  {
    v94 = v87;
    *(_QWORD *)&v95[8] = v89.m128i_i64[0];
  }
  *(_QWORD *)v95 = v19;
  v88 = 0;
  v87 = &v89;
  v89.m128i_i8[0] = 0;
  if ( v20 == &v108 )
  {
    v20 = &v100;
    v100 = _mm_loadu_si128(&v108);
  }
  else
  {
    v100.m128i_i64[0] = v108.m128i_i64[0];
  }
  v101 = &unk_49E64B0;
  v102 = &v104;
  if ( v22 == (__m128i *)&v95[8] )
  {
    v22 = &v104;
    v104 = _mm_loadu_si128((const __m128i *)&v95[8]);
  }
  else
  {
    v102 = v22;
    v104.m128i_i64[0] = *(_QWORD *)&v95[8];
  }
  v103 = v19;
  v105 = "tooltip=\"{0} -> {1}\\nProbability {2:P}\" ";
  v107 = (void ***)v119;
  v106 = 40;
  v108.m128i_i8[8] = 1;
  v109 = (__int64)&unk_49E64E0;
  v108.m128i_i64[0] = 3;
  v110 = (__int64 *)&v79;
  v111 = &unk_49E64B0;
  v112 = (double *)&v114;
  if ( v20 == &v100 )
  {
    v114 = _mm_loadu_si128(&v100);
  }
  else
  {
    v112 = (double *)v20;
    v114.m128i_i64[0] = v100.m128i_i64[0];
  }
  v113 = v21;
  v115 = &unk_49E64B0;
  v116 = &v118;
  if ( v22 == &v104 )
  {
    v118 = _mm_load_si128(&v104);
  }
  else
  {
    v116 = v22;
    v118.m128i_i64[0] = v104.m128i_i64[0];
  }
  v117 = v19;
  v81 = &v83;
  v100.m128i_i64[0] = 0x100000000LL;
  v119[0] = &v115;
  v119[1] = &v111;
  v119[2] = &v109;
  v96 = (__m128i *)&unk_49DD210;
  v100.m128i_i64[1] = (__int64)&v81;
  v82 = 0;
  v83.m128i_i8[0] = 0;
  v97 = 0;
  v98 = 0u;
  v99 = 0;
  sub_CB5980((__int64)&v96, 0, 0, 0);
  sub_CB6840((__int64)&v96, (__int64)&v105);
  if ( v99 != v98.m128i_i64[0] )
    sub_CB5AE0((__int64 *)&v96);
  v96 = (__m128i *)&unk_49DD210;
  sub_CB5840((__int64)&v96);
  v115 = &unk_49E64B0;
  if ( v116 != &v118 )
    j_j___libc_free_0(v116, v118.m128i_i64[0] + 1);
  v111 = &unk_49E64B0;
  if ( v112 != (double *)&v114 )
    j_j___libc_free_0(v112, v114.m128i_i64[0] + 1);
  if ( v87 != &v89 )
    j_j___libc_free_0(v87, v89.m128i_i64[0] + 1);
  if ( v90 != &v92 )
    j_j___libc_free_0(v90, v92.m128i_i64[0] + 1);
  v23 = sub_B46E30(v72);
  if ( v23 == 1 )
  {
    v27 = v81;
    v28 = v82;
    *(_QWORD *)a1 = a1 + 16;
    sub_11F4570((__int64 *)a1, v27, (__int64)v27->m128i_i64 + v28);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8)) > 9 )
    {
      sub_2241490(a1, "penwidth=2", 10, v29);
LABEL_44:
      if ( v81 != &v83 )
        j_j___libc_free_0(v81, v83.m128i_i64[0] + 1);
      return a1;
    }
    goto LABEL_161;
  }
  if ( a3 < v23 )
  {
    v24 = *(_BYTE *)(a4 + 42) == 0;
    v80 = v79 + 1.0;
    if ( v24 )
    {
      v105 = "label=\"{0:P}\" penwidth={1}";
      v107 = &v113;
      v110 = (__int64 *)&v80;
      v108.m128i_i8[8] = 1;
      v109 = (__int64)&unk_49E64E0;
      v111 = &unk_49E64E0;
      v106 = 26;
      v112 = &v79;
      v108.m128i_i64[0] = 2;
      v113 = &v111;
      v93 = v95;
      v114.m128i_i64[0] = (__int64)&v109;
      v100.m128i_i64[0] = 0x100000000LL;
      v94 = 0;
      v95[0] = 0;
      v96 = (__m128i *)&unk_49DD210;
      v97 = 0;
      v98 = 0u;
      v99 = 0;
      v100.m128i_i64[1] = (__int64)&v93;
      sub_CB5980((__int64)&v96, 0, 0, 0);
      sub_CB6840((__int64)&v96, (__int64)&v105);
      if ( v99 != v98.m128i_i64[0] )
        sub_CB5AE0((__int64 *)&v96);
      v96 = (__m128i *)&unk_49DD210;
      sub_CB5840((__int64)&v96);
      v25 = (__m128i *)sub_2241130(&v93, 0, 0, v81, v82);
      *(_QWORD *)a1 = a1 + 16;
      if ( (__m128i *)v25->m128i_i64[0] == &v25[1] )
      {
        *(__m128i *)(a1 + 16) = _mm_loadu_si128(v25 + 1);
      }
      else
      {
        *(_QWORD *)a1 = v25->m128i_i64[0];
        *(_QWORD *)(a1 + 16) = v25[1].m128i_i64[0];
      }
      v26 = v25->m128i_i64[1];
      v25->m128i_i64[0] = (__int64)v25[1].m128i_i64;
      v25->m128i_i64[1] = 0;
      *(_QWORD *)(a1 + 8) = v26;
      v25[1].m128i_i8[0] = 0;
      if ( v93 != v95 )
        j_j___libc_free_0(v93, *(_QWORD *)v95 + 1LL);
      goto LABEL_44;
    }
    v30 = sub_FDD860(*(__int64 **)(a4 + 8), a2);
    if ( v30 < 0 )
      v31 = (double)(int)(v30 & 1 | ((unsigned __int64)v30 >> 1))
          + (double)(int)(v30 & 1 | ((unsigned __int64)v30 >> 1));
    else
      v31 = (double)(int)v30;
    v32 = v31 * v79;
    if ( v32 >= 9.223372036854776e18 )
      v33 = (unsigned int)(int)(v32 - 9.223372036854776e18) ^ 0x8000000000000000LL;
    else
      v33 = (unsigned int)(int)v32;
    v105 = "label=\"W:{0}\" penwidth={1}";
    v112 = (double *)v33;
    v107 = &v113;
    v113 = &v111;
    v109 = (__int64)&unk_49E64E0;
    v110 = (__int64 *)&v80;
    v114.m128i_i64[0] = (__int64)&v109;
    v100.m128i_i64[0] = 0x100000000LL;
    v111 = &unk_49E6510;
    v108.m128i_i8[8] = 1;
    v96 = (__m128i *)&unk_49DD210;
    v106 = 26;
    v108.m128i_i64[0] = 2;
    v93 = v95;
    v94 = 0;
    v95[0] = 0;
    v97 = 0;
    v98 = 0u;
    v99 = 0;
    v100.m128i_i64[1] = (__int64)&v93;
    sub_CB5980((__int64)&v96, 0, 0, 0);
    sub_CB6840((__int64)&v96, (__int64)&v105);
    if ( v99 != v98.m128i_i64[0] )
      sub_CB5AE0((__int64 *)&v96);
    v96 = (__m128i *)&unk_49DD210;
    sub_CB5840((__int64)&v96);
    v34 = (__m128i *)sub_2241130(&v93, 0, 0, v81, v82);
    v84 = &v86;
    if ( (__m128i *)v34->m128i_i64[0] == &v34[1] )
    {
      v86 = _mm_loadu_si128(v34 + 1);
    }
    else
    {
      v84 = (__m128i *)v34->m128i_i64[0];
      v86.m128i_i64[0] = v34[1].m128i_i64[0];
    }
    v35 = v34->m128i_i64[1];
    v34[1].m128i_i8[0] = 0;
    v85 = v35;
    v34->m128i_i64[0] = (__int64)v34[1].m128i_i64;
    v34->m128i_i64[1] = 0;
    if ( v93 != v95 )
      j_j___libc_free_0(v93, *(_QWORD *)v95 + 1LL);
    v36 = v85;
    if ( v85 )
    {
      *(_QWORD *)a1 = a1 + 16;
      if ( v84 == &v86 )
      {
        *(__m128i *)(a1 + 16) = _mm_load_si128(&v86);
      }
      else
      {
        *(_QWORD *)a1 = v84;
        *(_QWORD *)(a1 + 16) = v86.m128i_i64[0];
      }
      *(_QWORD *)(a1 + 8) = v36;
      goto LABEL_44;
    }
    v37 = sub_BC89C0(v72);
    if ( !v37 )
    {
      *(_QWORD *)a1 = a1 + 16;
      v40 = v81;
      if ( v81 != &v83 )
        goto LABEL_83;
      goto LABEL_91;
    }
    v38 = *(_BYTE *)(v37 - 16);
    v39 = a3 + 1;
    if ( (v38 & 2) != 0 )
    {
      if ( (unsigned int)v39 >= *(_DWORD *)(v37 - 24) )
        goto LABEL_82;
      v42 = *(_QWORD *)(v37 - 32);
    }
    else
    {
      if ( (unsigned int)v39 >= ((*(_WORD *)(v37 - 16) >> 6) & 0xFu) )
      {
LABEL_82:
        *(_QWORD *)a1 = a1 + 16;
        v40 = v81;
        if ( v81 != &v83 )
        {
LABEL_83:
          *(_QWORD *)a1 = v40;
          *(_QWORD *)(a1 + 16) = v83.m128i_i64[0];
LABEL_84:
          v41 = v82;
          v83.m128i_i8[0] = 0;
          v82 = 0;
          *(_QWORD *)(a1 + 8) = v41;
          v81 = &v83;
LABEL_85:
          if ( v84 != &v86 )
            j_j___libc_free_0(v84, v86.m128i_i64[0] + 1);
          goto LABEL_44;
        }
LABEL_91:
        *(__m128i *)(a1 + 16) = _mm_load_si128(&v83);
        goto LABEL_84;
      }
      v42 = v37 - 8LL * ((v38 >> 2) & 0xF) - 16;
    }
    v43 = *(_QWORD *)(v42 + 8 * v39);
    if ( *(_BYTE *)v43 != 1 || **(_BYTE **)(v43 + 136) != 17 )
    {
      *(_QWORD *)a1 = a1 + 16;
      v40 = v81;
      if ( v81 != &v83 )
        goto LABEL_83;
      goto LABEL_91;
    }
    v75 = *(_QWORD *)(v43 + 136);
    sub_11F4620(
      (__int64 *)&v105,
      (__int64 (__fastcall *)(_BYTE *, __int64, __int64, __va_list_tag *))&vsnprintf,
      328,
      (__int64)"%f",
      v80);
    v44 = v75;
    if ( *(_DWORD *)(v75 + 32) <= 0x40u )
      v45 = *(_QWORD *)(v75 + 24);
    else
      v45 = **(_QWORD **)(v75 + 24);
    if ( v45 > 9 )
    {
      if ( v45 <= 0x63 )
      {
        v77 = v45;
        v90 = &v92;
        sub_2240A50(&v90, 2, 0, v45, v44);
        v50 = v90;
        v51 = v77;
      }
      else
      {
        if ( v45 <= 0x3E7 )
        {
          v47 = 1;
LABEL_152:
          v47 += 2;
        }
        else if ( v45 <= 0x270F )
        {
          v47 = 1;
LABEL_150:
          v47 += 3;
        }
        else
        {
          v46 = v45;
          v47 = 1;
          v44 = 0x346DC5D63886594BLL;
          while ( 1 )
          {
            v48 = v46;
            v49 = v47;
            v47 += 4;
            v46 /= 0x2710u;
            if ( v48 <= 0x1869F )
              break;
            if ( v48 <= 0xF423F )
            {
              v74 = v45;
              v90 = &v92;
              sub_2240A50(&v90, v49 + 5, 0, &v92, 0x346DC5D63886594BLL);
              v50 = v90;
              v51 = v74;
              v52 = (_DWORD)v91 - 1;
              goto LABEL_108;
            }
            if ( v48 <= (unsigned __int64)&loc_98967F )
              goto LABEL_152;
            if ( v48 <= 0x5F5E0FF )
              goto LABEL_150;
          }
        }
        v76 = v45;
        v90 = &v92;
        sub_2240A50(&v90, v47, 0, v45, v44);
        v50 = v90;
        v51 = v76;
        v52 = (_DWORD)v91 - 1;
        do
        {
LABEL_108:
          v53 = v51
              - 20
              * (v51 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v51 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
          v54 = v51;
          v51 /= 0x64u;
          v55 = a00010203040506_0[2 * v53 + 1];
          LOBYTE(v53) = a00010203040506_0[2 * v53];
          v50->m128i_i8[v52] = v55;
          v56 = (unsigned int)(v52 - 1);
          v52 -= 2;
          v50->m128i_i8[v56] = v53;
        }
        while ( v54 > 0x270F );
        if ( v54 <= 0x3E7 )
          goto LABEL_110;
      }
      v50->m128i_i8[1] = a00010203040506_0[2 * v51 + 1];
      v50->m128i_i8[0] = a00010203040506_0[2 * v51];
LABEL_111:
      v87 = &v89;
      sub_11F4570((__int64 *)&v87, v81, (__int64)v81->m128i_i64 + v82);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v88) <= 8 )
        goto LABEL_161;
      sub_2241490(&v87, "label=\"W:", 9, v57);
      v58 = 15;
      v59 = 15;
      if ( v87 != &v89 )
        v59 = v89.m128i_i64[0];
      v60 = (unsigned __int64)v91 + v88;
      if ( (unsigned __int64)v91 + v88 <= v59 )
        goto LABEL_118;
      if ( v90 != &v92 )
        v58 = v92.m128i_i64[0];
      if ( v60 <= v58 )
      {
        v61 = (__m128i *)sub_2241130(&v90, 0, 0, v87, v88);
        v93 = v95;
        v62 = (_BYTE *)v61->m128i_i64[0];
        v63 = v61 + 1;
        if ( (__m128i *)v61->m128i_i64[0] != &v61[1] )
          goto LABEL_119;
      }
      else
      {
LABEL_118:
        v61 = (__m128i *)sub_2241490(&v87, v90, v91, v60);
        v93 = v95;
        v62 = (_BYTE *)v61->m128i_i64[0];
        v63 = v61 + 1;
        if ( (__m128i *)v61->m128i_i64[0] != &v61[1] )
        {
LABEL_119:
          v93 = v62;
          *(_QWORD *)v95 = v61[1].m128i_i64[0];
          goto LABEL_120;
        }
      }
      *(__m128i *)v95 = _mm_loadu_si128(v61 + 1);
LABEL_120:
      v64 = (__m128i *)v61->m128i_i64[1];
      v94 = v64;
      v61->m128i_i64[0] = (__int64)v63;
      v61->m128i_i64[1] = 0;
      v61[1].m128i_i8[0] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)v94) > 0xA )
      {
        v65 = (__m128i *)sub_2241490(&v93, "\" penwidth=", 11, v64);
        v96 = &v98;
        if ( (__m128i *)v65->m128i_i64[0] == &v65[1] )
        {
          v98 = _mm_loadu_si128(v65 + 1);
        }
        else
        {
          v96 = (__m128i *)v65->m128i_i64[0];
          v98.m128i_i64[0] = v65[1].m128i_i64[0];
        }
        v97 = v65->m128i_i64[1];
        v65->m128i_i64[0] = (__int64)v65[1].m128i_i64;
        v65->m128i_i64[1] = 0;
        v65[1].m128i_i8[0] = 0;
        v66 = 15;
        v67 = 15;
        if ( v96 != &v98 )
          v67 = v98.m128i_i64[0];
        v68 = v97 + v106;
        if ( v97 + v106 <= v67 )
          goto LABEL_129;
        if ( v105 != (const char *)&v107 )
          v66 = (unsigned __int64)v107;
        if ( v68 <= v66 )
        {
          v69 = (__m128i *)sub_2241130(&v105, 0, 0, v96, v97);
          *(_QWORD *)a1 = a1 + 16;
          v70 = v69->m128i_i64[0];
          v71 = v69 + 1;
          if ( (__m128i *)v69->m128i_i64[0] != &v69[1] )
            goto LABEL_130;
        }
        else
        {
LABEL_129:
          v69 = (__m128i *)sub_2241490(&v96, v105, v106, v68);
          *(_QWORD *)a1 = a1 + 16;
          v70 = v69->m128i_i64[0];
          v71 = v69 + 1;
          if ( (__m128i *)v69->m128i_i64[0] != &v69[1] )
          {
LABEL_130:
            *(_QWORD *)a1 = v70;
            *(_QWORD *)(a1 + 16) = v69[1].m128i_i64[0];
LABEL_131:
            *(_QWORD *)(a1 + 8) = v69->m128i_i64[1];
            v69->m128i_i64[0] = (__int64)v71;
            v69->m128i_i64[1] = 0;
            v69[1].m128i_i8[0] = 0;
            if ( v96 != &v98 )
              j_j___libc_free_0(v96, v98.m128i_i64[0] + 1);
            if ( v93 != v95 )
              j_j___libc_free_0(v93, *(_QWORD *)v95 + 1LL);
            if ( v87 != &v89 )
              j_j___libc_free_0(v87, v89.m128i_i64[0] + 1);
            if ( v90 != &v92 )
              j_j___libc_free_0(v90, v92.m128i_i64[0] + 1);
            if ( v105 != (const char *)&v107 )
              j_j___libc_free_0(v105, (char *)v107 + 1);
            goto LABEL_85;
          }
        }
        *(__m128i *)(a1 + 16) = _mm_loadu_si128(v69 + 1);
        goto LABEL_131;
      }
LABEL_161:
      sub_4262D8((__int64)"basic_string::append");
    }
    v78 = v45;
    v90 = &v92;
    sub_2240A50(&v90, 1, 0, v45, v44);
    v50 = v90;
    LOBYTE(v51) = v78;
LABEL_110:
    v50->m128i_i8[0] = v51 + 48;
    goto LABEL_111;
  }
  *(_QWORD *)a1 = a1 + 16;
  if ( v81 == &v83 )
  {
    *(__m128i *)(a1 + 16) = _mm_load_si128(&v83);
  }
  else
  {
    *(_QWORD *)a1 = v81;
    *(_QWORD *)(a1 + 16) = v83.m128i_i64[0];
  }
  *(_QWORD *)(a1 + 8) = v82;
  return a1;
}
