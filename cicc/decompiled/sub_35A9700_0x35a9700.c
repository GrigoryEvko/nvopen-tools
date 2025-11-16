// Function: sub_35A9700
// Address: 0x35a9700
//
__int64 __fastcall sub_35A9700(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rdi
  __int64 v10; // rbx
  int v11; // eax
  __int64 v12; // rax
  _BYTE *v13; // rsi
  __int64 v14; // rax
  unsigned __int8 *v15; // r13
  int v16; // eax
  __int64 v17; // rax
  __m128i *v18; // rdx
  __int64 v19; // r10
  __m128i si128; // xmm0
  int v21; // esi
  int v22; // r9d
  __int64 v23; // rdx
  unsigned int v24; // eax
  _QWORD *v25; // r12
  __int64 v26; // rcx
  int *v27; // r12
  int v28; // esi
  int v29; // r11d
  __int64 v30; // rcx
  unsigned int v31; // edx
  _QWORD *v32; // rax
  __int64 v33; // rdi
  int *v34; // rbx
  unsigned __int64 *v35; // rsi
  unsigned __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rsi
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // rdi
  __int64 v42; // rax
  _QWORD *v43; // rdx
  __int64 v44; // rdi
  __int64 v45; // rax
  _QWORD *v46; // rdx
  __int64 v47; // rdi
  __int64 v48; // rdi
  _BYTE *v49; // rax
  __int64 v50; // rdi
  _BYTE *v51; // rsi
  __int64 v52; // rdx
  __int64 v53; // rcx
  unsigned __int64 v54; // r11
  __int64 v55; // rax
  unsigned int v56; // r10d
  unsigned int v57; // r9d
  unsigned int v58; // r8d
  __int64 v59; // r15
  unsigned int v60; // r13d
  unsigned int v61; // r12d
  unsigned int v62; // ebx
  __int64 v63; // rax
  __int64 (*v64)(void); // rdx
  __int64 v65; // rax
  unsigned __int64 v66; // rbx
  unsigned __int64 v67; // rdi
  int v69; // edx
  int v70; // ecx
  __int64 v71; // rdi
  __int64 v72; // [rsp+0h] [rbp-230h]
  _QWORD *v75; // [rsp+28h] [rbp-208h]
  _QWORD *v76; // [rsp+38h] [rbp-1F8h] BYREF
  __int64 v77; // [rsp+40h] [rbp-1F0h] BYREF
  __int64 v78; // [rsp+48h] [rbp-1E8h] BYREF
  unsigned __int64 v79; // [rsp+50h] [rbp-1E0h] BYREF
  _BYTE *v80; // [rsp+58h] [rbp-1D8h]
  _BYTE *v81; // [rsp+60h] [rbp-1D0h]
  __int64 v82[4]; // [rsp+70h] [rbp-1C0h] BYREF
  __int64 v83; // [rsp+90h] [rbp-1A0h] BYREF
  unsigned __int64 v84; // [rsp+98h] [rbp-198h]
  __int64 v85; // [rsp+A0h] [rbp-190h]
  unsigned int v86; // [rsp+A8h] [rbp-188h]
  __int64 v87; // [rsp+B0h] [rbp-180h] BYREF
  __int64 v88; // [rsp+B8h] [rbp-178h]
  __int64 v89; // [rsp+C0h] [rbp-170h]
  unsigned int v90; // [rsp+C8h] [rbp-168h]
  __int64 v91; // [rsp+D0h] [rbp-160h] BYREF
  unsigned __int64 v92; // [rsp+D8h] [rbp-158h]
  __int64 v93; // [rsp+E0h] [rbp-150h]
  unsigned __int64 v94; // [rsp+E8h] [rbp-148h]
  __int64 v95; // [rsp+F0h] [rbp-140h] BYREF
  unsigned __int64 v96; // [rsp+F8h] [rbp-138h]
  __int64 v97; // [rsp+100h] [rbp-130h]
  __int64 v98; // [rsp+108h] [rbp-128h]
  __int64 v99; // [rsp+118h] [rbp-118h]
  unsigned int v100; // [rsp+128h] [rbp-108h]
  __int64 v101; // [rsp+138h] [rbp-F8h]
  unsigned int v102; // [rsp+148h] [rbp-E8h]
  __int64 v103; // [rsp+160h] [rbp-D0h] BYREF
  __int64 v104; // [rsp+168h] [rbp-C8h]
  unsigned __int64 v105; // [rsp+170h] [rbp-C0h]
  __int64 v106; // [rsp+178h] [rbp-B8h]
  __int64 v107; // [rsp+180h] [rbp-B0h]
  __int64 v108; // [rsp+188h] [rbp-A8h]
  __int64 v109; // [rsp+190h] [rbp-A0h]
  __int64 v110; // [rsp+198h] [rbp-98h]
  __int64 v111; // [rsp+1A0h] [rbp-90h]
  __int64 v112; // [rsp+1A8h] [rbp-88h]
  int v113; // [rsp+1B8h] [rbp-78h] BYREF
  unsigned __int64 v114; // [rsp+1C0h] [rbp-70h]
  int *v115; // [rsp+1C8h] [rbp-68h]
  int *v116; // [rsp+1D0h] [rbp-60h]
  __int64 v117; // [rsp+1D8h] [rbp-58h]
  __int64 v118; // [rsp+1E0h] [rbp-50h]
  __int64 v119; // [rsp+1E8h] [rbp-48h]
  __int64 v120; // [rsp+1F0h] [rbp-40h]
  unsigned int v121; // [rsp+1F8h] [rbp-38h]

  v3 = *(__int64 **)(a1 + 8);
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_109:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_501EACC )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_109;
  }
  v72 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_501EACC)
      + 200;
  v6 = sub_2EA6400(a3);
  v7 = sub_C5F790(a3, (__int64)&unk_501EACC);
  v8 = sub_904010(v7, "--- ModuloScheduleTest running on BB#");
  v9 = (__int64 *)sub_CB59F0(v8, *((int *)v6 + 6));
  sub_904010((__int64)v9, "\n");
  v10 = v6[7];
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v76 = (_QWORD *)v10;
  v75 = v6 + 6;
  if ( (_QWORD *)v10 != v6 + 6 )
  {
    while ( 1 )
    {
      v11 = *(_DWORD *)(v10 + 44);
      if ( (v11 & 4) != 0 || (v11 & 8) == 0 )
      {
        v12 = (*(_QWORD *)(*(_QWORD *)(v10 + 16) + 24LL) >> 9) & 1LL;
      }
      else
      {
        v9 = (__int64 *)v10;
        LOBYTE(v12) = sub_2E88A90(v10, 512, 1);
      }
      if ( (_BYTE)v12 )
        goto LABEL_48;
      v103 = v10;
      v13 = v80;
      if ( v80 == v81 )
      {
        v9 = (__int64 *)&v79;
        sub_2E26050((__int64)&v79, v80, &v103);
      }
      else
      {
        if ( v80 )
        {
          *(_QWORD *)v80 = v10;
          v13 = v80;
        }
        v13 += 8;
        v80 = v13;
      }
      v14 = *(_QWORD *)(v10 + 48);
      v15 = (unsigned __int8 *)(v14 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v14 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_48;
      v16 = v14 & 7;
      if ( v16 != 2 )
      {
        if ( v16 != 3 )
          goto LABEL_48;
        if ( !v15[5] )
          goto LABEL_48;
        v15 = *(unsigned __int8 **)&v15[8 * v15[4] + 16 + 8 * (__int64)*(int *)v15];
        if ( !v15 )
          goto LABEL_48;
      }
      v17 = sub_C5F790((__int64)v9, (__int64)v13);
      v18 = *(__m128i **)(v17 + 32);
      v19 = v17;
      if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 0x1Du )
      {
        v19 = sub_CB6200(v17, "Parsing post-instr symbol for ", 0x1Eu);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_44E6380);
        qmemcpy(&v18[1], "tr symbol for ", 14);
        *v18 = si128;
        *(_QWORD *)(v17 + 32) += 30LL;
      }
      sub_2E91850(v10, v19, 1u, 0, 0, 1, 0);
      v21 = v90;
      v78 = v10;
      if ( !v90 )
        break;
      v22 = 1;
      v23 = 0;
      v24 = (v90 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v25 = (_QWORD *)(v88 + 16LL * v24);
      v26 = *v25;
      if ( v10 == *v25 )
      {
LABEL_20:
        v27 = (int *)(v25 + 1);
        goto LABEL_21;
      }
      while ( v26 != -4096 )
      {
        if ( v26 == -8192 && !v23 )
          v23 = (__int64)v25;
        v24 = (v90 - 1) & (v22 + v24);
        v25 = (_QWORD *)(v88 + 16LL * v24);
        v26 = *v25;
        if ( v10 == *v25 )
          goto LABEL_20;
        ++v22;
      }
      if ( !v23 )
        v23 = (__int64)v25;
      ++v87;
      v70 = v89 + 1;
      v103 = v23;
      if ( 4 * ((int)v89 + 1) >= 3 * v90 )
        goto LABEL_101;
      v71 = v10;
      if ( v90 - HIDWORD(v89) - v70 <= v90 >> 3 )
        goto LABEL_102;
LABEL_94:
      LODWORD(v89) = v70;
      if ( *(_QWORD *)v23 != -4096 )
        --HIDWORD(v89);
      *(_QWORD *)v23 = v71;
      v27 = (int *)(v23 + 8);
      *(_DWORD *)(v23 + 8) = 0;
LABEL_21:
      v28 = v86;
      v77 = v10;
      if ( !v86 )
      {
        ++v83;
        v103 = 0;
        goto LABEL_104;
      }
      v29 = 1;
      v30 = 0;
      v31 = (v86 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v32 = (_QWORD *)(v84 + 16LL * v31);
      v33 = *v32;
      if ( v10 != *v32 )
      {
        while ( v33 != -4096 )
        {
          if ( !v30 && v33 == -8192 )
            v30 = (__int64)v32;
          v31 = (v86 - 1) & (v29 + v31);
          v32 = (_QWORD *)(v84 + 16LL * v31);
          v33 = *v32;
          if ( v10 == *v32 )
            goto LABEL_23;
          ++v29;
        }
        if ( !v30 )
          v30 = (__int64)v32;
        ++v83;
        v69 = v85 + 1;
        v103 = v30;
        if ( 4 * ((int)v85 + 1) < 3 * v86 )
        {
          if ( v86 - HIDWORD(v85) - v69 > v86 >> 3 )
          {
LABEL_81:
            LODWORD(v85) = v69;
            if ( *(_QWORD *)v30 != -4096 )
              --HIDWORD(v85);
            *(_QWORD *)v30 = v10;
            v34 = (int *)(v30 + 8);
            *(_DWORD *)(v30 + 8) = 0;
            goto LABEL_24;
          }
LABEL_105:
          sub_354C5D0((__int64)&v83, v28);
          sub_3546FB0((__int64)&v83, &v77, &v103);
          v10 = v77;
          v30 = v103;
          v69 = v85 + 1;
          goto LABEL_81;
        }
LABEL_104:
        v28 = 2 * v86;
        goto LABEL_105;
      }
LABEL_23:
      v34 = (int *)(v32 + 1);
LABEL_24:
      if ( (v15[8] & 1) != 0 )
      {
        v35 = (unsigned __int64 *)*((_QWORD *)v15 - 1);
        v36 = *v35;
        v37 = (__int64)(v35 + 3);
      }
      else
      {
        v36 = 0;
        v37 = 0;
      }
      sub_C92270(&v91, v37, v36, (__int64)"_", 1);
      sub_C92270(&v95, v91, v92, (__int64)"-", 1);
      sub_C92270(&v103, v93, v94, (__int64)"-", 1);
      if ( v96 != 5
        || *(_DWORD *)v95 != 1734440019
        || *(_BYTE *)(v95 + 4) != 101
        || v104 != 6
        || *(_DWORD *)v103 != 1668891487
        || *(_WORD *)(v103 + 4) != 25964 )
      {
        BUG();
      }
      v38 = v98;
      v39 = v97;
      if ( v98 )
      {
        v38 = v98 - 1;
        v39 = v97 + 1;
      }
      if ( !sub_C93CC0(v39, v38, 0xAu, v82) && v82[0] == SLODWORD(v82[0]) )
        *v27 = v82[0];
      v40 = v106;
      v41 = v105;
      if ( v106 )
      {
        v40 = v106 - 1;
        v41 = v105 + 1;
      }
      if ( !sub_C93CC0(v41, v40, 0xAu, v82) && v82[0] == SLODWORD(v82[0]) )
        *v34 = v82[0];
      v42 = sub_C5F790(v41, v40);
      v43 = *(_QWORD **)(v42 + 32);
      v44 = v42;
      if ( *(_QWORD *)(v42 + 24) - (_QWORD)v43 <= 7u )
      {
        v44 = sub_CB6200(v42, "  Stage=", 8u);
      }
      else
      {
        *v43 = 0x3D65676174532020LL;
        *(_QWORD *)(v42 + 32) += 8LL;
      }
      v45 = sub_CB59F0(v44, *v27);
      v46 = *(_QWORD **)(v45 + 32);
      v47 = v45;
      if ( *(_QWORD *)(v45 + 24) - (_QWORD)v46 <= 7u )
      {
        v47 = sub_CB6200(v45, ", Cycle=", 8u);
      }
      else
      {
        *v46 = 0x3D656C637943202CLL;
        *(_QWORD *)(v45 + 32) += 8LL;
      }
      v48 = sub_CB59F0(v47, *v34);
      v49 = *(_BYTE **)(v48 + 32);
      if ( *(_BYTE **)(v48 + 24) == v49 )
      {
        sub_CB6200(v48, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v49 = 10;
        ++*(_QWORD *)(v48 + 32);
      }
LABEL_48:
      v9 = (__int64 *)&v76;
      sub_2FD79B0((__int64 *)&v76);
      v10 = (__int64)v76;
      if ( v76 == v75 )
      {
        v50 = v79;
        v51 = v80;
        v52 = v87 + 1;
        v53 = (__int64)v81;
        v54 = v84;
        v55 = v83 + 1;
        v56 = v85;
        v57 = HIDWORD(v85);
        v58 = v86;
        v59 = v88;
        v60 = v89;
        v61 = HIDWORD(v89);
        v62 = v90;
        goto LABEL_50;
      }
    }
    ++v87;
    v103 = 0;
LABEL_101:
    v21 = 2 * v90;
LABEL_102:
    sub_354C5D0((__int64)&v87, v21);
    sub_3546FB0((__int64)&v87, &v78, &v103);
    v71 = v78;
    v23 = v103;
    v70 = v89 + 1;
    goto LABEL_94;
  }
  v62 = 0;
  v61 = 0;
  v60 = 0;
  v59 = 0;
  v52 = 1;
  v58 = 0;
  v57 = 0;
  v56 = 0;
  v54 = 0;
  v55 = 1;
  v53 = 0;
  v51 = 0;
  v50 = 0;
LABEL_50:
  v87 = v52;
  LODWORD(v94) = v58;
  v82[0] = v50;
  v82[1] = (__int64)v51;
  v82[2] = v53;
  v103 = 1;
  v104 = v59;
  v105 = __PAIR64__(v61, v60);
  LODWORD(v106) = v62;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 1;
  v83 = v55;
  v92 = v54;
  v93 = __PAIR64__(v57, v56);
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v81 = 0;
  v80 = 0;
  v79 = 0;
  sub_3598BB0((__int64)&v95, a2, a3, (__int64)v82, (__int64)&v91, (__int64)&v103);
  if ( v82[0] )
    j_j___libc_free_0(v82[0]);
  sub_C7D6A0(v92, 16LL * (unsigned int)v94, 8);
  sub_C7D6A0(v104, 16LL * (unsigned int)v106, 8);
  v103 = (__int64)&v95;
  v104 = a2;
  v63 = *(_QWORD *)(a2 + 32);
  v105 = *(_QWORD *)(a2 + 16);
  v106 = v63;
  v64 = *(__int64 (**)(void))(*(_QWORD *)v105 + 128LL);
  v65 = 0;
  if ( v64 != sub_2DAC790 )
    v65 = v64();
  v107 = v65;
  v109 = 0;
  v110 = 0;
  v108 = v72;
  v115 = &v113;
  v116 = &v113;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v117 = 0;
  v118 = 1;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  sub_C7D6A0(0, 0, 8);
  sub_35A93B0(&v103);
  sub_3598EB0((__int64)&v103);
  sub_C7D6A0(v119, 24LL * v121, 8);
  v66 = v114;
  while ( v66 )
  {
    sub_3598580(*(_QWORD *)(v66 + 24));
    v67 = v66;
    v66 = *(_QWORD *)(v66 + 16);
    j_j___libc_free_0(v67);
  }
  if ( v112 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v112 + 8LL))(v112);
  sub_C7D6A0(v101, 16LL * v102, 8);
  sub_C7D6A0(v99, 16LL * v100, 8);
  if ( v96 )
    j_j___libc_free_0(v96);
  if ( v79 )
    j_j___libc_free_0(v79);
  sub_C7D6A0(v88, 16LL * v90, 8);
  return sub_C7D6A0(v84, 16LL * v86, 8);
}
