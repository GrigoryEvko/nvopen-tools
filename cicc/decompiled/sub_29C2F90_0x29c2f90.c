// Function: sub_29C2F90
// Address: 0x29c2f90
//
__int64 __fastcall sub_29C2F90(__int64 *a1, __int64 a2, __int64 a3, void *a4, size_t a5, __int64 a6)
{
  __int64 v9; // r14
  __m128i *v10; // rdi
  unsigned __int64 v11; // rax
  __int64 v13; // rax
  __int64 *v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rcx
  int v26; // r14d
  __int64 v27; // rax
  __int64 v28; // rax
  int v29; // ebx
  const char *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r13
  __int64 v33; // r12
  const char *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r13
  int v37; // esi
  __int64 v38; // rsi
  __int64 v39; // rdx
  unsigned __int8 *v40; // rsi
  __int64 v41; // r12
  _QWORD *v42; // rax
  __int64 v43; // r12
  __int64 v44; // rax
  unsigned __int64 v45; // rax
  __int64 v46; // rcx
  char v47; // r13
  char v48; // dh
  __int64 v49; // rbx
  _QWORD *v50; // r12
  char v51; // cl
  char v52; // r14
  __int64 v53; // r13
  int v54; // eax
  unsigned __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // r13
  __int64 v59; // r12
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rbx
  __int64 v67; // r14
  __int64 v68; // r13
  unsigned __int64 v69; // r12
  __int64 v70; // rsi
  char *v71; // rbx
  char *v72; // r12
  __int64 v73; // rsi
  unsigned __int64 v74; // rbx
  unsigned __int64 v75; // r12
  unsigned __int64 v76; // rdi
  char *v77; // rbx
  char *v78; // r12
  __int64 v79; // rsi
  char *v80; // rbx
  char *v81; // r12
  __int64 v82; // rsi
  char *v83; // rbx
  char *v84; // r12
  __int64 v85; // rsi
  __int64 v86; // rdi
  unsigned __int64 v87; // rax
  __int64 v88; // rax
  _BYTE *v89; // [rsp+0h] [rbp-300h]
  char v92; // [rsp+2Fh] [rbp-2D1h]
  __int64 v94; // [rsp+40h] [rbp-2C0h]
  __int64 v95; // [rsp+50h] [rbp-2B0h]
  __int64 v96; // [rsp+58h] [rbp-2A8h]
  __int64 v97; // [rsp+60h] [rbp-2A0h]
  int v98; // [rsp+60h] [rbp-2A0h]
  __int64 v99; // [rsp+68h] [rbp-298h]
  __int64 v100; // [rsp+68h] [rbp-298h]
  unsigned __int64 v101; // [rsp+68h] [rbp-298h]
  int v102; // [rsp+74h] [rbp-28Ch] BYREF
  __int64 v103; // [rsp+78h] [rbp-288h] BYREF
  __int64 v104; // [rsp+80h] [rbp-280h] BYREF
  __int64 v105; // [rsp+88h] [rbp-278h] BYREF
  _QWORD v106[4]; // [rsp+90h] [rbp-270h] BYREF
  __int64 v107[2]; // [rsp+B0h] [rbp-250h] BYREF
  __int64 v108; // [rsp+C0h] [rbp-240h]
  __int64 v109; // [rsp+D0h] [rbp-230h] BYREF
  __int64 v110; // [rsp+D8h] [rbp-228h]
  __int64 v111; // [rsp+E0h] [rbp-220h]
  unsigned int v112; // [rsp+E8h] [rbp-218h]
  __int128 v113; // [rsp+F0h] [rbp-210h] BYREF
  __int128 v114; // [rsp+100h] [rbp-200h]
  __int64 *v115; // [rsp+110h] [rbp-1F0h]
  _QWORD *v116; // [rsp+118h] [rbp-1E8h]
  __int64 v117[7]; // [rsp+120h] [rbp-1E0h] BYREF
  char *v118; // [rsp+158h] [rbp-1A8h]
  int v119; // [rsp+160h] [rbp-1A0h]
  char v120; // [rsp+168h] [rbp-198h] BYREF
  char *v121; // [rsp+188h] [rbp-178h]
  int v122; // [rsp+190h] [rbp-170h]
  char v123; // [rsp+198h] [rbp-168h] BYREF
  char *v124; // [rsp+1B8h] [rbp-148h]
  char v125; // [rsp+1C8h] [rbp-138h] BYREF
  char *v126; // [rsp+1E8h] [rbp-118h]
  char v127; // [rsp+1F8h] [rbp-108h] BYREF
  char *v128; // [rsp+218h] [rbp-E8h]
  int v129; // [rsp+220h] [rbp-E0h]
  char v130; // [rsp+228h] [rbp-D8h] BYREF
  __int64 v131; // [rsp+250h] [rbp-B0h]
  unsigned int v132; // [rsp+260h] [rbp-A0h]
  unsigned __int64 v133; // [rsp+268h] [rbp-98h]
  unsigned int v134; // [rsp+270h] [rbp-90h]
  char *v135; // [rsp+278h] [rbp-88h] BYREF
  int v136; // [rsp+280h] [rbp-80h]
  char v137; // [rsp+288h] [rbp-78h] BYREF
  __int64 v138; // [rsp+2B8h] [rbp-48h]
  unsigned int v139; // [rsp+2C8h] [rbp-38h]

  if ( sub_BA8DC0((__int64)a1, (__int64)"llvm.dbg.cu", 11) )
  {
    if ( (_BYTE)qword_5008FC8 )
    {
      v9 = (__int64)sub_CB7330();
      v10 = *(__m128i **)(v9 + 32);
      v11 = *(_QWORD *)(v9 + 24) - (_QWORD)v10;
      if ( v11 >= a5 )
      {
LABEL_4:
        if ( a5 )
        {
          memcpy(v10, a4, a5);
          v88 = *(_QWORD *)(v9 + 24);
          v10 = (__m128i *)(a5 + *(_QWORD *)(v9 + 32));
          *(_QWORD *)(v9 + 32) = v10;
          v11 = v88 - (_QWORD)v10;
        }
        if ( v11 > 0x1F )
          goto LABEL_7;
        goto LABEL_11;
      }
    }
    else
    {
      v9 = (__int64)sub_CB72A0();
      v10 = *(__m128i **)(v9 + 32);
      v11 = *(_QWORD *)(v9 + 24) - (_QWORD)v10;
      if ( v11 >= a5 )
        goto LABEL_4;
    }
    v13 = sub_CB6200(v9, (unsigned __int8 *)a4, a5);
    v10 = *(__m128i **)(v13 + 32);
    v9 = v13;
    if ( *(_QWORD *)(v13 + 24) - (_QWORD)v10 > 0x1Fu )
    {
LABEL_7:
      *v10 = _mm_load_si128((const __m128i *)&xmmword_439AB00);
      v10[1] = _mm_load_si128((const __m128i *)&xmmword_439AB10);
      *(_QWORD *)(v9 + 32) += 32LL;
      return 0;
    }
LABEL_11:
    sub_CB6200(v9, "Skipping module with debug info\n", 0x20u);
    return 0;
  }
  sub_AE0470((__int64)v117, a1, 1, 0);
  v14 = (__int64 *)*a1;
  v15 = sub_BCB2D0((_QWORD *)*a1);
  v16 = a1[22];
  v17 = a1[21];
  LOBYTE(v108) = 0;
  BYTE8(v114) = 0;
  v103 = v15;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v106[0] = a1;
  v106[1] = &v109;
  v106[2] = v117;
  v102 = 1;
  v19 = sub_ADC750((__int64)v117, v17, v16, (__int64)"/", 1, v18, v113, v114, v107[0], v107[1], 0);
  v20 = 2;
  v104 = v19;
  v21 = sub_ADDEF0(
          v117,
          2,
          v19,
          (__int64)"debugify",
          8,
          1u,
          (__int64)byte_3F871B3,
          0,
          0,
          0,
          0,
          1,
          0,
          1u,
          0,
          0,
          0,
          0,
          0,
          0,
          0);
  v25 = a3;
  v95 = a2;
  v89 = (_BYTE *)v21;
  if ( a2 == a3 )
  {
    v58 = 0;
  }
  else
  {
    v26 = 1;
    do
    {
      v27 = 0;
      if ( v95 )
        v27 = v95 - 56;
      v94 = v27;
      if ( !sub_B2FC80(v27) && !sub_B2FC80(v94) )
      {
        v92 = sub_B2FC00((_BYTE *)v94);
        if ( !v92 )
        {
          v28 = sub_ADD430((__int64)v117, 0, 0);
          v97 = sub_ADCD40((__int64)v117, v28, 0, 0);
          v99 = v104;
          v29 = -(((*(_BYTE *)(v94 + 32) + 9) & 0xFu) < 2);
          v30 = sub_BD5D20(v94);
          v32 = v31;
          v33 = (__int64)v30;
          v34 = sub_BD5D20(v94);
          v105 = sub_ADE3D0(
                   (__int64)v117,
                   v89,
                   (__int64)v34,
                   v35,
                   v33,
                   v32,
                   v99,
                   v26,
                   v97,
                   v26,
                   0,
                   (v29 & 4u) + 24,
                   0,
                   0,
                   0,
                   0,
                   (__int64)byte_3F871B3,
                   0);
          sub_B994C0(v94, v105);
          *(_QWORD *)&v114 = v117;
          *(_QWORD *)&v113 = &v102;
          *((_QWORD *)&v113 + 1) = &v103;
          *((_QWORD *)&v114 + 1) = &v105;
          v115 = &v104;
          v116 = v106;
          v96 = *(_QWORD *)(v94 + 80);
          if ( v94 + 72 != v96 )
          {
            while ( 1 )
            {
              if ( !v96 )
                BUG();
              v36 = *(_QWORD *)(v96 + 32);
              v100 = v96 + 24;
              if ( v96 + 24 != v36 )
                break;
LABEL_35:
              if ( (int)qword_5008C88 > 0 )
              {
                v43 = v96 - 24;
                v44 = sub_AA4FF0(v96 - 24);
                if ( !v44 )
                  BUG();
                v45 = (unsigned int)*(unsigned __int8 *)(v44 - 24) - 39;
                if ( (unsigned int)v45 > 0x38 || (v46 = 0x100060000000001LL, !_bittest64(&v46, v45)) )
                {
                  v47 = 0;
                  v101 = sub_29C0D60(v43);
                  v49 = sub_AA5190(v43);
                  v50 = *(_QWORD **)(v96 + 32);
                  if ( v49 )
                    v47 = v48;
                  if ( v50 )
                    v50 -= 3;
                  if ( (_QWORD *)v101 != v50 )
                  {
                    v98 = v26;
                    v51 = v92;
                    v52 = v47;
                    v53 = v49;
                    do
                    {
                      if ( *(_BYTE *)(v50[1] + 8LL) != 7 )
                      {
                        v54 = *(unsigned __int8 *)v50;
                        if ( (_BYTE)v54 != 84 )
                        {
                          v55 = (unsigned int)(v54 - 39);
                          if ( (unsigned int)v55 > 0x38 || (v56 = 0x100060000000001LL, !_bittest64(&v56, v55)) )
                          {
                            v53 = v50[4];
                            v52 = 0;
                          }
                        }
                        sub_29C2990((unsigned int **)&v113, (__int64)v50, v53, 0, v52);
                        v51 = 1;
                      }
                      v57 = v50[4];
                      if ( v57 == v50[5] + 48LL || !v57 )
                        v50 = 0;
                      else
                        v50 = (_QWORD *)(v57 - 24);
                    }
                    while ( (_QWORD *)v101 != v50 );
                    v92 = v51;
                    v26 = v98;
                  }
                }
              }
              v96 = *(_QWORD *)(v96 + 8);
              if ( v94 + 72 == v96 )
                goto LABEL_57;
            }
            v37 = v26;
            while ( 2 )
            {
              v41 = v36 - 24;
              if ( !v36 )
                v41 = 0;
              ++v26;
              v42 = sub_B01860(v14, v37, 1u, v105, 0, 0, 0, 1);
              sub_B10CB0(v107, (__int64)v42);
              v39 = v41 + 48;
              if ( (__int64 *)(v41 + 48) == v107 )
              {
                if ( v107[0] )
                {
                  sub_B91220((__int64)v107, v107[0]);
                  v36 = *(_QWORD *)(v36 + 8);
                  if ( v100 == v36 )
                    goto LABEL_35;
                  goto LABEL_29;
                }
              }
              else
              {
                v38 = *(_QWORD *)(v41 + 48);
                if ( v38 )
                {
                  sub_B91220(v41 + 48, v38);
                  v39 = v41 + 48;
                }
                v40 = (unsigned __int8 *)v107[0];
                *(_QWORD *)(v41 + 48) = v107[0];
                if ( v40 )
                  sub_B976B0((__int64)v107, v40, v39);
              }
              v36 = *(_QWORD *)(v36 + 8);
              if ( v100 == v36 )
                goto LABEL_35;
LABEL_29:
              v37 = v26;
              continue;
            }
          }
LABEL_57:
          if ( (_DWORD)qword_5008C88 == 1 && !v92 )
          {
            v86 = *(_QWORD *)(v94 + 80);
            if ( v86 )
              v86 -= 24;
            v87 = sub_29C0D60(v86);
            sub_29C2990((unsigned int **)&v113, v87, v87 + 24, 0, 0);
          }
          if ( *(_QWORD *)(a6 + 16) )
            (*(void (__fastcall **)(__int64, __int64 *, __int64))(a6 + 24))(a6, v117, v94);
          v20 = v105;
          sub_ADC590((__int64)v117, v105);
        }
      }
      v95 = *(_QWORD *)(v95 + 8);
    }
    while ( a3 != v95 );
    v58 = (unsigned int)(v26 - 1);
  }
  sub_ADCDB0((__int64)v117, v20, v22, v25, v23, v24);
  v59 = sub_BA8E40((__int64)a1, "llvm.debugify", 0xDu);
  v60 = sub_ACD640(v103, v58, 0);
  *(_QWORD *)&v113 = sub_B98A20(v60, v58);
  v61 = sub_B9C770(v14, (__int64 *)&v113, (__int64 *)1, 0, 1);
  sub_B979A0(v59, v61);
  v62 = (unsigned int)(v102 - 1);
  v63 = sub_ACD640(v103, v62, 0);
  *(_QWORD *)&v113 = sub_B98A20(v63, v62);
  v64 = sub_B9C770(v14, (__int64 *)&v113, (__int64 *)1, 0, 1);
  sub_B979A0(v59, v64);
  if ( !sub_BA91D0((__int64)a1, "Debug Info Version", 0x12u) )
    sub_BA93D0((__int64 **)a1, 2u, "Debug Info Version", 0x12u, 3u);
  sub_C7D6A0(v110, 16LL * v112, 8);
  v65 = v139;
  if ( v139 )
  {
    v66 = v138;
    v67 = v138 + 56LL * v139;
    do
    {
      if ( *(_QWORD *)v66 != -4096 && *(_QWORD *)v66 != -8192 )
      {
        v68 = *(_QWORD *)(v66 + 8);
        v69 = v68 + 8LL * *(unsigned int *)(v66 + 16);
        if ( v68 != v69 )
        {
          do
          {
            v70 = *(_QWORD *)(v69 - 8);
            v69 -= 8LL;
            if ( v70 )
              sub_B91220(v69, v70);
          }
          while ( v68 != v69 );
          v69 = *(_QWORD *)(v66 + 8);
        }
        if ( v69 != v66 + 24 )
          _libc_free(v69);
      }
      v66 += 56;
    }
    while ( v67 != v66 );
    v65 = v139;
  }
  sub_C7D6A0(v138, 56 * v65, 8);
  v71 = v135;
  v72 = &v135[8 * v136];
  if ( v135 != v72 )
  {
    do
    {
      v73 = *((_QWORD *)v72 - 1);
      v72 -= 8;
      if ( v73 )
        sub_B91220((__int64)v72, v73);
    }
    while ( v71 != v72 );
    v72 = v135;
  }
  if ( v72 != &v137 )
    _libc_free((unsigned __int64)v72);
  v74 = v133;
  v75 = v133 + 56LL * v134;
  if ( v133 != v75 )
  {
    do
    {
      v75 -= 56LL;
      v76 = *(_QWORD *)(v75 + 40);
      if ( v76 != v75 + 56 )
        _libc_free(v76);
      sub_C7D6A0(*(_QWORD *)(v75 + 16), 8LL * *(unsigned int *)(v75 + 32), 8);
    }
    while ( v74 != v75 );
    v75 = v133;
  }
  if ( (char **)v75 != &v135 )
    _libc_free(v75);
  sub_C7D6A0(v131, 16LL * v132, 8);
  v77 = v128;
  v78 = &v128[8 * v129];
  if ( v128 != v78 )
  {
    do
    {
      v79 = *((_QWORD *)v78 - 1);
      v78 -= 8;
      if ( v79 )
        sub_B91220((__int64)v78, v79);
    }
    while ( v77 != v78 );
    v78 = v128;
  }
  if ( v78 != &v130 )
    _libc_free((unsigned __int64)v78);
  if ( v126 != &v127 )
    _libc_free((unsigned __int64)v126);
  if ( v124 != &v125 )
    _libc_free((unsigned __int64)v124);
  v80 = v121;
  v81 = &v121[8 * v122];
  if ( v121 != v81 )
  {
    do
    {
      v82 = *((_QWORD *)v81 - 1);
      v81 -= 8;
      if ( v82 )
        sub_B91220((__int64)v81, v82);
    }
    while ( v80 != v81 );
    v81 = v121;
  }
  if ( v81 != &v123 )
    _libc_free((unsigned __int64)v81);
  v83 = v118;
  v84 = &v118[8 * v119];
  if ( v118 != v84 )
  {
    do
    {
      v85 = *((_QWORD *)v84 - 1);
      v84 -= 8;
      if ( v85 )
        sub_B91220((__int64)v84, v85);
    }
    while ( v83 != v84 );
    v84 = v118;
  }
  if ( v84 != &v120 )
    _libc_free((unsigned __int64)v84);
  return 1;
}
