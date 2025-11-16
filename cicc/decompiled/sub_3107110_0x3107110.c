// Function: sub_3107110
// Address: 0x3107110
//
__int64 __fastcall sub_3107110(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 j; // rdx
  __int64 v8; // rsi
  void *v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // r12
  _QWORD *v12; // rbx
  unsigned __int64 v13; // r13
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r14
  __m128i *v18; // rdx
  __m128i si128; // xmm0
  _BYTE *v20; // rax
  __int64 v21; // r14
  __int64 v22; // rbx
  __int64 v23; // rdi
  _QWORD *v24; // rax
  __int64 v25; // r8
  __int64 v26; // rcx
  unsigned __int64 *v27; // r14
  _QWORD *v28; // rcx
  unsigned int v29; // r14d
  __int64 v30; // r12
  __int64 i; // r13
  __int64 v32; // rax
  const char *v33; // rax
  size_t v34; // rdx
  _WORD *v35; // rdi
  unsigned __int8 *v36; // rsi
  unsigned __int64 v37; // rax
  _BYTE *v38; // rax
  __int64 v39; // r14
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  int v44; // r14d
  __int64 *v45; // rdx
  int v46; // edi
  __int64 v47; // rax
  unsigned __int64 v48; // r9
  unsigned int v49; // eax
  __int64 v50; // r11
  int v51; // esi
  __int64 *v52; // rcx
  unsigned int v53; // ecx
  __int64 v54; // r8
  int v55; // esi
  __int64 *v56; // rax
  __int64 v58; // [rsp+28h] [rbp-298h]
  __int64 v59; // [rsp+30h] [rbp-290h]
  void (__fastcall *v60)(_BYTE *, __int64 *, __int64); // [rsp+38h] [rbp-288h]
  __int64 v61; // [rsp+40h] [rbp-280h]
  unsigned int v62; // [rsp+48h] [rbp-278h]
  __int64 v63; // [rsp+48h] [rbp-278h]
  __int64 v64; // [rsp+50h] [rbp-270h]
  __int64 v65; // [rsp+50h] [rbp-270h]
  void *v67; // [rsp+60h] [rbp-260h]
  __int64 v68; // [rsp+68h] [rbp-258h]
  __int64 v69; // [rsp+70h] [rbp-250h]
  void *v70; // [rsp+70h] [rbp-250h]
  __int64 v71; // [rsp+78h] [rbp-248h]
  void *v72; // [rsp+80h] [rbp-240h]
  __int64 v73; // [rsp+88h] [rbp-238h]
  __int64 v74; // [rsp+90h] [rbp-230h]
  __int64 v75; // [rsp+98h] [rbp-228h]
  __int64 v76; // [rsp+98h] [rbp-228h]
  size_t v77; // [rsp+98h] [rbp-228h]
  __int64 v78; // [rsp+98h] [rbp-228h]
  unsigned __int64 v79; // [rsp+98h] [rbp-228h]
  _QWORD v80[2]; // [rsp+A0h] [rbp-220h] BYREF
  void (__fastcall *v81)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-210h]
  __int64 (__fastcall *v82)(__int64 *, __int64); // [rsp+B8h] [rbp-208h]
  _QWORD v83[2]; // [rsp+C0h] [rbp-200h] BYREF
  void (__fastcall *v84)(_BYTE *, _QWORD *); // [rsp+D0h] [rbp-1F0h]
  __int64 (__fastcall *v85)(__int64 *, __int64); // [rsp+D8h] [rbp-1E8h]
  _QWORD v86[2]; // [rsp+E0h] [rbp-1E0h] BYREF
  void (__fastcall *v87)(_BYTE *, __int64 *, __int64); // [rsp+F0h] [rbp-1D0h]
  __int64 (__fastcall *v88)(__int64 *, __int64); // [rsp+F8h] [rbp-1C8h]
  _BYTE v89[16]; // [rsp+100h] [rbp-1C0h] BYREF
  void (__fastcall *v90)(_BYTE *, _BYTE *, __int64); // [rsp+110h] [rbp-1B0h]
  __int64 (__fastcall *v91)(__int64 *, __int64); // [rsp+118h] [rbp-1A8h]
  _BYTE v92[16]; // [rsp+120h] [rbp-1A0h] BYREF
  void (__fastcall *v93)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-190h]
  __int64 (__fastcall *v94)(__int64 *, __int64); // [rsp+138h] [rbp-188h]
  __int64 v95; // [rsp+140h] [rbp-180h] BYREF
  void *v96; // [rsp+148h] [rbp-178h]
  void (__fastcall *v97)(_BYTE *, __int64 *, __int64); // [rsp+150h] [rbp-170h]
  __int64 (__fastcall *v98)(__int64 *, __int64); // [rsp+158h] [rbp-168h]
  __int64 v99; // [rsp+160h] [rbp-160h]
  __int64 v100; // [rsp+168h] [rbp-158h]
  __int64 v101; // [rsp+170h] [rbp-150h]
  __int64 v102; // [rsp+178h] [rbp-148h]
  __int64 v103; // [rsp+180h] [rbp-140h] BYREF
  _BYTE v104[16]; // [rsp+188h] [rbp-138h] BYREF
  void (__fastcall *v105)(_BYTE *, _BYTE *, __int64); // [rsp+198h] [rbp-128h]
  __int64 (__fastcall *v106)(__int64 *, __int64); // [rsp+1A0h] [rbp-120h]
  _BYTE v107[16]; // [rsp+1A8h] [rbp-118h] BYREF
  void (__fastcall *v108)(_BYTE *, _BYTE *, __int64); // [rsp+1B8h] [rbp-108h]
  __int64 (__fastcall *v109)(__int64 *, __int64); // [rsp+1C0h] [rbp-100h]
  _BYTE v110[16]; // [rsp+1C8h] [rbp-F8h] BYREF
  void (__fastcall *v111)(_BYTE *, _BYTE *, __int64); // [rsp+1D8h] [rbp-E8h]
  __int64 (__fastcall *v112)(__int64 *, __int64); // [rsp+1E0h] [rbp-E0h]
  __int64 v113; // [rsp+1E8h] [rbp-D8h]
  __int64 v114; // [rsp+1F0h] [rbp-D0h]
  __int64 v115; // [rsp+1F8h] [rbp-C8h]
  unsigned int v116; // [rsp+200h] [rbp-C0h]
  __int64 v117; // [rsp+208h] [rbp-B8h]
  __int64 v118; // [rsp+210h] [rbp-B0h]
  __int64 v119; // [rsp+218h] [rbp-A8h]
  unsigned int v120; // [rsp+220h] [rbp-A0h]
  __int64 v121; // [rsp+228h] [rbp-98h] BYREF
  _QWORD *v122; // [rsp+230h] [rbp-90h]
  __int64 v123; // [rsp+238h] [rbp-88h]
  unsigned int v124; // [rsp+240h] [rbp-80h]
  char v125[8]; // [rsp+248h] [rbp-78h] BYREF
  void *src; // [rsp+250h] [rbp-70h]
  unsigned int v127; // [rsp+260h] [rbp-60h]
  __int64 v128; // [rsp+270h] [rbp-50h]
  __int64 v129; // [rsp+278h] [rbp-48h]
  __int64 v130; // [rsp+280h] [rbp-40h]

  v5 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  v82 = sub_3101B60;
  v81 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))sub_3101C60;
  v80[0] = v5;
  v83[0] = v5;
  v86[0] = v5;
  v88 = sub_3101BA0;
  v87 = (void (__fastcall *)(_BYTE *, __int64 *, __int64))sub_3101CC0;
  v85 = sub_3101B80;
  v84 = (void (__fastcall *)(_BYTE *, _QWORD *))sub_3101C90;
  v97 = 0;
  sub_3101CC0(&v95, v86, 2);
  v93 = 0;
  v98 = v88;
  v97 = v87;
  if ( sub_3101C90 )
  {
    v84(v92, v83);
    v94 = v85;
    v93 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v84;
  }
  v90 = 0;
  if ( v81 )
  {
    v81(v89, v80, 2);
    v103 = 65793;
    v105 = 0;
    v91 = v82;
    v90 = v81;
    if ( v81 )
    {
      v81(v104, v89, 2);
      v106 = v91;
      v105 = v90;
    }
  }
  else
  {
    v103 = 65793;
    v105 = 0;
  }
  v108 = 0;
  if ( v93 )
  {
    v93(v107, v92, 2);
    v109 = v94;
    v108 = v93;
  }
  v111 = 0;
  if ( v97 )
  {
    v97(v110, &v95, 2);
    v112 = v98;
    v111 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v97;
  }
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  sub_3106C40((__int64)v125, (__int64)&v103, 0);
  if ( v90 )
    v90(v89, v89, 3);
  if ( v93 )
    v93(v92, v92, 3);
  if ( v97 )
    v97(&v95, &v95, 3);
  v58 = a3 + 24;
  v59 = *(_QWORD *)(a3 + 32);
  if ( v59 != a3 + 24 )
  {
    do
    {
      if ( !v59 )
        BUG();
      v6 = *(_QWORD *)(v59 + 24);
      v61 = v59 + 16;
      if ( v59 + 16 != v6 )
      {
        if ( !v6 )
          BUG();
        while ( 1 )
        {
          j = *(_QWORD *)(v6 + 32);
          if ( j != v6 + 24 )
            goto LABEL_46;
          v6 = *(_QWORD *)(v6 + 8);
          if ( v59 + 16 == v6 )
            goto LABEL_24;
          if ( !v6 )
            BUG();
        }
      }
      v71 = 0;
      while ( v61 != v6 )
      {
        v15 = v71 - 24;
        if ( !v71 )
          v15 = 0;
        v16 = v15;
        v17 = *a2;
        v18 = *(__m128i **)(*a2 + 32);
        if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v18 <= 0x16u )
        {
          v17 = sub_CB6200(*a2, "-- Explore context of: ", 0x17u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_44CECA0);
          v18[1].m128i_i32[0] = 1864397944;
          v18[1].m128i_i16[2] = 14950;
          v18[1].m128i_i8[6] = 32;
          *v18 = si128;
          *(_QWORD *)(v17 + 32) += 23LL;
        }
        sub_A69870(v16, (_BYTE *)v17, 0);
        v20 = *(_BYTE **)(v17 + 32);
        if ( *(_BYTE **)(v17 + 24) == v20 )
        {
          sub_CB6200(v17, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v20 = 10;
          ++*(_QWORD *)(v17 + 32);
        }
        sub_C7D6A0(0, 0, 8);
        v72 = 0;
        v62 = v127;
        if ( v127 )
        {
          v21 = 8LL * v127;
          v72 = (void *)sub_C7D670(v21, 8);
          memcpy(v72, src, v21);
        }
        v22 = v128;
        v74 = v129;
        v73 = v130;
        if ( v124 )
        {
          LODWORD(v23) = (v124 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v24 = &v122[2 * (unsigned int)v23];
          v25 = *v24;
          if ( v16 == *v24 )
          {
LABEL_58:
            v26 = v24[1];
            v27 = v24 + 1;
            if ( v26 )
              goto LABEL_59;
            goto LABEL_103;
          }
          v44 = 1;
          v45 = 0;
          while ( v25 != -4096 )
          {
            if ( v25 == -8192 && !v45 )
              v45 = v24;
            v23 = (v124 - 1) & ((_DWORD)v23 + v44);
            v24 = &v122[2 * v23];
            v25 = *v24;
            if ( v16 == *v24 )
              goto LABEL_58;
            ++v44;
          }
          if ( !v45 )
            v45 = v24;
          ++v121;
          v46 = v123 + 1;
          if ( 4 * ((int)v123 + 1) < 3 * v124 )
          {
            if ( v124 - HIDWORD(v123) - v46 <= v124 >> 3 )
            {
              sub_2512960((__int64)&v121, v124);
              if ( !v124 )
              {
LABEL_140:
                LODWORD(v123) = v123 + 1;
                BUG();
              }
              v53 = (v124 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
              v46 = v123 + 1;
              v45 = &v122[2 * v53];
              v54 = *v45;
              if ( v16 != *v45 )
              {
                v55 = 1;
                v56 = 0;
                while ( v54 != -4096 )
                {
                  if ( v54 == -8192 && !v56 )
                    v56 = v45;
                  v53 = (v124 - 1) & (v55 + v53);
                  v45 = &v122[2 * v53];
                  v54 = *v45;
                  if ( v16 == *v45 )
                    goto LABEL_100;
                  ++v55;
                }
                if ( v56 )
                  v45 = v56;
              }
            }
            goto LABEL_100;
          }
        }
        else
        {
          ++v121;
        }
        sub_2512960((__int64)&v121, 2 * v124);
        if ( !v124 )
          goto LABEL_140;
        v46 = v123 + 1;
        v49 = (v124 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v45 = &v122[2 * v49];
        v50 = *v45;
        if ( v16 != *v45 )
        {
          v51 = 1;
          v52 = 0;
          while ( v50 != -4096 )
          {
            if ( !v52 && v50 == -8192 )
              v52 = v45;
            v49 = (v124 - 1) & (v51 + v49);
            v45 = &v122[2 * v49];
            v50 = *v45;
            if ( v16 == *v45 )
              goto LABEL_100;
            ++v51;
          }
          if ( v52 )
            v45 = v52;
        }
LABEL_100:
        LODWORD(v123) = v46;
        if ( *v45 != -4096 )
          --HIDWORD(v123);
        *v45 = v16;
        v27 = (unsigned __int64 *)(v45 + 1);
        v45[1] = 0;
LABEL_103:
        v47 = sub_22077B0(0x40u);
        v26 = v47;
        if ( v47 )
        {
          v78 = v47;
          sub_3106C40(v47, (__int64)&v103, v16);
          v26 = v78;
        }
        v48 = *v27;
        *v27 = v26;
        if ( v48 )
        {
          v79 = v48;
          sub_C7D6A0(*(_QWORD *)(v48 + 8), 8LL * *(unsigned int *)(v48 + 24), 8);
          j_j___libc_free_0(v79);
          v26 = *v27;
        }
LABEL_59:
        v75 = v26;
        sub_C7D6A0(0, 0, 8);
        v28 = (_QWORD *)v75;
        v29 = *(_DWORD *)(v75 + 24);
        if ( v29 )
        {
          v68 = 8LL * v29;
          v67 = (void *)sub_C7D670(v68, 8);
          v60 = *(void (__fastcall **)(_BYTE *, __int64 *, __int64))(v75 + 16);
          memcpy(v67, *(const void **)(v75 + 8), v68);
          v28 = (_QWORD *)v75;
        }
        else
        {
          v60 = 0;
          v68 = 0;
          v67 = 0;
        }
        v30 = v28[7];
        v76 = v28[4];
        v69 = v28[5];
        v64 = v28[6];
        sub_C7D6A0(0, 0, 8);
        sub_C7D6A0(0, 0, 8);
        sub_C7D6A0(0, 0, 8);
        sub_C7D6A0(0, 0, 8);
        v95 = 0;
        v96 = 0;
        v97 = 0;
        v98 = 0;
        sub_C7D6A0(0, 0, 8);
        LODWORD(v98) = v29;
        if ( v29 )
        {
          v96 = (void *)sub_C7D670(v68, 8);
          v97 = v60;
          memcpy(v96, v67, 8LL * (unsigned int)v98);
        }
        else
        {
          v96 = 0;
          v97 = 0;
        }
        v102 = v30;
        v99 = v76;
        v100 = v69;
        v101 = v64;
        sub_C7D6A0(0, 0, 8);
        v70 = 0;
        v65 = 8LL * v62;
        if ( v62 )
        {
          v70 = (void *)sub_C7D670(8LL * v62, 8);
          memcpy(v70, v72, 8LL * v62);
        }
        v63 = v6;
        for ( i = v100; v22 != i || v101 != v74 || v102 != v73; i = v100 )
        {
          v39 = *a2;
          v40 = *(_QWORD *)(*a2 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(*a2 + 24) - v40) > 5 )
          {
            *(_DWORD *)v40 = 1180377120;
            *(_WORD *)(v40 + 4) = 8250;
            *(_QWORD *)(v39 + 32) += 6LL;
          }
          else
          {
            v39 = sub_CB6200(*a2, "  [F: ", 6u);
          }
          v32 = sub_B43CB0(i);
          v33 = sub_BD5D20(v32);
          v35 = *(_WORD **)(v39 + 32);
          v36 = (unsigned __int8 *)v33;
          v37 = *(_QWORD *)(v39 + 24) - (_QWORD)v35;
          if ( v34 > v37 )
          {
            v41 = sub_CB6200(v39, v36, v34);
            v35 = *(_WORD **)(v41 + 32);
            v39 = v41;
            v37 = *(_QWORD *)(v41 + 24) - (_QWORD)v35;
          }
          else if ( v34 )
          {
            v77 = v34;
            memcpy(v35, v36, v34);
            v43 = *(_QWORD *)(v39 + 24);
            v35 = (_WORD *)(v77 + *(_QWORD *)(v39 + 32));
            *(_QWORD *)(v39 + 32) = v35;
            v37 = v43 - (_QWORD)v35;
          }
          if ( v37 <= 1 )
          {
            v39 = sub_CB6200(v39, (unsigned __int8 *)"] ", 2u);
          }
          else
          {
            *v35 = 8285;
            *(_QWORD *)(v39 + 32) += 2LL;
          }
          sub_A69870(i, (_BYTE *)v39, 0);
          v38 = *(_BYTE **)(v39 + 32);
          if ( *(_BYTE **)(v39 + 24) == v38 )
          {
            sub_CB6200(v39, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v38 = 10;
            ++*(_QWORD *)(v39 + 32);
          }
          v100 = sub_3106C80((__int64)&v95);
        }
        v6 = v63;
        sub_C7D6A0((__int64)v70, v65, 8);
        sub_C7D6A0((__int64)v96, 8LL * (unsigned int)v98, 8);
        sub_C7D6A0((__int64)v72, v65, 8);
        sub_C7D6A0((__int64)v67, v68, 8);
        for ( j = *(_QWORD *)(v71 + 8); ; j = *(_QWORD *)(v6 + 32) )
        {
          v42 = v6 - 24;
          if ( !v6 )
            v42 = 0;
          if ( j != v42 + 48 )
            break;
          v6 = *(_QWORD *)(v6 + 8);
          if ( v61 == v6 )
            goto LABEL_24;
          if ( !v6 )
            BUG();
        }
LABEL_46:
        v71 = j;
      }
LABEL_24:
      v59 = *(_QWORD *)(v59 + 8);
    }
    while ( v58 != v59 );
  }
  v8 = v127;
  v9 = src;
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  sub_C7D6A0((__int64)v9, 8 * v8, 8);
  v10 = v124;
  if ( v124 )
  {
    v11 = v122;
    v12 = &v122[2 * v124];
    do
    {
      if ( *v11 != -8192 && *v11 != -4096 )
      {
        v13 = v11[1];
        if ( v13 )
        {
          sub_C7D6A0(*(_QWORD *)(v13 + 8), 8LL * *(unsigned int *)(v13 + 24), 8);
          j_j___libc_free_0(v13);
        }
      }
      v11 += 2;
    }
    while ( v12 != v11 );
    v10 = v124;
  }
  sub_C7D6A0((__int64)v122, 16 * v10, 8);
  sub_C7D6A0(v118, 16LL * v120, 8);
  sub_C7D6A0(v114, 16LL * v116, 8);
  if ( v111 )
    v111(v110, v110, 3);
  if ( v108 )
    v108(v107, v107, 3);
  if ( v105 )
    v105(v104, v104, 3);
  if ( v87 )
    v87(v86, v86, 3);
  if ( v84 )
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v84)(v83, v83, 3);
  if ( v81 )
    v81(v80, v80, 3);
  return a1;
}
