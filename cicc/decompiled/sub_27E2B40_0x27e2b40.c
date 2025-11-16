// Function: sub_27E2B40
// Address: 0x27e2b40
//
void __fastcall sub_27E2B40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r8
  int v16; // eax
  int v17; // eax
  unsigned int v18; // edx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  _QWORD *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r9
  _BYTE *v26; // r13
  bool v27; // bl
  __int64 v28; // r12
  const char *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  unsigned int v32; // esi
  __int64 v33; // r8
  __int64 v34; // r9
  int v35; // edi
  _BYTE *v36; // rax
  _QWORD *v37; // rbx
  int v38; // edx
  _BYTE *v39; // rdx
  unsigned __int64 *v40; // r14
  __int64 v41; // rdx
  _QWORD *v42; // r14
  __int64 v43; // rcx
  _QWORD *v44; // rdx
  _BYTE *v45; // rdi
  void *v46; // rcx
  __int64 v47; // rax
  _QWORD *v48; // rax
  _QWORD *v49; // rdx
  _QWORD *v50; // r14
  __int64 v51; // r13
  __int64 v52; // rcx
  __int64 v53; // r10
  __int64 v54; // rax
  __int64 v55; // rax
  _BYTE *v56; // r8
  __int64 v57; // rdx
  __int64 v58; // r9
  unsigned int v59; // edi
  __int64 v60; // rsi
  _BYTE *v61; // rbx
  __int64 v62; // rdx
  __int64 v63; // rsi
  __int64 v64; // rsi
  const char *v65; // r13
  const char *v66; // rbx
  int v67; // esi
  int v68; // r13d
  __int64 v69; // rax
  bool v70; // r14
  __int64 v71; // rbx
  __int64 v72; // r8
  _QWORD *v73; // r12
  _BYTE *v74; // rcx
  __int64 v75; // rax
  __int64 v76; // rdi
  unsigned int v77; // esi
  __int64 v78; // rdx
  __int64 v79; // r9
  __int64 v80; // rax
  _QWORD *v81; // rax
  __int64 v82; // r13
  _QWORD *v83; // rbx
  int v84; // r13d
  __int64 v85; // r12
  _QWORD *v86; // rax
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // r12
  __int64 v90; // rbx
  unsigned __int8 *v91; // rdx
  __int64 v92; // rsi
  int v93; // edx
  int v94; // r11d
  int v95; // ecx
  int v96; // edi
  int v97; // edi
  int v98; // esi
  _QWORD *v99; // rcx
  unsigned int v100; // edx
  _BYTE *v101; // r10
  int v102; // edi
  int v103; // esi
  unsigned int v104; // edx
  _BYTE *v105; // r10
  __int64 v107; // [rsp+18h] [rbp-288h]
  __int64 v108; // [rsp+20h] [rbp-280h]
  __int64 v109; // [rsp+38h] [rbp-268h]
  __int64 v111; // [rsp+48h] [rbp-258h]
  __int64 *v112; // [rsp+58h] [rbp-248h]
  __int64 *v113; // [rsp+58h] [rbp-248h]
  __int64 v115; // [rsp+68h] [rbp-238h]
  unsigned __int16 v117; // [rsp+78h] [rbp-228h]
  __int64 v118; // [rsp+78h] [rbp-228h]
  __int64 v119; // [rsp+78h] [rbp-228h]
  __int64 v120; // [rsp+88h] [rbp-218h] BYREF
  __int64 v121; // [rsp+90h] [rbp-210h] BYREF
  __int64 v122; // [rsp+98h] [rbp-208h]
  __m128i v123; // [rsp+A0h] [rbp-200h] BYREF
  char v124[32]; // [rsp+B0h] [rbp-1F0h] BYREF
  __int64 v125; // [rsp+D0h] [rbp-1D0h] BYREF
  __int64 v126; // [rsp+D8h] [rbp-1C8h]
  __int64 v127; // [rsp+E0h] [rbp-1C0h]
  unsigned int v128; // [rsp+E8h] [rbp-1B8h]
  __int64 *v129; // [rsp+F0h] [rbp-1B0h] BYREF
  __int64 v130; // [rsp+F8h] [rbp-1A8h]
  _BYTE v131[48]; // [rsp+100h] [rbp-1A0h] BYREF
  const char *v132; // [rsp+130h] [rbp-170h] BYREF
  __int64 v133; // [rsp+138h] [rbp-168h] BYREF
  __int64 v134; // [rsp+140h] [rbp-160h] BYREF
  _BYTE *v135; // [rsp+148h] [rbp-158h]
  __int64 v136; // [rsp+150h] [rbp-150h]
  int v137; // [rsp+248h] [rbp-58h] BYREF
  unsigned __int64 v138; // [rsp+250h] [rbp-50h]
  const char *v139; // [rsp+258h] [rbp-48h]
  int *v140; // [rsp+260h] [rbp-40h]
  __int64 v141; // [rsp+268h] [rbp-38h]

  v107 = a5;
  v120 = a2;
  if ( !a3 )
    BUG();
  v8 = a3;
  v108 = *(_QWORD *)(a3 + 16);
  while ( *(_BYTE *)(v8 - 24) == 84 )
  {
    sub_B43C20((__int64)&v129, a7);
    v132 = sub_BD5D20(v8 - 24);
    LOWORD(v136) = 261;
    v133 = v9;
    v112 = v129;
    v111 = *(_QWORD *)(v8 - 16);
    v117 = v130;
    v10 = sub_BD2DA0(80);
    v11 = v10;
    if ( v10 )
    {
      sub_B44260(v10, v111, 55, 0x8000000u, (__int64)v112, v117);
      *(_DWORD *)(v11 + 72) = 1;
      sub_BD6B50((unsigned __int8 *)v11, &v132);
      sub_BD2A10(v11, *(_DWORD *)(v11 + 72), 1);
    }
    v12 = *(_QWORD *)(v8 - 32);
    if ( (*(_DWORD *)(v8 - 20) & 0x7FFFFFF) != 0 )
    {
      v13 = 0;
      while ( a8 != *(_QWORD *)(v12 + 32LL * *(unsigned int *)(v8 + 48) + 8 * v13) )
      {
        if ( (*(_DWORD *)(v8 - 20) & 0x7FFFFFF) == (_DWORD)++v13 )
          goto LABEL_128;
      }
      v14 = 32 * v13;
    }
    else
    {
LABEL_128:
      v14 = 0x1FFFFFFFE0LL;
    }
    v15 = *(_QWORD *)(v12 + v14);
    v16 = *(_DWORD *)(v11 + 4) & 0x7FFFFFF;
    if ( v16 == *(_DWORD *)(v11 + 72) )
    {
      v119 = v15;
      sub_B48D90(v11);
      v15 = v119;
      v16 = *(_DWORD *)(v11 + 4) & 0x7FFFFFF;
    }
    v17 = (v16 + 1) & 0x7FFFFFF;
    v18 = v17 | *(_DWORD *)(v11 + 4) & 0xF8000000;
    v19 = *(_QWORD *)(v11 - 8) + 32LL * (unsigned int)(v17 - 1);
    *(_DWORD *)(v11 + 4) = v18;
    if ( *(_QWORD *)v19 )
    {
      v20 = *(_QWORD *)(v19 + 8);
      **(_QWORD **)(v19 + 16) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = *(_QWORD *)(v19 + 16);
    }
    *(_QWORD *)v19 = v15;
    if ( v15 )
    {
      v21 = *(_QWORD *)(v15 + 16);
      *(_QWORD *)(v19 + 8) = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 16) = v19 + 8;
      *(_QWORD *)(v19 + 16) = v15 + 16;
      *(_QWORD *)(v15 + 16) = v19;
    }
    *(_QWORD *)(*(_QWORD *)(v11 - 8)
              + 32LL * *(unsigned int *)(v11 + 72)
              + 8LL * ((*(_DWORD *)(v11 + 4) & 0x7FFFFFFu) - 1)) = a8;
    v22 = sub_27E1A50(a2, v8 - 24);
    v23 = v22[2];
    if ( v23 != v11 )
    {
      if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
        sub_BD60C0(v22);
      v22[2] = v11;
      if ( v11 != -8192 && v11 != -4096 )
        sub_BD73F0((__int64)v22);
    }
    v24 = a4;
    v8 = *(_QWORD *)(v8 + 8);
    LOWORD(v24) = 0;
    a4 = v24;
    if ( !v8 )
      BUG();
  }
  v118 = v8;
  v129 = (__int64 *)v131;
  v130 = 0x600000000LL;
  v125 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v113 = (__int64 *)sub_AA48A0(a8);
  sub_F46340(v8, a4, a5, a6, (__int64)&v129, v25);
  sub_F4C4C0(v129, (unsigned int)v130, (__int64)&v125, "thread", 6u, (__int64)v113);
  if ( a5 != v8 )
  {
    while ( 1 )
    {
      if ( v118 )
      {
        v26 = (_BYTE *)(v118 - 24);
        v27 = v118 != -4072 && v118 != -8168;
      }
      else
      {
        v26 = 0;
        v27 = 0;
      }
      v28 = sub_B47F80(v26);
      v29 = sub_BD5D20((__int64)v26);
      LOWORD(v136) = 261;
      v132 = v29;
      v133 = v30;
      sub_BD6B50((unsigned __int8 *)v28, &v132);
      v31 = v115;
      LOWORD(v31) = 0;
      v115 = v31;
      sub_B44240((_QWORD *)v28, a7, (unsigned __int64 *)(a7 + 48), v31);
      v135 = v26;
      v133 = 2;
      v134 = 0;
      if ( v27 )
        sub_BD73F0((__int64)&v133);
      v32 = *(_DWORD *)(a2 + 24);
      v136 = a2;
      v132 = (const char *)&unk_49DD7B0;
      if ( !v32 )
        break;
      v36 = v135;
      v34 = v32 - 1;
      v33 = *(_QWORD *)(a2 + 8);
      LODWORD(v43) = v34 & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
      v44 = (_QWORD *)(v33 + ((unsigned __int64)(unsigned int)v43 << 6));
      v45 = (_BYTE *)v44[3];
      if ( v135 != v45 )
      {
        v94 = 1;
        v37 = 0;
        while ( v45 != (_BYTE *)-4096LL )
        {
          if ( !v37 && v45 == (_BYTE *)-8192LL )
            v37 = v44;
          v43 = (unsigned int)v34 & ((_DWORD)v43 + v94);
          v44 = (_QWORD *)(v33 + (v43 << 6));
          v45 = (_BYTE *)v44[3];
          if ( v135 == v45 )
            goto LABEL_50;
          ++v94;
        }
        v95 = *(_DWORD *)(a2 + 16);
        if ( !v37 )
          v37 = v44;
        ++*(_QWORD *)a2;
        v38 = v95 + 1;
        if ( 4 * (v95 + 1) < 3 * v32 )
        {
          if ( v32 - *(_DWORD *)(a2 + 20) - v38 > v32 >> 3 )
            goto LABEL_39;
          sub_CF32C0(a2, v32);
          v96 = *(_DWORD *)(a2 + 24);
          if ( v96 )
          {
            v36 = v135;
            v97 = v96 - 1;
            v33 = *(_QWORD *)(a2 + 8);
            v98 = 1;
            v99 = 0;
            v100 = v97 & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
            v37 = (_QWORD *)(v33 + ((unsigned __int64)v100 << 6));
            v101 = (_BYTE *)v37[3];
            if ( v135 != v101 )
            {
              while ( v101 != (_BYTE *)-4096LL )
              {
                if ( !v99 && v101 == (_BYTE *)-8192LL )
                  v99 = v37;
                v34 = (unsigned int)(v98 + 1);
                v100 = v97 & (v98 + v100);
                v37 = (_QWORD *)(v33 + ((unsigned __int64)v100 << 6));
                v101 = (_BYTE *)v37[3];
                if ( v135 == v101 )
                  goto LABEL_38;
                ++v98;
              }
LABEL_166:
              if ( v99 )
                v37 = v99;
            }
LABEL_38:
            v38 = *(_DWORD *)(a2 + 16) + 1;
LABEL_39:
            *(_DWORD *)(a2 + 16) = v38;
            if ( v37[3] == -4096 )
            {
              v40 = v37 + 1;
              if ( v36 != (_BYTE *)-4096LL )
                goto LABEL_44;
            }
            else
            {
              --*(_DWORD *)(a2 + 20);
              v39 = (_BYTE *)v37[3];
              if ( v36 != v39 )
              {
                v40 = v37 + 1;
                if ( v39 != 0 && v39 + 4096 != 0 && v39 != (_BYTE *)-8192LL )
                {
                  sub_BD60C0(v37 + 1);
                  v36 = v135;
                }
LABEL_44:
                v37[3] = v36;
                if ( v36 != 0 && v36 + 4096 != 0 && v36 != (_BYTE *)-8192LL )
                  sub_BD6050(v40, v133 & 0xFFFFFFFFFFFFFFF8LL);
                v36 = v135;
              }
            }
            v41 = v136;
            v42 = v37 + 5;
            v37[5] = 6;
            v37[6] = 0;
            v37[4] = v41;
            v37[7] = 0;
            goto LABEL_51;
          }
LABEL_37:
          v36 = v135;
          v37 = 0;
          goto LABEL_38;
        }
LABEL_36:
        sub_CF32C0(a2, 2 * v32);
        v35 = *(_DWORD *)(a2 + 24);
        if ( v35 )
        {
          v36 = v135;
          v102 = v35 - 1;
          v33 = *(_QWORD *)(a2 + 8);
          v103 = 1;
          v99 = 0;
          v104 = v102 & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
          v37 = (_QWORD *)(v33 + ((unsigned __int64)v104 << 6));
          v105 = (_BYTE *)v37[3];
          if ( v135 != v105 )
          {
            while ( v105 != (_BYTE *)-4096LL )
            {
              if ( !v99 && v105 == (_BYTE *)-8192LL )
                v99 = v37;
              v34 = (unsigned int)(v103 + 1);
              v104 = v102 & (v103 + v104);
              v37 = (_QWORD *)(v33 + ((unsigned __int64)v104 << 6));
              v105 = (_BYTE *)v37[3];
              if ( v135 == v105 )
                goto LABEL_38;
              ++v103;
            }
            goto LABEL_166;
          }
          goto LABEL_38;
        }
        goto LABEL_37;
      }
LABEL_50:
      v42 = v44 + 5;
LABEL_51:
      v46 = &unk_49DB358;
      LOBYTE(v46) = v36 + 4096 != 0;
      v132 = (const char *)&unk_49DB368;
      if ( ((v36 != 0) & (unsigned __int8)v46) != 0 && v36 != (_BYTE *)-8192LL )
        sub_BD60C0(&v133);
      v47 = v42[2];
      if ( v28 != v47 )
      {
        LOBYTE(v46) = v47 != -4096;
        if ( ((v47 != 0) & (unsigned __int8)v46) != 0 && v47 != -8192 )
          sub_BD60C0(v42);
        v42[2] = v28;
        if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
          sub_BD73F0((__int64)v42);
      }
      sub_F460A0(v28, &v125, v113, (__int64)v46, v33, v34);
      LOBYTE(v133) = 0;
      v48 = sub_B43F50(v28, (__int64)v26, (__int64)v132, 0, 0);
      v50 = v49;
      v51 = (__int64)v48;
      if ( v48 != v49 )
      {
        while ( *(_BYTE *)(v51 + 32) )
        {
          v51 = *(_QWORD *)(v51 + 8);
          if ( v49 == (_QWORD *)v51 )
            goto LABEL_69;
        }
        if ( v49 != (_QWORD *)v51 )
        {
LABEL_66:
          sub_27E28D0((__int64)&v120, v51);
          while ( 1 )
          {
            v51 = *(_QWORD *)(v51 + 8);
            if ( v50 == (_QWORD *)v51 )
              break;
            if ( !*(_BYTE *)(v51 + 32) )
            {
              if ( v50 != (_QWORD *)v51 )
                goto LABEL_66;
              break;
            }
          }
        }
      }
LABEL_69:
      if ( *(_BYTE *)v28 == 85
        && (v69 = *(_QWORD *)(v28 - 32)) != 0
        && !*(_BYTE *)v69
        && *(_QWORD *)(v69 + 24) == *(_QWORD *)(v28 + 80)
        && (*(_BYTE *)(v69 + 33) & 0x20) != 0
        && (v70 = *(_DWORD *)(v69 + 36) == 68 || *(_DWORD *)(v69 + 36) == 71) )
      {
        v137 = 0;
        v132 = (const char *)&v134;
        v133 = 0x1000000000LL;
        v139 = (const char *)&v137;
        v140 = &v137;
        v138 = 0;
        v141 = 0;
        sub_B58E30(&v121, v28);
        v71 = v121;
        v72 = v122;
        if ( v122 != v121 )
        {
          v109 = v28;
          v73 = (_QWORD *)v122;
          while ( 1 )
          {
            v82 = v71;
            v83 = (_QWORD *)(v71 & 0xFFFFFFFFFFFFFFF8LL);
            v84 = (v82 >> 2) & 1;
            if ( v84 )
            {
              v74 = *(_BYTE **)(*v83 + 136LL);
              if ( *v74 <= 0x1Cu )
                goto LABEL_138;
            }
            else
            {
              v74 = (_BYTE *)v83[17];
              if ( *v74 <= 0x1Cu )
                goto LABEL_122;
            }
            v75 = *(unsigned int *)(a2 + 24);
            if ( (_DWORD)v75 )
            {
              v76 = *(_QWORD *)(a2 + 8);
              v77 = (v75 - 1) & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
              v78 = v76 + ((unsigned __int64)v77 << 6);
              v79 = *(_QWORD *)(v78 + 24);
              if ( v74 == (_BYTE *)v79 )
              {
LABEL_119:
                if ( v78 != v76 + (v75 << 6) )
                {
                  v80 = *(_QWORD *)(v78 + 56);
                  v123.m128i_i64[0] = (__int64)v74;
                  v123.m128i_i64[1] = v80;
                  sub_27E2720((__int64)v124, (__int64)&v132, &v123, (__int64)v74, v72, v79);
                }
              }
              else
              {
                v93 = 1;
                while ( v79 != -4096 )
                {
                  v72 = (unsigned int)(v93 + 1);
                  v77 = (v75 - 1) & (v93 + v77);
                  v78 = v76 + ((unsigned __int64)v77 << 6);
                  v79 = *(_QWORD *)(v78 + 24);
                  if ( (_BYTE *)v79 == v74 )
                    goto LABEL_119;
                  v93 = v72;
                }
              }
            }
            if ( v84 )
            {
LABEL_138:
              v71 = (unsigned __int64)(v83 + 1) | 4;
              v81 = (_QWORD *)v71;
              goto LABEL_123;
            }
LABEL_122:
            v81 = v83 + 18;
            v71 = (__int64)(v83 + 18);
LABEL_123:
            if ( v81 == v73 )
            {
              v28 = v109;
              break;
            }
          }
        }
        if ( v141 )
        {
          v65 = v139;
          v66 = (const char *)&v137;
          v70 = 0;
        }
        else
        {
          v65 = v132;
          v66 = &v132[16 * (unsigned int)v133];
        }
        while ( v70 )
        {
          if ( v66 == v65 )
            goto LABEL_95;
          v91 = (unsigned __int8 *)*((_QWORD *)v65 + 1);
          v92 = *(_QWORD *)v65;
          v65 += 16;
          sub_B59720(v28, v92, v91);
        }
        while ( v66 != v65 )
        {
          sub_B59720(v28, *((_QWORD *)v65 + 4), *((unsigned __int8 **)v65 + 5));
          v65 = (const char *)sub_220EF30((__int64)v65);
        }
LABEL_95:
        sub_27DBC40(v138);
        if ( v132 != (const char *)&v134 )
          _libc_free((unsigned __int64)v132);
      }
      else
      {
        v52 = 0;
        v53 = 32LL * (*(_DWORD *)(v28 + 4) & 0x7FFFFFF);
        if ( (*(_DWORD *)(v28 + 4) & 0x7FFFFFF) != 0 )
        {
          do
          {
            if ( (*(_BYTE *)(v28 + 7) & 0x40) != 0 )
              v54 = *(_QWORD *)(v28 - 8);
            else
              v54 = v28 - 32LL * (*(_DWORD *)(v28 + 4) & 0x7FFFFFF);
            v55 = v52 + v54;
            v56 = *(_BYTE **)v55;
            if ( **(_BYTE **)v55 > 0x1Cu )
            {
              v57 = *(unsigned int *)(a2 + 24);
              if ( (_DWORD)v57 )
              {
                v58 = *(_QWORD *)(a2 + 8);
                v59 = (v57 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
                v60 = v58 + ((unsigned __int64)v59 << 6);
                v61 = *(_BYTE **)(v60 + 24);
                if ( v56 == v61 )
                {
LABEL_76:
                  if ( v60 != v58 + (v57 << 6) )
                  {
                    v62 = *(_QWORD *)(v60 + 56);
                    v63 = *(_QWORD *)(v55 + 8);
                    **(_QWORD **)(v55 + 16) = v63;
                    if ( v63 )
                      *(_QWORD *)(v63 + 16) = *(_QWORD *)(v55 + 16);
                    *(_QWORD *)v55 = v62;
                    if ( v62 )
                    {
                      v64 = *(_QWORD *)(v62 + 16);
                      *(_QWORD *)(v55 + 8) = v64;
                      if ( v64 )
                        *(_QWORD *)(v64 + 16) = v55 + 8;
                      *(_QWORD *)(v55 + 16) = v62 + 16;
                      *(_QWORD *)(v62 + 16) = v55;
                    }
                  }
                }
                else
                {
                  v67 = 1;
                  while ( v61 != (_BYTE *)-4096LL )
                  {
                    v68 = v67 + 1;
                    v59 = (v57 - 1) & (v67 + v59);
                    v60 = v58 + ((unsigned __int64)v59 << 6);
                    v61 = *(_BYTE **)(v60 + 24);
                    if ( v56 == v61 )
                      goto LABEL_76;
                    v67 = v68;
                  }
                }
              }
            }
            v52 += 32;
          }
          while ( v52 != v53 );
        }
      }
      v118 = *(_QWORD *)(v118 + 8);
      if ( a5 == v118 )
      {
        if ( a5 != v108 + 48 )
        {
          if ( a5 )
            goto LABEL_100;
LABEL_101:
          if ( sub_B44020(v107) )
          {
            v85 = sub_AA6160(v108, a5);
            v86 = sub_AA7AD0(a7, a7 + 48);
            LOBYTE(v133) = 0;
            v87 = sub_B14600((__int64)v86, v85, (__int64)v132, 0, 0);
            v89 = v88;
            v90 = v87;
            if ( v87 != v88 )
            {
              while ( *(_BYTE *)(v90 + 32) )
              {
                v90 = *(_QWORD *)(v90 + 8);
                if ( v88 == v90 )
                  goto LABEL_102;
              }
              if ( v88 != v90 )
              {
LABEL_134:
                sub_27E28D0((__int64)&v120, v90);
                while ( 1 )
                {
                  v90 = *(_QWORD *)(v90 + 8);
                  if ( v89 == v90 )
                    break;
                  if ( !*(_BYTE *)(v90 + 32) )
                  {
                    if ( v89 != v90 )
                      goto LABEL_134;
                    goto LABEL_102;
                  }
                }
              }
            }
          }
        }
        goto LABEL_102;
      }
    }
    ++*(_QWORD *)a2;
    goto LABEL_36;
  }
  if ( a5 != v108 + 48 )
  {
LABEL_100:
    v107 = a5 - 24;
    goto LABEL_101;
  }
LABEL_102:
  sub_C7D6A0(v126, 16LL * v128, 8);
  if ( v129 != (__int64 *)v131 )
    _libc_free((unsigned __int64)v129);
}
