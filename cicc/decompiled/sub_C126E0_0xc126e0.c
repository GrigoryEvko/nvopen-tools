// Function: sub_C126E0
// Address: 0xc126e0
//
_QWORD *__fastcall sub_C126E0(__int64 *a1, void (__fastcall *a2)(__int64, void **), __int64 a3)
{
  _QWORD *result; // rax
  __int64 v5; // r14
  _BYTE *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rcx
  __int64 (__fastcall *v11)(_QWORD *); // r12
  __int64 v12; // r12
  __int64 (__fastcall *v13)(__int64, _QWORD *, _BYTE *); // r15
  __int64 v14; // rsi
  __int64 v15; // rcx
  _QWORD *v16; // rdx
  __int64 (__fastcall *v17)(_QWORD *, const char *, _QWORD, const char *, _QWORD); // r15
  _QWORD *v18; // r15
  __int64 (*v19)(void); // rax
  __int64 (__fastcall *v20)(_QWORD *, _QWORD, _QWORD); // r8
  void (__fastcall *v21)(void **); // rax
  _QWORD *v22; // rsi
  __int64 v23; // r14
  __int64 (__fastcall *v24)(_QWORD *, __int64, __int64, _BYTE *); // rax
  __m128i v25; // xmm0
  __m128i v26; // xmm1
  void (__fastcall *v27)(__m128i *, __m128i *, __int64); // rcx
  void (__fastcall *v28)(_QWORD *, _QWORD *, __int64); // rax
  void (*v29)(); // rax
  void **v30; // rsi
  __int64 v31; // rbx
  __int64 v32; // r14
  __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // r8
  __int64 v36; // r14
  __int64 v37; // rbx
  _QWORD *v38; // rdi
  _QWORD *v39; // rbx
  _QWORD *v40; // r13
  __int64 v41; // rbx
  __int64 v42; // r13
  __int64 v43; // rdi
  __int64 (__fastcall *v44)(_QWORD *); // rax
  _QWORD *v45; // rdi
  _QWORD *v46; // rdi
  _QWORD *v47; // rdi
  _QWORD *v48; // rdi
  _QWORD *v49; // rbx
  _QWORD *v50; // r13
  __int64 (__fastcall *v51)(__int64); // rax
  _QWORD *v52; // rbx
  _QWORD *v53; // r13
  __int64 v54; // r8
  _QWORD *v55; // rbx
  _QWORD *v56; // r13
  __int64 v57; // rsi
  __int64 v58; // rbx
  __int64 v59; // r13
  __int64 v60; // rdi
  __int64 (__fastcall *v61)(_QWORD *); // rax
  _QWORD *v62; // rdi
  _QWORD *v63; // rdi
  _QWORD *v64; // rdi
  _QWORD *v65; // rdi
  _QWORD *v66; // rbx
  _QWORD *v67; // r13
  _QWORD *v68; // rbx
  _QWORD *v69; // r13
  __int64 v70; // rbx
  __int64 v71; // r14
  __int64 v72; // rdi
  __int64 v73; // r14
  __int64 v74; // rbx
  _QWORD *v75; // rdi
  __int64 v76; // [rsp+18h] [rbp-D98h]
  __int64 v77; // [rsp+30h] [rbp-D80h]
  _QWORD *v78; // [rsp+38h] [rbp-D78h]
  void *v79; // [rsp+38h] [rbp-D78h]
  _QWORD *v82; // [rsp+58h] [rbp-D58h]
  __int64 v83; // [rsp+58h] [rbp-D58h]
  __int64 v84; // [rsp+68h] [rbp-D48h] BYREF
  __int64 v85[2]; // [rsp+70h] [rbp-D40h] BYREF
  _QWORD v86[2]; // [rsp+80h] [rbp-D30h] BYREF
  _QWORD v87[2]; // [rsp+90h] [rbp-D20h] BYREF
  void (__fastcall *v88)(_QWORD *, _QWORD *, __int64); // [rsp+A0h] [rbp-D10h]
  _BYTE *(__fastcall *v89)(__int64 **, __int64, char *); // [rsp+A8h] [rbp-D08h]
  __m128i v90; // [rsp+B0h] [rbp-D00h] BYREF
  void (__fastcall *v91)(__m128i *, __m128i *, __int64); // [rsp+C0h] [rbp-CF0h]
  _BYTE *(__fastcall *v92)(__int64 **, __int64, char *); // [rsp+C8h] [rbp-CE8h]
  _QWORD *v93; // [rsp+D0h] [rbp-CE0h] BYREF
  _QWORD *v94; // [rsp+D8h] [rbp-CD8h]
  _QWORD v95[6]; // [rsp+E0h] [rbp-CD0h] BYREF
  __int64 v96; // [rsp+110h] [rbp-CA0h] BYREF
  __int64 v97; // [rsp+118h] [rbp-C98h]
  __int64 v98; // [rsp+120h] [rbp-C90h]
  _QWORD *v99; // [rsp+128h] [rbp-C88h]
  _QWORD *v100; // [rsp+130h] [rbp-C80h]
  __int64 v101; // [rsp+138h] [rbp-C78h]
  __int64 v102; // [rsp+140h] [rbp-C70h]
  __int64 v103; // [rsp+148h] [rbp-C68h]
  _BYTE v104[32]; // [rsp+150h] [rbp-C60h] BYREF
  _QWORD *v105; // [rsp+170h] [rbp-C40h]
  _QWORD v106[2]; // [rsp+180h] [rbp-C30h] BYREF
  _QWORD *v107; // [rsp+190h] [rbp-C20h]
  _QWORD v108[2]; // [rsp+1A0h] [rbp-C10h] BYREF
  _QWORD *v109; // [rsp+1B0h] [rbp-C00h]
  _QWORD v110[2]; // [rsp+1C0h] [rbp-BF0h] BYREF
  _QWORD *v111; // [rsp+1D0h] [rbp-BE0h]
  _QWORD v112[2]; // [rsp+1E0h] [rbp-BD0h] BYREF
  _QWORD *v113; // [rsp+1F0h] [rbp-BC0h]
  _QWORD v114[2]; // [rsp+200h] [rbp-BB0h] BYREF
  _QWORD *v115; // [rsp+210h] [rbp-BA0h]
  _QWORD v116[2]; // [rsp+220h] [rbp-B90h] BYREF
  _QWORD *v117; // [rsp+230h] [rbp-B80h]
  _QWORD *v118; // [rsp+238h] [rbp-B78h]
  __int64 v119; // [rsp+240h] [rbp-B70h]
  void *v120; // [rsp+250h] [rbp-B60h] BYREF
  _QWORD *v121; // [rsp+258h] [rbp-B58h]
  __int16 v122; // [rsp+270h] [rbp-B40h]
  __int64 v123; // [rsp+380h] [rbp-A30h]
  unsigned int v124; // [rsp+388h] [rbp-A28h]
  int v125; // [rsp+38Ch] [rbp-A24h]
  __int64 v126; // [rsp+3A0h] [rbp-A10h]
  unsigned int v127; // [rsp+3B0h] [rbp-A00h]
  __int64 v128; // [rsp+3B8h] [rbp-9F8h]
  unsigned int v129; // [rsp+3C0h] [rbp-9F0h]
  char v130; // [rsp+3C8h] [rbp-9E8h] BYREF
  _QWORD v131[2]; // [rsp+3D0h] [rbp-9E0h] BYREF
  _QWORD v132[13]; // [rsp+3E0h] [rbp-9D0h] BYREF
  __m128i v133; // [rsp+448h] [rbp-968h] BYREF
  void (__fastcall *v134)(__m128i *, __m128i *, __int64); // [rsp+458h] [rbp-958h]
  _BYTE *(__fastcall *v135)(__int64 **, __int64, char *); // [rsp+460h] [rbp-950h]
  void *v136; // [rsp+478h] [rbp-938h]

  result = (_QWORD *)sub_B6F970(*a1);
  if ( !*((_BYTE *)result + 16) )
  {
    result = (_QWORD *)a1[11];
    v5 = a1[12];
    v78 = result;
    if ( v5 )
    {
      v6 = (_BYTE *)a1[29];
      v7 = a1[30];
      v85[0] = (__int64)v86;
      v93 = v95;
      v85[1] = 0;
      LOBYTE(v86[0]) = 0;
      sub_C11DF0((__int64 *)&v93, v6, (__int64)&v6[v7]);
      v95[2] = a1[33];
      v95[3] = a1[34];
      v95[4] = a1[35];
      v8 = sub_C0D4F0((__int64)&v93, v85);
      v9 = v93;
      v10 = v8;
      v82 = (_QWORD *)v8;
      result = v94;
      v11 = *(__int64 (__fastcall **)(_QWORD *))(v10 + 80);
      if ( !v11 )
        goto LABEL_100;
      v121 = v94;
      v120 = v93;
      v122 = 261;
      sub_CC9F70(v131, &v120);
      v12 = v11(v131);
      result = v132;
      if ( (_QWORD *)v131[0] != v132 )
        result = (_QWORD *)j_j___libc_free_0(v131[0], v132[0] + 1LL);
      if ( !v12 )
      {
LABEL_99:
        v9 = v93;
LABEL_100:
        if ( v9 == v95 )
          goto LABEL_102;
        goto LABEL_101;
      }
      sub_EA1890(v104);
      v13 = (__int64 (__fastcall *)(__int64, _QWORD *, _BYTE *))v82[6];
      if ( !v13 )
        goto LABEL_133;
      v120 = v93;
      v122 = 261;
      v121 = v94;
      sub_CC9F70(v131, &v120);
      v14 = (__int64)v131;
      v77 = v13(v12, v131, v104);
      if ( (_QWORD *)v131[0] != v132 )
      {
        v14 = v132[0] + 1LL;
        j_j___libc_free_0(v131[0], v132[0] + 1LL);
      }
      if ( !v77 )
        goto LABEL_133;
      v15 = (__int64)v93;
      v16 = v94;
      v17 = (__int64 (__fastcall *)(_QWORD *, const char *, _QWORD, const char *, _QWORD))v82[11];
      if ( !v17 )
        goto LABEL_132;
      v120 = v93;
      v122 = 261;
      v121 = v94;
      sub_CC9F70(v131, &v120);
      v14 = (__int64)byte_3F871B3;
      v18 = (_QWORD *)v17(v131, byte_3F871B3, 0, byte_3F871B3, 0);
      if ( (_QWORD *)v131[0] != v132 )
      {
        v14 = v132[0] + 1LL;
        j_j___libc_free_0(v131[0], v132[0] + 1LL);
      }
      if ( !v18 )
      {
LABEL_132:
        (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64))(*(_QWORD *)v77 + 8LL))(v77, v14, v16, v15);
LABEL_133:
        v66 = v118;
        v67 = v117;
        if ( v118 != v117 )
        {
          do
          {
            if ( (_QWORD *)*v67 != v67 + 2 )
              j_j___libc_free_0(*v67, v67[2] + 1LL);
            v67 += 4;
          }
          while ( v66 != v67 );
          v67 = v117;
        }
        if ( v67 )
          j_j___libc_free_0(v67, v119 - (_QWORD)v67);
        if ( v115 != v116 )
          j_j___libc_free_0(v115, v116[0] + 1LL);
        if ( v113 != v114 )
          j_j___libc_free_0(v113, v114[0] + 1LL);
        if ( v111 != v112 )
          j_j___libc_free_0(v111, v112[0] + 1LL);
        if ( v109 != v110 )
          j_j___libc_free_0(v109, v110[0] + 1LL);
        if ( v107 != v108 )
          j_j___libc_free_0(v107, v108[0] + 1LL);
        if ( v105 != v106 )
          j_j___libc_free_0(v105, v106[0] + 1LL);
        v51 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL);
        if ( v51 == sub_C11FA0 )
        {
          v68 = *(_QWORD **)(v12 + 232);
          v69 = *(_QWORD **)(v12 + 224);
          *(_QWORD *)v12 = &unk_49E3560;
          if ( v68 != v69 )
          {
            do
            {
              if ( *v69 )
                j_j___libc_free_0(*v69, v69[2] - *v69);
              v69 += 3;
            }
            while ( v68 != v69 );
            v69 = *(_QWORD **)(v12 + 224);
          }
          if ( v69 )
            j_j___libc_free_0(v69, *(_QWORD *)(v12 + 240) - (_QWORD)v69);
          sub_C7D6A0(*(_QWORD *)(v12 + 200), 8LL * *(unsigned int *)(v12 + 216), 4);
          sub_C7D6A0(*(_QWORD *)(v12 + 168), 8LL * *(unsigned int *)(v12 + 184), 4);
          result = (_QWORD *)j_j___libc_free_0(v12, 248);
          v9 = v93;
          if ( v93 == v95 )
            goto LABEL_102;
LABEL_101:
          result = (_QWORD *)j_j___libc_free_0(v9, v95[0] + 1LL);
LABEL_102:
          if ( (_QWORD *)v85[0] != v86 )
            return (_QWORD *)j_j___libc_free_0(v85[0], v86[0] + 1LL);
          return result;
        }
LABEL_179:
        result = (_QWORD *)v51(v12);
        goto LABEL_99;
      }
      v19 = (__int64 (*)(void))v82[8];
      if ( !v19 || (v76 = v19()) == 0 )
      {
LABEL_122:
        v61 = *(__int64 (__fastcall **)(_QWORD *))(*v18 + 8LL);
        if ( v61 == sub_C12070 )
        {
          v62 = (_QWORD *)v18[34];
          *v18 = &unk_49E41D0;
          if ( v62 != v18 + 36 )
            j_j___libc_free_0(v62, v18[36] + 1LL);
          v63 = (_QWORD *)v18[12];
          if ( v63 != v18 + 14 )
            j_j___libc_free_0(v63, v18[14] + 1LL);
          v64 = (_QWORD *)v18[8];
          if ( v64 != v18 + 10 )
            j_j___libc_free_0(v64, v18[10] + 1LL);
          v65 = (_QWORD *)v18[1];
          if ( v65 != v18 + 3 )
            j_j___libc_free_0(v65, v18[3] + 1LL);
          v14 = 304;
          j_j___libc_free_0(v18, 304);
        }
        else
        {
          v61(v18);
        }
        goto LABEL_132;
      }
      sub_C7DA90(&v84, v78, v5, "<inline asm>", 12, 1);
      v96 = 0;
      v97 = 0;
      v131[0] = v84;
      v98 = 0;
      v99 = 0;
      v100 = 0;
      v101 = 0;
      v102 = 0;
      v103 = 0;
      v84 = 0;
      v131[1] = 0;
      v132[0] = 0;
      sub_C12520(&v96, 0, (__int64)v131);
      sub_C8EE20(v131, 0);
      sub_E64450((unsigned int)v131, (unsigned int)&v93, v77, v12, (_DWORD)v18, (unsigned int)&v96, 0, 1, 0);
      v20 = (__int64 (__fastcall *)(_QWORD *, _QWORD, _QWORD))v82[7];
      if ( v20 )
      {
        v79 = (void *)v20(v131, 0, 0);
      }
      else
      {
        v79 = (void *)sub_22077B0(928);
        if ( v79 )
        {
          memset(v79, 0, 0x3A0u);
          *(_QWORD *)v79 = &unk_49E2FA0;
        }
        sub_E89910(v79, v131, 0, 0);
      }
      v136 = v79;
      sub_C149C0(&v120, v131, a1);
      v21 = (void (__fastcall *)(void **))v82[23];
      if ( v21 )
        v21(&v120);
      v22 = v131;
      v23 = sub_EA87E0(&v96, v131, &v120, v77, 0);
      v24 = (__int64 (__fastcall *)(_QWORD *, __int64, __int64, _BYTE *))v82[14];
      if ( v24 && (v22 = (_QWORD *)v23, (v83 = v24(v18, v23, v76, v104)) != 0) )
      {
        v87[0] = a1;
        v89 = sub_C11D80;
        v88 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))sub_C11CB0;
        v91 = 0;
        sub_C11CB0(&v90, v87, 2);
        v25 = _mm_loadu_si128(&v90);
        v26 = _mm_loadu_si128(&v133);
        v27 = v134;
        v92 = v135;
        v28 = v88;
        v91 = v134;
        v134 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v88;
        v135 = v89;
        v90 = v26;
        v133 = v25;
        if ( v91 )
        {
          v27(&v90, &v90, 3);
          v28 = v88;
        }
        if ( v28 )
          v28(v87, v87, 3);
        v29 = *(void (**)())(*(_QWORD *)v23 + 72LL);
        if ( v29 != nullsub_97 )
          ((void (__fastcall *)(__int64, _QWORD))v29)(v23, 0);
        sub_ECD790(v23, v83);
        v22 = 0;
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v23 + 80LL))(v23, 0, 0) )
        {
          v30 = &v120;
          a2(a3, &v120);
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v83 + 8LL))(v83);
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
          v31 = v128;
          v32 = v128 + 32LL * v129;
          v120 = &unk_49DB460;
          if ( v128 != v32 )
          {
            do
            {
              v33 = *(_QWORD *)(v32 - 24);
              v32 -= 32;
              if ( v33 )
              {
                v30 = (void **)(*(_QWORD *)(v32 + 24) - v33);
                j_j___libc_free_0(v33, v30);
              }
            }
            while ( v31 != v32 );
            v32 = v128;
          }
          if ( (char *)v32 != &v130 )
            _libc_free(v32, v30);
          v34 = 16LL * v127;
          sub_C7D6A0(v126, v34, 8);
          if ( v125 )
          {
            v35 = v123;
            if ( v124 )
            {
              v36 = 8LL * v124;
              v37 = 0;
              do
              {
                v38 = *(_QWORD **)(v35 + v37);
                if ( v38 != (_QWORD *)-8LL && v38 )
                {
                  v34 = *v38 + 17LL;
                  sub_C7D6A0(v38, v34, 8);
                  v35 = v123;
                }
                v37 += 8;
              }
              while ( v36 != v37 );
            }
          }
          else
          {
            v35 = v123;
          }
          _libc_free(v35, v34);
          sub_E98B30(&v120);
          if ( v79 )
            (*(void (__fastcall **)(void *))(*(_QWORD *)v79 + 8LL))(v79);
          sub_E68A10(v131);
          v39 = v100;
          v40 = v99;
          if ( v100 != v99 )
          {
            do
            {
              if ( (_QWORD *)*v40 != v40 + 2 )
              {
                v34 = v40[2] + 1LL;
                j_j___libc_free_0(*v40, v34);
              }
              v40 += 4;
            }
            while ( v39 != v40 );
            v40 = v99;
          }
          if ( v40 )
          {
            v34 = v101 - (_QWORD)v40;
            j_j___libc_free_0(v40, v101 - (_QWORD)v40);
          }
          v41 = v97;
          v42 = v96;
          if ( v97 != v96 )
          {
            do
            {
              v43 = v42;
              v42 += 24;
              sub_C8EE20(v43, v34);
            }
            while ( v41 != v42 );
            v42 = v96;
          }
          if ( v42 )
            j_j___libc_free_0(v42, v98 - v42);
          if ( v84 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v84 + 8LL))(v84);
          j_j___libc_free_0(v76, 48);
          v44 = *(__int64 (__fastcall **)(_QWORD *))(*v18 + 8LL);
          if ( v44 == sub_C12070 )
          {
            v45 = (_QWORD *)v18[34];
            *v18 = &unk_49E41D0;
            if ( v45 != v18 + 36 )
              j_j___libc_free_0(v45, v18[36] + 1LL);
            v46 = (_QWORD *)v18[12];
            if ( v46 != v18 + 14 )
              j_j___libc_free_0(v46, v18[14] + 1LL);
            v47 = (_QWORD *)v18[8];
            if ( v47 != v18 + 10 )
              j_j___libc_free_0(v47, v18[10] + 1LL);
            v48 = (_QWORD *)v18[1];
            if ( v48 != v18 + 3 )
              j_j___libc_free_0(v48, v18[3] + 1LL);
            j_j___libc_free_0(v18, 304);
          }
          else
          {
            v44(v18);
          }
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v77 + 8LL))(v77);
          v49 = v118;
          v50 = v117;
          if ( v118 != v117 )
          {
            do
            {
              if ( (_QWORD *)*v50 != v50 + 2 )
                j_j___libc_free_0(*v50, v50[2] + 1LL);
              v50 += 4;
            }
            while ( v49 != v50 );
            v50 = v117;
          }
          if ( v50 )
            j_j___libc_free_0(v50, v119 - (_QWORD)v50);
          if ( v115 != v116 )
            j_j___libc_free_0(v115, v116[0] + 1LL);
          if ( v113 != v114 )
            j_j___libc_free_0(v113, v114[0] + 1LL);
          if ( v111 != v112 )
            j_j___libc_free_0(v111, v112[0] + 1LL);
          if ( v109 != v110 )
            j_j___libc_free_0(v109, v110[0] + 1LL);
          if ( v107 != v108 )
            j_j___libc_free_0(v107, v108[0] + 1LL);
          if ( v105 != v106 )
            j_j___libc_free_0(v105, v106[0] + 1LL);
          v51 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL);
          if ( v51 == sub_C11FA0 )
          {
            v52 = *(_QWORD **)(v12 + 232);
            v53 = *(_QWORD **)(v12 + 224);
            *(_QWORD *)v12 = &unk_49E3560;
            if ( v52 != v53 )
            {
              do
              {
                if ( *v53 )
                  j_j___libc_free_0(*v53, v53[2] - *v53);
                v53 += 3;
              }
              while ( v52 != v53 );
              v53 = *(_QWORD **)(v12 + 224);
            }
            if ( v53 )
              j_j___libc_free_0(v53, *(_QWORD *)(v12 + 240) - (_QWORD)v53);
            sub_C7D6A0(*(_QWORD *)(v12 + 200), 8LL * *(unsigned int *)(v12 + 216), 4);
            sub_C7D6A0(*(_QWORD *)(v12 + 168), 8LL * *(unsigned int *)(v12 + 184), 4);
            result = (_QWORD *)j_j___libc_free_0(v12, 248);
            goto LABEL_99;
          }
          goto LABEL_179;
        }
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v83 + 8LL))(v83);
      }
      else if ( !v23 )
      {
        goto LABEL_164;
      }
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
LABEL_164:
      v70 = v128;
      v71 = v128 + 32LL * v129;
      v120 = &unk_49DB460;
      if ( v128 != v71 )
      {
        do
        {
          v72 = *(_QWORD *)(v71 - 24);
          v71 -= 32;
          if ( v72 )
          {
            v22 = (_QWORD *)(*(_QWORD *)(v71 + 24) - v72);
            j_j___libc_free_0(v72, v22);
          }
        }
        while ( v70 != v71 );
        v71 = v128;
      }
      if ( (char *)v71 != &v130 )
        _libc_free(v71, v22);
      v57 = 16LL * v127;
      sub_C7D6A0(v126, v57, 8);
      if ( v125 )
      {
        v54 = v123;
        if ( v124 )
        {
          v73 = 8LL * v124;
          v74 = 0;
          do
          {
            v75 = *(_QWORD **)(v54 + v74);
            if ( v75 != (_QWORD *)-8LL && v75 )
            {
              v57 = *v75 + 17LL;
              sub_C7D6A0(v75, v57, 8);
              v54 = v123;
            }
            v74 += 8;
          }
          while ( v73 != v74 );
        }
      }
      else
      {
        v54 = v123;
      }
      _libc_free(v54, v57);
      sub_E98B30(&v120);
      if ( v79 )
        (*(void (__fastcall **)(void *))(*(_QWORD *)v79 + 8LL))(v79);
      sub_E68A10(v131);
      v55 = v100;
      v56 = v99;
      if ( v100 != v99 )
      {
        do
        {
          if ( (_QWORD *)*v56 != v56 + 2 )
          {
            v57 = v56[2] + 1LL;
            j_j___libc_free_0(*v56, v57);
          }
          v56 += 4;
        }
        while ( v55 != v56 );
        v56 = v99;
      }
      if ( v56 )
      {
        v57 = v101 - (_QWORD)v56;
        j_j___libc_free_0(v56, v101 - (_QWORD)v56);
      }
      v58 = v97;
      v59 = v96;
      if ( v97 != v96 )
      {
        do
        {
          v60 = v59;
          v59 += 24;
          sub_C8EE20(v60, v57);
        }
        while ( v58 != v59 );
        v59 = v96;
      }
      if ( v59 )
        j_j___libc_free_0(v59, v98 - v59);
      if ( v84 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v84 + 8LL))(v84);
      v14 = 48;
      j_j___libc_free_0(v76, 48);
      goto LABEL_122;
    }
  }
  return result;
}
