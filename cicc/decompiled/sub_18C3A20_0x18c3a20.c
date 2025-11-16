// Function: sub_18C3A20
// Address: 0x18c3a20
//
__int64 __fastcall sub_18C3A20(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        __m128 a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  bool v11; // zf
  __int64 *v12; // rdx
  __int64 *v13; // rcx
  _QWORD *v14; // rdi
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // rdi
  __int64 v21; // rax
  char v22; // al
  double v23; // xmm4_8
  double v24; // xmm5_8
  unsigned int v25; // r12d
  __int64 v26; // r14
  __int64 v27; // r15
  __int64 v28; // rdi
  __int64 v29; // rdi
  unsigned __int64 *v30; // r13
  unsigned __int64 *v31; // r14
  unsigned __int64 v32; // rdi
  unsigned __int64 *v33; // r13
  unsigned __int64 *v34; // r14
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // r8
  __int64 v37; // r13
  __int64 v38; // rbx
  unsigned __int64 v39; // rdi
  __int64 (__fastcall **v40)(__int64, __int64); // rdx
  __int64 (__fastcall **v41)(__int64, __int64); // rax
  _QWORD *v42; // rdi
  __int64 v43; // rax
  _QWORD *v44; // rdi
  __int64 v45; // rax
  _QWORD *v46; // rdi
  __int64 v47; // rax
  _QWORD *v48; // rdi
  __int64 v49; // rax
  char v50; // al
  double v51; // xmm4_8
  double v52; // xmm5_8
  __int64 v53; // rbx
  __int64 v54; // r13
  __int64 v55; // rdi
  __int64 v56; // rdi
  __int64 v57; // r13
  __int64 v58; // r14
  __m128i *v60; // rax
  __m128i **v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  char v65; // al
  char *v66; // rsi
  __m128i *v67; // rax
  __int64 v68; // r8
  __m128i *v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // rdi
  __int64 v72; // rax
  __int64 v73; // r10
  __int64 v74; // rsi
  int v75; // eax
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 *v78; // rsi
  __int64 v79; // rdx
  __int64 v80; // rcx
  int v81; // r8d
  int v82; // r9d
  __int64 v83; // [rsp+10h] [rbp-860h]
  __int64 v84; // [rsp+10h] [rbp-860h]
  __int64 v85; // [rsp+38h] [rbp-838h] BYREF
  __int64 *v86; // [rsp+40h] [rbp-830h] BYREF
  __int64 v87; // [rsp+48h] [rbp-828h] BYREF
  __int64 v88; // [rsp+50h] [rbp-820h] BYREF
  __int64 v89; // [rsp+58h] [rbp-818h]
  __m128i *v90; // [rsp+60h] [rbp-810h] BYREF
  __int64 v91; // [rsp+68h] [rbp-808h]
  __m128i v92; // [rsp+70h] [rbp-800h] BYREF
  _DWORD v93[4]; // [rsp+80h] [rbp-7F0h] BYREF
  __int64 (__fastcall *v94)(_QWORD *, _DWORD *, int); // [rsp+90h] [rbp-7E0h]
  __int64 (__fastcall *v95)(unsigned int *); // [rsp+98h] [rbp-7D8h]
  __m128i *v96; // [rsp+A0h] [rbp-7D0h] BYREF
  __int64 v97; // [rsp+A8h] [rbp-7C8h]
  __m128i v98; // [rsp+B0h] [rbp-7C0h] BYREF
  _DWORD v99[4]; // [rsp+C0h] [rbp-7B0h] BYREF
  __int64 (__fastcall *v100)(_QWORD *, _DWORD *, int); // [rsp+D0h] [rbp-7A0h]
  __int64 (__fastcall *v101)(unsigned int *); // [rsp+D8h] [rbp-798h]
  __m128i *v102; // [rsp+F0h] [rbp-780h] BYREF
  __int64 (__fastcall *v103)(__int64, __int64); // [rsp+F8h] [rbp-778h]
  __m128i v104; // [rsp+100h] [rbp-770h] BYREF
  __int64 *v105; // [rsp+110h] [rbp-760h]
  __int64 v106; // [rsp+118h] [rbp-758h]
  __int64 v107; // [rsp+120h] [rbp-750h]
  __int64 v108; // [rsp+128h] [rbp-748h]
  __int64 v109; // [rsp+130h] [rbp-740h]
  __int64 v110; // [rsp+138h] [rbp-738h]
  char v111; // [rsp+140h] [rbp-730h]
  __int64 **(__fastcall *v112)(__int64 ****, __int64 *); // [rsp+148h] [rbp-728h]
  __int64 **v113; // [rsp+150h] [rbp-720h]
  __int64 v114; // [rsp+158h] [rbp-718h]
  __int64 v115; // [rsp+160h] [rbp-710h]
  __int64 v116; // [rsp+168h] [rbp-708h]
  int v117; // [rsp+170h] [rbp-700h]
  __int64 v118; // [rsp+178h] [rbp-6F8h]
  __int64 v119; // [rsp+180h] [rbp-6F0h]
  __int64 v120; // [rsp+188h] [rbp-6E8h]
  int v121; // [rsp+198h] [rbp-6D8h] BYREF
  __int64 v122; // [rsp+1A0h] [rbp-6D0h]
  int *v123; // [rsp+1A8h] [rbp-6C8h]
  int *v124; // [rsp+1B0h] [rbp-6C0h]
  __int64 v125; // [rsp+1B8h] [rbp-6B8h]
  __int64 *v126; // [rsp+210h] [rbp-660h] BYREF
  __int64 (__fastcall *v127)(__int64, __int64); // [rsp+218h] [rbp-658h] BYREF
  _QWORD *v128; // [rsp+220h] [rbp-650h]
  __int64 (__fastcall **v129)(__int64, __int64); // [rsp+228h] [rbp-648h]
  __int64 (__fastcall **v130)(__int64, __int64); // [rsp+230h] [rbp-640h]
  __int64 v131; // [rsp+238h] [rbp-638h]
  unsigned __int64 v132; // [rsp+240h] [rbp-630h]
  __int64 v133; // [rsp+248h] [rbp-628h]
  __int64 v134; // [rsp+250h] [rbp-620h]
  __int64 v135; // [rsp+258h] [rbp-618h]
  char v136; // [rsp+260h] [rbp-610h]
  __int64 **(__fastcall *v137)(__int64 ****, __int64 *); // [rsp+268h] [rbp-608h] BYREF
  __int64 **v138; // [rsp+270h] [rbp-600h]
  __int64 **(__fastcall **v139)(__int64 ****, __int64 *); // [rsp+278h] [rbp-5F8h]
  __int64 **(__fastcall **v140)(__int64 ****, __int64 *); // [rsp+280h] [rbp-5F0h]
  __int64 v141; // [rsp+288h] [rbp-5E8h]
  int v142; // [rsp+290h] [rbp-5E0h]
  __int64 v143; // [rsp+298h] [rbp-5D8h] BYREF
  __int64 v144; // [rsp+2A0h] [rbp-5D0h]
  __int64 *v145; // [rsp+2A8h] [rbp-5C8h]
  __int64 *v146; // [rsp+2B0h] [rbp-5C0h]
  __int64 v147; // [rsp+2B8h] [rbp-5B8h] BYREF
  __int64 v148; // [rsp+2C0h] [rbp-5B0h]
  __int64 *v149; // [rsp+2C8h] [rbp-5A8h]
  __int64 *v150; // [rsp+2D0h] [rbp-5A0h] BYREF
  _QWORD *v151; // [rsp+2D8h] [rbp-598h]
  __int64 **v152; // [rsp+2E0h] [rbp-590h]
  __int64 **v153; // [rsp+2E8h] [rbp-588h]
  __int64 v154; // [rsp+2F0h] [rbp-580h]
  int v155; // [rsp+300h] [rbp-570h] BYREF
  _QWORD *v156; // [rsp+308h] [rbp-568h]
  int *v157; // [rsp+310h] [rbp-560h]
  int *v158; // [rsp+318h] [rbp-558h]
  __int64 v159; // [rsp+320h] [rbp-550h]
  _QWORD v160[2]; // [rsp+328h] [rbp-548h] BYREF
  unsigned __int64 *v161; // [rsp+338h] [rbp-538h]
  __int64 v162; // [rsp+340h] [rbp-530h]
  _BYTE v163[32]; // [rsp+348h] [rbp-528h] BYREF
  unsigned __int64 *v164; // [rsp+368h] [rbp-508h]
  __int64 v165; // [rsp+370h] [rbp-500h]
  _QWORD v166[5]; // [rsp+378h] [rbp-4F8h] BYREF
  _QWORD v167[10]; // [rsp+3A0h] [rbp-4D0h] BYREF
  char v168; // [rsp+3F0h] [rbp-480h]
  __int64 v169; // [rsp+3F8h] [rbp-478h]
  __int64 v170; // [rsp+6C0h] [rbp-1B0h]
  unsigned __int64 v171; // [rsp+6C8h] [rbp-1A8h]
  __int64 v172; // [rsp+728h] [rbp-148h]
  unsigned __int64 v173; // [rsp+730h] [rbp-140h]
  char v174; // [rsp+7C8h] [rbp-A8h]
  _QWORD v175[12]; // [rsp+7D0h] [rbp-A0h] BYREF
  char v176; // [rsp+830h] [rbp-40h]

  v11 = *(_BYTE *)(a1 + 153) == 0;
  v85 = 0;
  v86 = &v85;
  if ( !v11 )
  {
    v176 = 0;
    v129 = &v127;
    v130 = &v127;
    v134 = 0x2800000000LL;
    v139 = &v137;
    v140 = &v137;
    v145 = &v143;
    v146 = &v143;
    v152 = &v150;
    v153 = &v150;
    v157 = &v155;
    v167[0] = a1;
    v174 = 0;
    LODWORD(v127) = 0;
    v128 = 0;
    v131 = 0;
    v132 = 0;
    v133 = 0;
    LODWORD(v137) = 0;
    v138 = 0;
    v141 = 0;
    LODWORD(v143) = 0;
    v144 = 0;
    v147 = 0;
    LOWORD(v148) = 0;
    BYTE2(v148) = 0;
    LODWORD(v150) = 0;
    v151 = 0;
    v154 = 0;
    v155 = 0;
    v156 = 0;
    v158 = &v155;
    v161 = (unsigned __int64 *)v163;
    v162 = 0x400000000LL;
    v159 = 0;
    v160[0] = 0;
    v160[1] = 0;
    v164 = v166;
    v165 = 0;
    v166[0] = 0;
    v166[1] = 1;
    v166[3] = v160;
    if ( qword_4FAD9A8 )
    {
      sub_8FD6D0((__int64)&v90, "-wholeprogramdevirt-read-summary: ", &qword_4FAD9A0);
      if ( v91 == 0x3FFFFFFFFFFFFFFFLL || v91 == 4611686018427387902LL )
        goto LABEL_100;
      v60 = (__m128i *)sub_2241490(&v90, ": ", 2);
      v102 = &v104;
      if ( (__m128i *)v60->m128i_i64[0] == &v60[1] )
      {
        a3 = (__m128)_mm_loadu_si128(v60 + 1);
        v104 = (__m128i)a3;
      }
      else
      {
        v102 = (__m128i *)v60->m128i_i64[0];
        v104.m128i_i64[0] = v60[1].m128i_i64[0];
      }
      v103 = (__int64 (__fastcall *)(__int64, __int64))v60->m128i_i64[1];
      v60->m128i_i64[0] = (__int64)v60[1].m128i_i64;
      v60->m128i_i64[1] = 0;
      v60[1].m128i_i8[0] = 0;
      v96 = &v98;
      if ( v102 == &v104 )
      {
        a4 = (__m128)_mm_load_si128(&v104);
        v98 = (__m128i)a4;
      }
      else
      {
        v96 = v102;
        v98.m128i_i64[0] = v104.m128i_i64[0];
      }
      v99[0] = 1;
      v97 = (__int64)v103;
      v101 = sub_1872040;
      v100 = sub_1872500;
      if ( v90 != &v92 )
        j_j___libc_free_0(v90, v92.m128i_i64[0] + 1);
      v104.m128i_i16[0] = 260;
      v61 = &v102;
      v102 = (__m128i *)&qword_4FAD9A0;
      sub_16C2DE0((__int64)&v90, (__int64)&v102, 0xFFFFFFFFFFFFFFFFLL, 1, 0);
      v65 = v92.m128i_i8[0] & 1;
      if ( (v92.m128i_i8[0] & 1) != 0 && (v61 = (__m128i **)(unsigned int)v90, v64 = v91, (_DWORD)v90) )
      {
        sub_16BCB40(&v88, (int)v90, v91);
        v62 = v88 | 1;
        v11 = (v88 & 0xFFFFFFFFFFFFFFFELL) == 0;
        v88 |= 1uLL;
        if ( !v11 )
          sub_18B54C0((__int64)&v96, &v88, v62);
        v73 = 0;
        v65 = v92.m128i_i8[0] & 1;
      }
      else
      {
        v73 = (__int64)v90;
        v90 = 0;
      }
      if ( !v65 && v90 )
      {
        v84 = v73;
        (*(void (__fastcall **)(__m128i *, __m128i **, __int64, __int64, __int64))(v90->m128i_i64[0] + 8))(
          v90,
          v61,
          v62,
          v63,
          v64);
        v73 = v84;
      }
      v74 = *(_QWORD *)(v73 + 8);
      v83 = v73;
      sub_16E40A0((__int64)&v102, v74, *(_QWORD *)(v73 + 16) - v74, 0, 0, 0);
      sub_16E7420((__int64)&v102, v74);
      sub_16E3830((__int64)&v102);
      sub_1885EF0((char *)&v102, (__int64)&v126);
      sub_16E46D0((__int64)&v102);
      v75 = sub_16E4240((__int64)&v102);
      sub_16BCB40(&v88, v75, v76);
      v77 = v88;
      v88 = 0;
      v90 = (__m128i *)(v77 | 1);
      if ( (v77 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_18B54C0((__int64)&v96, (__int64 *)&v90, v77 | 1);
      sub_16E3EB0((__int64)&v102);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v83 + 8LL))(v83);
      if ( v100 )
        v100(v99, v99, 3);
      if ( v96 != &v98 )
        j_j___libc_free_0(v96, v98.m128i_i64[0] + 1);
    }
    if ( dword_4FADAC0 == 1 )
    {
      v12 = (__int64 *)&v126;
    }
    else
    {
      v12 = 0;
      v13 = (__int64 *)&v126;
      if ( dword_4FADAC0 == 2 )
      {
LABEL_5:
        v14 = (_QWORD *)*a2;
        v104.m128i_i64[1] = (__int64)v13;
        v105 = v12;
        v103 = sub_1833CC0;
        v102 = (__m128i *)a2;
        v104.m128i_i64[0] = (__int64)v167;
        v15 = sub_1643330(v14);
        v16 = (_QWORD *)*a2;
        v106 = v15;
        v17 = sub_16471D0(v16, 0);
        v18 = (_QWORD *)*a2;
        v107 = v17;
        v19 = sub_1643350(v18);
        v20 = (_QWORD *)*a2;
        v108 = v19;
        v109 = sub_1643360(v20);
        v21 = sub_1632FA0((__int64)a2);
        v110 = sub_15A9620(v21, *a2, 0);
        v22 = sub_18B58D0((__int64)v102);
        v113 = &v86;
        v111 = v22;
        v112 = sub_18B4D10;
        v114 = 0;
        v115 = 0;
        v116 = 0;
        v117 = 0;
        v118 = 0;
        v119 = 0;
        v120 = 0;
        v121 = 0;
        v122 = 0;
        v123 = &v121;
        v124 = &v121;
        v125 = 0;
        v25 = sub_18C04B0((__int64)&v102, a3, a4, a5, a6, v23, v24, a9, a10);
        sub_18B4F50(v122);
        v26 = v119;
        v27 = v118;
        if ( v119 != v118 )
        {
          do
          {
            sub_18B4EC0(*(_QWORD **)(v27 + 88));
            v28 = *(_QWORD *)(v27 + 48);
            if ( v28 )
              j_j___libc_free_0(v28, *(_QWORD *)(v27 + 64) - v28);
            v29 = *(_QWORD *)(v27 + 16);
            if ( v29 )
              j_j___libc_free_0(v29, *(_QWORD *)(v27 + 32) - v29);
            v27 += 120;
          }
          while ( v26 != v27 );
          v27 = v118;
        }
        if ( v27 )
          j_j___libc_free_0(v27, v120 - v27);
        j___libc_free_0(v115);
        if ( !qword_4FAD888 )
        {
LABEL_15:
          v30 = v161;
          v31 = &v161[(unsigned int)v162];
          if ( v161 != v31 )
          {
            do
            {
              v32 = *v30++;
              _libc_free(v32);
            }
            while ( v31 != v30 );
          }
          v33 = v164;
          v34 = &v164[2 * (unsigned int)v165];
          if ( v164 != v34 )
          {
            do
            {
              v35 = *v33;
              v33 += 2;
              _libc_free(v35);
            }
            while ( v34 != v33 );
            v34 = v164;
          }
          if ( v34 != v166 )
            _libc_free((unsigned __int64)v34);
          if ( v161 != (unsigned __int64 *)v163 )
            _libc_free((unsigned __int64)v161);
          sub_18B6710(v156);
          sub_18B6710(v151);
          sub_18B5120(v144);
          sub_18B5A40(v138);
          if ( HIDWORD(v133) )
          {
            v36 = v132;
            if ( (_DWORD)v133 )
            {
              v37 = 8LL * (unsigned int)v133;
              v38 = 0;
              do
              {
                v39 = *(_QWORD *)(v36 + v38);
                if ( v39 != -8 && v39 )
                {
                  _libc_free(v39);
                  v36 = v132;
                }
                v38 += 8;
              }
              while ( v38 != v37 );
            }
          }
          else
          {
            v36 = v132;
          }
          _libc_free(v36);
          sub_18B4D90(v128);
          if ( v176 )
            goto LABEL_50;
          goto LABEL_42;
        }
        sub_8FD6D0((__int64)&v102, "-wholeprogramdevirt-write-summary: ", &qword_4FAD880);
        if ( v103 != (__int64 (__fastcall *)(__int64, __int64))0x3FFFFFFFFFFFFFFFLL
          && (_QWORD)v103 != 4611686018427387902LL )
        {
          v66 = ": ";
          v67 = (__m128i *)sub_2241490(&v102, ": ", 2);
          v69 = v67 + 1;
          v96 = &v98;
          if ( (__m128i *)v67->m128i_i64[0] == &v67[1] )
          {
            v98 = _mm_loadu_si128(v67 + 1);
          }
          else
          {
            v96 = (__m128i *)v67->m128i_i64[0];
            v98.m128i_i64[0] = v67[1].m128i_i64[0];
          }
          v70 = v67->m128i_i64[1];
          v97 = v70;
          v67->m128i_i64[0] = (__int64)v69;
          v67->m128i_i64[1] = 0;
          v67[1].m128i_i8[0] = 0;
          v90 = &v92;
          if ( v96 == &v98 )
          {
            v92 = _mm_load_si128(&v98);
          }
          else
          {
            v90 = v96;
            v92.m128i_i64[0] = v98.m128i_i64[0];
          }
          v71 = (__int64)v102;
          v93[0] = 1;
          v91 = v97;
          v95 = sub_1872040;
          v94 = sub_1872500;
          if ( v102 != &v104 )
          {
            v66 = (char *)(v104.m128i_i64[0] + 1);
            j_j___libc_free_0(v102, v104.m128i_i64[0] + 1);
          }
          LODWORD(v88) = 0;
          v89 = sub_2241E40(v71, v66, v69, v70, v68);
          sub_16E8AF0((__int64)&v96, (_BYTE *)qword_4FAD880, qword_4FAD888, (__int64)&v88, 1u);
          sub_16BCB40(&v87, v88, v89);
          v72 = v87;
          v87 = 0;
          v102 = (__m128i *)(v72 | 1);
          if ( (v72 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_18B54C0((__int64)&v90, (__int64 *)&v102, v72 | 1);
          sub_16E4AB0((__int64)&v102, (__int64)&v96, 0, 70);
          nullsub_622();
          v78 = 0;
          if ( (unsigned __int8)sub_16E4B20() )
          {
            sub_16E3D10((__int64)&v102, 0, v79, v80, v81, v82);
            v78 = (__int64 *)&v126;
            sub_1885EF0((char *)&v102, (__int64)&v126);
            sub_16E3410((__int64)&v102);
            nullsub_628();
          }
          sub_16E4BA0((__int64)&v102);
          sub_16E3E40(&v102);
          sub_16E7C30((int *)&v96, (__int64)v78);
          if ( v94 )
            v94(v93, v93, 3);
          if ( v90 != &v92 )
            j_j___libc_free_0(v90, v92.m128i_i64[0] + 1);
          goto LABEL_15;
        }
LABEL_100:
        sub_4262D8((__int64)"basic_string::append");
      }
    }
    v13 = 0;
    goto LABEL_5;
  }
  v40 = *(__int64 (__fastcall ***)(__int64, __int64))(a1 + 160);
  v167[0] = a1;
  v41 = *(__int64 (__fastcall ***)(__int64, __int64))(a1 + 168);
  v42 = (_QWORD *)*a2;
  v126 = a2;
  v127 = sub_1833CC0;
  v128 = v167;
  v129 = v40;
  v174 = 0;
  v176 = 0;
  v130 = v41;
  v43 = sub_1643330(v42);
  v44 = (_QWORD *)*a2;
  v131 = v43;
  v45 = sub_16471D0(v44, 0);
  v46 = (_QWORD *)*a2;
  v132 = v45;
  v47 = sub_1643350(v46);
  v48 = (_QWORD *)*a2;
  v133 = v47;
  v134 = sub_1643360(v48);
  v49 = sub_1632FA0((__int64)a2);
  v135 = sub_15A9620(v49, *a2, 0);
  v50 = sub_18B58D0((__int64)v126);
  v138 = &v86;
  v136 = v50;
  v137 = sub_18B4D10;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  LODWORD(v147) = 0;
  v148 = 0;
  v149 = &v147;
  v150 = &v147;
  v151 = 0;
  v25 = sub_18C04B0((__int64)&v126, a3, a4, a5, a6, v51, v52, a9, a10);
  sub_18B4F50(v148);
  v53 = v144;
  v54 = v143;
  if ( v144 != v143 )
  {
    do
    {
      sub_18B4EC0(*(_QWORD **)(v54 + 88));
      v55 = *(_QWORD *)(v54 + 48);
      if ( v55 )
        j_j___libc_free_0(v55, *(_QWORD *)(v54 + 64) - v55);
      v56 = *(_QWORD *)(v54 + 16);
      if ( v56 )
        j_j___libc_free_0(v56, *(_QWORD *)(v54 + 32) - v56);
      v54 += 120;
    }
    while ( v53 != v54 );
    v54 = v143;
  }
  if ( v54 )
    j_j___libc_free_0(v54, (char *)v145 - v54);
  j___libc_free_0(v140);
  if ( v176 )
  {
LABEL_50:
    sub_134CA00(v175);
    if ( !v174 )
      goto LABEL_43;
    goto LABEL_51;
  }
LABEL_42:
  if ( !v174 )
    goto LABEL_43;
LABEL_51:
  if ( v173 != v172 )
    _libc_free(v173);
  if ( v171 != v170 )
    _libc_free(v171);
  if ( (v168 & 1) == 0 )
    j___libc_free_0(v169);
LABEL_43:
  v57 = v85;
  if ( v85 )
  {
    v58 = *(_QWORD *)(v85 + 16);
    if ( v58 )
    {
      sub_1368A00(*(__int64 **)(v85 + 16));
      j_j___libc_free_0(v58, 8);
    }
    j_j___libc_free_0(v57, 24);
  }
  return v25;
}
