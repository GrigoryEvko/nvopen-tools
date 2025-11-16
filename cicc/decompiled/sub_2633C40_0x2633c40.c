// Function: sub_2633C40
// Address: 0x2633c40
//
void __fastcall sub_2633C40(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  int v3; // ecx
  void *v4; // rdi
  __int64 v5; // rax
  __int64 v6; // r12
  __m128i v7; // rax
  __m128i v8; // xmm1
  __m128i v9; // xmm0
  __int64 v10; // rsi
  __int64 v11; // rcx
  int v12; // edx
  void *v13; // rax
  __m128i v14; // xmm2
  __m128i v15; // xmm3
  __m128i v16; // xmm4
  __int64 v17; // rax
  int *v18; // rcx
  int *v19; // rdx
  int *v20; // rax
  int *v21; // rdx
  void *v22; // rcx
  __int64 v23; // rax
  unsigned __int64 *v24; // rbx
  unsigned __int64 *v25; // r13
  int v26; // r11d
  unsigned int v27; // edx
  __int64 *v28; // rdi
  __int64 *v29; // rcx
  __int64 v30; // r8
  _QWORD *v31; // rdi
  _BYTE *v32; // rdi
  char *v33; // rsi
  __int64 v34; // rax
  int v35; // ecx
  _QWORD *v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // rbx
  _QWORD *v39; // r13
  unsigned __int64 v40; // r14
  unsigned __int64 v41; // r15
  unsigned __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rax
  _QWORD *v45; // r13
  _QWORD *v46; // r12
  int v47; // r11d
  unsigned int v48; // edx
  __int64 *v49; // rdi
  __int64 *v50; // rcx
  __int64 v51; // r8
  _QWORD *v52; // rdi
  _BYTE *v53; // rdi
  __int64 v54; // rsi
  __int64 v55; // rax
  int v56; // esi
  int v57; // ecx
  _QWORD *v58; // rax
  __int64 v59; // rax
  _QWORD *v60; // r13
  _QWORD *v61; // r14
  unsigned __int64 v62; // rbx
  unsigned __int64 v63; // r12
  unsigned __int64 v64; // rdi
  __int64 v65; // rax
  unsigned __int64 *v66; // rbx
  unsigned __int64 *v67; // r12
  unsigned __int64 *v68; // rbx
  unsigned __int64 *v69; // r12
  __int64 v70; // r8
  __int64 *v71; // rdi
  __int64 *v72; // rax
  __int64 v73; // rdx
  unsigned __int64 v74; // rcx
  _QWORD *v75; // rsi
  bool v76; // zf
  __int64 v77; // rax
  _BYTE *v78; // rsi
  _QWORD *v79; // rbx
  _QWORD *v80; // r13
  __int64 v81; // rdx
  _QWORD *v82; // rax
  _QWORD *v83; // r13
  _QWORD *v84; // rbx
  const __m128i *v85; // rsi
  __int64 v86; // rdx
  __int64 v87; // rcx
  __m128i *v88; // r14
  __m128i *v89; // r13
  __int64 v90; // rbx
  unsigned __int64 v91; // rax
  __m128i *v92; // rbx
  __m128i *v93; // rdi
  _QWORD *v94; // rax
  _QWORD *v95; // r13
  _QWORD *v96; // rbx
  const __m128i *v97; // rsi
  __int64 v98; // rdx
  __int64 v99; // rcx
  __m128i *v100; // r14
  __m128i *v101; // r13
  __int64 v102; // rbx
  unsigned __int64 v103; // rax
  __m128i *v104; // rbx
  __m128i *v105; // rdi
  __int64 v106; // [rsp+8h] [rbp-198h]
  __int64 v108; // [rsp+40h] [rbp-160h] BYREF
  __int64 *v109; // [rsp+48h] [rbp-158h] BYREF
  _QWORD v110[2]; // [rsp+50h] [rbp-150h] BYREF
  unsigned __int64 *v111; // [rsp+60h] [rbp-140h] BYREF
  unsigned __int64 *v112; // [rsp+68h] [rbp-138h] BYREF
  unsigned __int64 v113; // [rsp+70h] [rbp-130h]
  unsigned __int64 **v114; // [rsp+78h] [rbp-128h]
  unsigned __int64 **v115; // [rsp+80h] [rbp-120h]
  __int64 v116; // [rsp+88h] [rbp-118h]
  void *src[2]; // [rsp+90h] [rbp-110h] BYREF
  __m128i v118; // [rsp+A0h] [rbp-100h] BYREF
  __m128i v119; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v120; // [rsp+C0h] [rbp-E0h]
  int v121; // [rsp+D0h] [rbp-D0h] BYREF
  int *v122; // [rsp+D8h] [rbp-C8h]
  int *v123; // [rsp+E0h] [rbp-C0h]
  int *v124; // [rsp+E8h] [rbp-B8h]
  __int64 v125; // [rsp+F0h] [rbp-B0h]
  void *v126; // [rsp+100h] [rbp-A0h] BYREF
  __m128i v127; // [rsp+108h] [rbp-98h]
  __m128i v128; // [rsp+118h] [rbp-88h]
  __m128i v129; // [rsp+128h] [rbp-78h]
  __int64 v130; // [rsp+138h] [rbp-68h]
  int v131; // [rsp+148h] [rbp-58h] BYREF
  int *v132; // [rsp+150h] [rbp-50h]
  int *v133; // [rsp+158h] [rbp-48h]
  int *v134; // [rsp+160h] [rbp-40h]
  __int64 v135; // [rsp+168h] [rbp-38h]

  v2 = a1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, unsigned __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "GlobalValueMap",
         0,
         0,
         &v111) )
  {
    v76 = (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) == 0;
    v77 = *(_QWORD *)a1;
    if ( v76 )
    {
      (*(void (__fastcall **)(__int64))(v77 + 104))(a1);
      v78 = (_BYTE *)a1;
      (*(void (__fastcall **)(void **, __int64))(*(_QWORD *)a1 + 136LL))(&v126, a1);
      v79 = v126;
      v80 = (_QWORD *)v127.m128i_i64[0];
      if ( v126 != (void *)v127.m128i_i64[0] )
      {
        do
        {
          v78 = (_BYTE *)*v79;
          v81 = v79[1];
          v79 += 2;
          sub_26305F0(a1, v78, v81, (_QWORD *)a2);
        }
        while ( v80 != v79 );
        v80 = v126;
      }
      if ( v80 )
      {
        v78 = (_BYTE *)(v127.m128i_i64[1] - (_QWORD)v80);
        j_j___libc_free_0((unsigned __int64)v80);
      }
      (*(void (__fastcall **)(__int64, _BYTE *))(*(_QWORD *)a1 + 112LL))(a1, v78);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v77 + 104))(a1);
      sub_262F360(a1, a2);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    }
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, src[0]);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v70 = *(_QWORD *)(a2 + 24);
    if ( v70 != a2 + 8 )
    {
      while ( 1 )
      {
        v71 = *(__int64 **)(v70 + 64);
        v72 = *(__int64 **)(v70 + 56);
        if ( v72 != v71 )
          break;
LABEL_142:
        v70 = sub_220EEE0(v70);
        if ( a2 + 8 == v70 )
          goto LABEL_3;
      }
      while ( 1 )
      {
        v73 = *v72;
        if ( *(_DWORD *)(*v72 + 8) )
          goto LABEL_138;
        v74 = *(_QWORD *)(v73 + 56) & 0xFFFFFFFFFFFFFFF8LL;
        v75 = *(_QWORD **)(v74 + 24);
        if ( v75 == *(_QWORD **)(v74 + 32) )
        {
          ++v72;
          *(_QWORD *)(v73 + 56) = 0;
          *(_QWORD *)(v73 + 64) = 0;
          if ( v71 == v72 )
            goto LABEL_142;
        }
        else
        {
          *(_QWORD *)(v73 + 64) = *v75;
LABEL_138:
          if ( v71 == ++v72 )
            goto LABEL_142;
        }
      }
    }
  }
LABEL_3:
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v2 + 16LL))(v2) )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, void **, void **))(*(_QWORD *)v2 + 120LL))(
           v2,
           "TypeIdMap",
           0,
           0,
           src,
           &v126) )
    {
      sub_262CA90(v2, (_QWORD *)(a2 + 208));
      (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 128LL))(v2, v126);
    }
  }
  else
  {
    v114 = &v112;
    v115 = &v112;
    v5 = *(_QWORD *)v2;
    LODWORD(v112) = 0;
    v113 = 0;
    v116 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, void **, void **))(v5 + 120))(
           v2,
           "TypeIdMap",
           0,
           0,
           src,
           &v126) )
    {
      sub_262CA90(v2, &v111);
      (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 128LL))(v2, v126);
    }
    if ( v114 != &v112 )
    {
      v106 = v2;
      v6 = (__int64)v114;
      do
      {
        v7.m128i_i64[0] = (__int64)sub_C94910(a2 + 168, *(_QWORD **)(v6 + 40), *(_QWORD *)(v6 + 48));
        *(__m128i *)src = v7;
        v8 = _mm_loadu_si128((const __m128i *)(v6 + 56));
        v118 = v8;
        v9 = _mm_loadu_si128((const __m128i *)(v6 + 72));
        v119 = v9;
        v10 = *(_QWORD *)(v6 + 88);
        v120 = v10;
        v11 = *(_QWORD *)(v6 + 112);
        if ( v11 )
        {
          v12 = *(_DWORD *)(v6 + 104);
          v122 = *(int **)(v6 + 112);
          v121 = v12;
          v123 = *(int **)(v6 + 120);
          v124 = *(int **)(v6 + 128);
          *(_QWORD *)(v11 + 8) = &v121;
          v125 = *(_QWORD *)(v6 + 136);
          *(_QWORD *)(v6 + 120) = v6 + 104;
          *(_QWORD *)(v6 + 128) = v6 + 104;
          v13 = *(void **)(v6 + 32);
          *(_QWORD *)(v6 + 112) = 0;
          *(_QWORD *)(v6 + 136) = 0;
          v14 = _mm_loadu_si128((const __m128i *)src);
          v126 = v13;
          v15 = _mm_loadu_si128(&v118);
          v16 = _mm_loadu_si128(&v119);
          v131 = 0;
          v132 = 0;
          v130 = v120;
          v133 = &v131;
          v134 = &v131;
          v135 = 0;
          v127 = v14;
          v128 = v15;
          v129 = v16;
          if ( v122 )
          {
            v17 = sub_261B2A0(v122, (__int64)&v131);
            v18 = (int *)v17;
            do
            {
              v19 = (int *)v17;
              v17 = *(_QWORD *)(v17 + 16);
            }
            while ( v17 );
            v133 = v19;
            v20 = v18;
            do
            {
              v21 = v20;
              v20 = (int *)*((_QWORD *)v20 + 3);
            }
            while ( v20 );
            v134 = v21;
            v132 = v18;
            v135 = v125;
          }
        }
        else
        {
          v121 = 0;
          v122 = 0;
          v123 = &v121;
          v124 = &v121;
          v125 = 0;
          v22 = *(void **)(v6 + 32);
          v127 = v7;
          v126 = v22;
          v130 = v10;
          v131 = 0;
          v132 = 0;
          v133 = &v131;
          v134 = &v131;
          v135 = 0;
          v128 = v8;
          v129 = v9;
        }
        sub_9CA630((_QWORD *)(a2 + 208), (__int64 *)&v126);
        sub_261C430(v132);
        sub_261C430(v122);
        v6 = sub_220EEE0(v6);
      }
      while ( (unsigned __int64 **)v6 != &v112 );
      v2 = v106;
    }
    sub_261C4E0(v113);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, void **, void **))(*(_QWORD *)v2 + 120LL))(
         v2,
         "WithGlobalValueDeadStripping",
         0,
         0,
         src,
         &v126) )
  {
    sub_261B670(v2, (_BYTE *)(a2 + 336));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 128LL))(v2, v126);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v2 + 16LL))(v2) )
  {
    src[0] = 0;
    src[1] = 0;
    v3 = *(_DWORD *)(a2 + 368);
    v118.m128i_i64[0] = 0;
    if ( v3 )
    {
      v82 = *(_QWORD **)(a2 + 360);
      v83 = &v82[7 * *(unsigned int *)(a2 + 376)];
      if ( v82 != v83 )
      {
        while ( 1 )
        {
          v84 = v82;
          if ( *v82 <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          v82 += 7;
          if ( v83 == v82 )
            goto LABEL_10;
        }
        if ( v83 != v82 )
        {
          v85 = 0;
          do
          {
            v86 = v84[4];
            v87 = (__int64)(v84 + 2);
            v84 += 7;
            sub_261A2E0((unsigned __int64 *)src, v85, v86, v87);
            if ( v84 == v83 )
            {
LABEL_162:
              v88 = (__m128i *)src[1];
              goto LABEL_163;
            }
            while ( *v84 > 0xFFFFFFFFFFFFFFFDLL )
            {
              v84 += 7;
              if ( v83 == v84 )
                goto LABEL_162;
            }
            v85 = (const __m128i *)src[1];
          }
          while ( v83 != v84 );
          v88 = (__m128i *)src[1];
LABEL_163:
          v89 = (__m128i *)src[0];
          if ( v88 != src[0] )
          {
            v90 = (char *)v88 - (char *)src[0];
            _BitScanReverse64(&v91, ((char *)v88 - (char *)src[0]) >> 4);
            sub_26339F0((__m128i *)src[0], v88, 2LL * (int)(63 - (v91 ^ 0x3F)));
            if ( v90 <= 256 )
            {
              sub_A3B670(v89, v88);
              if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v2 + 56LL))(v2) )
                goto LABEL_12;
              goto LABEL_11;
            }
            v92 = v89 + 16;
            sub_A3B670(v89, v89 + 16);
            if ( &v89[16] != v88 )
            {
              do
              {
                v93 = v92++;
                sub_A3B600(v93);
              }
              while ( v88 != v92 );
              if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v2 + 56LL))(v2) )
                goto LABEL_12;
              goto LABEL_11;
            }
          }
        }
      }
    }
LABEL_10:
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v2 + 56LL))(v2) )
      goto LABEL_12;
LABEL_11:
    if ( src[1] == src[0] )
    {
LABEL_14:
      v126 = 0;
      v127 = 0u;
      if ( *(_DWORD *)(a2 + 400) )
      {
        v94 = *(_QWORD **)(a2 + 392);
        v95 = &v94[7 * *(unsigned int *)(a2 + 408)];
        if ( v94 != v95 )
        {
          while ( 1 )
          {
            v96 = v94;
            if ( *v94 <= 0xFFFFFFFFFFFFFFFDLL )
              break;
            v94 += 7;
            if ( v95 == v94 )
              goto LABEL_15;
          }
          if ( v95 != v94 )
          {
            v97 = 0;
            do
            {
              v98 = v96[4];
              v99 = (__int64)(v96 + 2);
              v96 += 7;
              sub_261A2E0((unsigned __int64 *)&v126, v97, v98, v99);
              if ( v96 == v95 )
              {
LABEL_180:
                v100 = (__m128i *)v127.m128i_i64[0];
                goto LABEL_181;
              }
              while ( *v96 > 0xFFFFFFFFFFFFFFFDLL )
              {
                v96 += 7;
                if ( v95 == v96 )
                  goto LABEL_180;
              }
              v97 = (const __m128i *)v127.m128i_i64[0];
            }
            while ( v95 != v96 );
            v100 = (__m128i *)v127.m128i_i64[0];
LABEL_181:
            v101 = (__m128i *)v126;
            if ( v126 != v100 )
            {
              v102 = (char *)v100 - (_BYTE *)v126;
              _BitScanReverse64(&v103, ((char *)v100 - (_BYTE *)v126) >> 4);
              sub_26339F0((__m128i *)v126, v100, 2LL * (int)(63 - (v103 ^ 0x3F)));
              if ( v102 <= 256 )
              {
                sub_A3B670(v101, v100);
              }
              else
              {
                v104 = v101 + 16;
                sub_A3B670(v101, v101 + 16);
                if ( &v101[16] != v100 )
                {
                  do
                  {
                    v105 = v104++;
                    sub_A3B600(v105);
                  }
                  while ( v100 != v104 );
                }
              }
            }
          }
        }
      }
LABEL_15:
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v2 + 56LL))(v2)
        || (v4 = v126, (void *)v127.m128i_i64[0] != v126) )
      {
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, _QWORD *, unsigned __int64 **))(*(_QWORD *)v2 + 120LL))(
               v2,
               "CfiFunctionDecls",
               0,
               0,
               v110,
               &v111) )
        {
          sub_26314E0(v2, (__int64)&v126);
          (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)v2 + 128LL))(v2, v111);
        }
        v4 = v126;
      }
      if ( v4 )
        j_j___libc_free_0((unsigned __int64)v4);
      if ( src[0] )
        j_j___libc_free_0((unsigned __int64)src[0]);
      return;
    }
LABEL_12:
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, unsigned __int64 **, void **))(*(_QWORD *)v2 + 120LL))(
           v2,
           "CfiFunctionDefs",
           0,
           0,
           &v111,
           &v126) )
    {
      sub_26314E0(v2, (__int64)src);
      (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 128LL))(v2, v126);
    }
    goto LABEL_14;
  }
  v23 = *(_QWORD *)v2;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(v23 + 56))(v2) && v112 == v111 )
  {
    v126 = 0;
    v127 = 0u;
    v128.m128i_i64[0] = 0;
  }
  else
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, void **, void **))(*(_QWORD *)v2 + 120LL))(
           v2,
           "CfiFunctionDefs",
           0,
           0,
           src,
           &v126) )
    {
      sub_2631600(v2, (__int64)&v111);
      (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 128LL))(v2, v126);
    }
    v24 = v111;
    v25 = v112;
    v126 = 0;
    v127 = 0u;
    v128.m128i_i64[0] = 0;
    if ( v112 != v111 )
    {
      while ( 1 )
      {
        v32 = (_BYTE *)*v24;
        v33 = (char *)v24[1];
        src[0] = (void *)*v24;
        src[1] = v33;
        if ( v33 && *v32 == 1 )
        {
          --v33;
          ++v32;
        }
        v34 = sub_B2F650((__int64)v32, (__int64)v33);
        v109 = (__int64 *)v34;
        if ( !v128.m128i_i32[0] )
          break;
        v26 = 1;
        v27 = (v128.m128i_i32[0] - 1) & (((0xBF58476D1CE4E5B9LL * v34) >> 31) ^ (484763065 * v34));
        v28 = 0;
        v29 = (__int64 *)(v127.m128i_i64[0] + 56LL * v27);
        v30 = *v29;
        if ( v34 != *v29 )
        {
          while ( v30 != -1 )
          {
            if ( v30 == -2 && !v28 )
              v28 = v29;
            v27 = (v128.m128i_i32[0] - 1) & (v26 + v27);
            v29 = (__int64 *)(v127.m128i_i64[0] + 56LL * v27);
            v30 = *v29;
            if ( v34 == *v29 )
              goto LABEL_47;
            ++v26;
          }
          if ( !v28 )
            v28 = v29;
          v126 = (char *)v126 + 1;
          v35 = v127.m128i_i32[2] + 1;
          v110[0] = v28;
          if ( 4 * (v127.m128i_i32[2] + 1) < (unsigned int)(3 * v128.m128i_i32[0]) )
          {
            if ( v128.m128i_i32[0] - v127.m128i_i32[3] - v35 <= (unsigned __int32)v128.m128i_i32[0] >> 3 )
            {
              sub_9EB160((__int64)&v126, v128.m128i_i32[0]);
              sub_262B770((__int64)&v126, (__int64 *)&v109, v110);
              v34 = (__int64)v109;
              v28 = (__int64 *)v110[0];
              v35 = v127.m128i_i32[2] + 1;
            }
            goto LABEL_66;
          }
LABEL_54:
          sub_9EB160((__int64)&v126, 2 * v128.m128i_i32[0]);
          sub_262B770((__int64)&v126, (__int64 *)&v109, v110);
          v34 = (__int64)v109;
          v28 = (__int64 *)v110[0];
          v35 = v127.m128i_i32[2] + 1;
LABEL_66:
          v127.m128i_i32[2] = v35;
          if ( *v28 != -1 )
            --v127.m128i_i32[3];
          *v28 = v34;
          v36 = v28 + 2;
          v31 = v28 + 1;
          *((_DWORD *)v31 + 2) = 0;
          v31[2] = 0;
          v31[3] = v36;
          v31[4] = v36;
          v31[5] = 0;
          goto LABEL_48;
        }
LABEL_47:
        v31 = v29 + 1;
LABEL_48:
        v24 += 4;
        sub_9D35B0(v31, (__int64)src);
        if ( v25 == v24 )
          goto LABEL_70;
      }
      v126 = (char *)v126 + 1;
      v110[0] = 0;
      goto LABEL_54;
    }
  }
LABEL_70:
  v37 = *(unsigned int *)(a2 + 376);
  if ( (_DWORD)v37 )
  {
    v38 = *(_QWORD **)(a2 + 360);
    v39 = &v38[7 * v37];
    do
    {
      while ( 1 )
      {
        if ( *v38 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v40 = v38[3];
          if ( v40 )
            break;
        }
        v38 += 7;
        if ( v39 == v38 )
          goto LABEL_79;
      }
      do
      {
        v41 = v40;
        sub_261C7C0(*(_QWORD **)(v40 + 24));
        v42 = *(_QWORD *)(v40 + 32);
        v40 = *(_QWORD *)(v40 + 16);
        if ( v42 != v41 + 48 )
          j_j___libc_free_0(v42);
        j_j___libc_free_0(v41);
      }
      while ( v40 );
      v38 += 7;
    }
    while ( v39 != v38 );
LABEL_79:
    v37 = *(unsigned int *)(a2 + 376);
  }
  sub_C7D6A0(*(_QWORD *)(a2 + 360), 56 * v37, 8);
  v43 = v127.m128i_i64[0];
  ++*(_QWORD *)(a2 + 352);
  *(_QWORD *)(a2 + 360) = v43;
  v126 = (char *)v126 + 1;
  *(_QWORD *)(a2 + 368) = v127.m128i_i64[1];
  v127.m128i_i64[0] = 0;
  *(_DWORD *)(a2 + 376) = v128.m128i_i32[0];
  v127.m128i_i64[1] = 0;
  v128.m128i_i32[0] = 0;
  sub_C7D6A0(0, 0, 8);
  v44 = *(_QWORD *)v2;
  src[0] = 0;
  src[1] = 0;
  v118.m128i_i64[0] = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(v44 + 56))(v2) && src[1] == src[0] )
  {
    v126 = 0;
    v127 = 0u;
    v128.m128i_i64[0] = 0;
  }
  else
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, _QWORD *, void **))(*(_QWORD *)v2 + 120LL))(
           v2,
           "CfiFunctionDecls",
           0,
           0,
           v110,
           &v126) )
    {
      sub_2631600(v2, (__int64)src);
      (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 128LL))(v2, v126);
    }
    v45 = src[0];
    v46 = src[1];
    v126 = 0;
    v127 = 0u;
    v128.m128i_i64[0] = 0;
    if ( src[1] != src[0] )
    {
      while ( 1 )
      {
        v53 = (_BYTE *)*v45;
        v54 = v45[1];
        v110[0] = *v45;
        v110[1] = v54;
        if ( v54 && *v53 == 1 )
        {
          --v54;
          ++v53;
        }
        v55 = sub_B2F650((__int64)v53, v54);
        v56 = v128.m128i_i32[0];
        v108 = v55;
        if ( !v128.m128i_i32[0] )
          break;
        v47 = 1;
        v48 = (v128.m128i_i32[0] - 1) & (((0xBF58476D1CE4E5B9LL * v55) >> 31) ^ (484763065 * v55));
        v49 = 0;
        v50 = (__int64 *)(v127.m128i_i64[0] + 56LL * v48);
        v51 = *v50;
        if ( v55 != *v50 )
        {
          while ( v51 != -1 )
          {
            if ( v51 == -2 && !v49 )
              v49 = v50;
            v48 = (v128.m128i_i32[0] - 1) & (v47 + v48);
            v50 = (__int64 *)(v127.m128i_i64[0] + 56LL * v48);
            v51 = *v50;
            if ( v55 == *v50 )
              goto LABEL_87;
            ++v47;
          }
          if ( !v49 )
            v49 = v50;
          v126 = (char *)v126 + 1;
          v57 = v127.m128i_i32[2] + 1;
          v109 = v49;
          if ( 4 * (v127.m128i_i32[2] + 1) < (unsigned int)(3 * v128.m128i_i32[0]) )
          {
            if ( v128.m128i_i32[0] - v127.m128i_i32[3] - v57 > (unsigned __int32)v128.m128i_i32[0] >> 3 )
              goto LABEL_106;
            goto LABEL_95;
          }
LABEL_94:
          v56 = 2 * v128.m128i_i32[0];
LABEL_95:
          sub_9EB160((__int64)&v126, v56);
          sub_262B770((__int64)&v126, &v108, &v109);
          v55 = v108;
          v49 = v109;
          v57 = v127.m128i_i32[2] + 1;
LABEL_106:
          v127.m128i_i32[2] = v57;
          if ( *v49 != -1 )
            --v127.m128i_i32[3];
          *v49 = v55;
          v58 = v49 + 2;
          v52 = v49 + 1;
          *((_DWORD *)v52 + 2) = 0;
          v52[2] = 0;
          v52[3] = v58;
          v52[4] = v58;
          v52[5] = 0;
          goto LABEL_88;
        }
LABEL_87:
        v52 = v50 + 1;
LABEL_88:
        v45 += 4;
        sub_9D35B0(v52, (__int64)v110);
        if ( v46 == v45 )
          goto LABEL_110;
      }
      v126 = (char *)v126 + 1;
      v109 = 0;
      goto LABEL_94;
    }
  }
LABEL_110:
  v59 = *(unsigned int *)(a2 + 408);
  if ( (_DWORD)v59 )
  {
    v60 = *(_QWORD **)(a2 + 392);
    v61 = &v60[7 * v59];
    do
    {
      while ( 1 )
      {
        if ( *v60 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v62 = v60[3];
          if ( v62 )
            break;
        }
        v60 += 7;
        if ( v61 == v60 )
          goto LABEL_119;
      }
      do
      {
        v63 = v62;
        sub_261C7C0(*(_QWORD **)(v62 + 24));
        v64 = *(_QWORD *)(v62 + 32);
        v62 = *(_QWORD *)(v62 + 16);
        if ( v64 != v63 + 48 )
          j_j___libc_free_0(v64);
        j_j___libc_free_0(v63);
      }
      while ( v62 );
      v60 += 7;
    }
    while ( v61 != v60 );
LABEL_119:
    v59 = *(unsigned int *)(a2 + 408);
  }
  sub_C7D6A0(*(_QWORD *)(a2 + 392), 56 * v59, 8);
  v65 = v127.m128i_i64[0];
  ++*(_QWORD *)(a2 + 384);
  *(_QWORD *)(a2 + 392) = v65;
  v126 = (char *)v126 + 1;
  *(_QWORD *)(a2 + 400) = v127.m128i_i64[1];
  v127.m128i_i64[0] = 0;
  *(_DWORD *)(a2 + 408) = v128.m128i_i32[0];
  v127.m128i_i64[1] = 0;
  v128.m128i_i32[0] = 0;
  sub_C7D6A0(0, 0, 8);
  v66 = (unsigned __int64 *)src[1];
  v67 = (unsigned __int64 *)src[0];
  if ( src[1] != src[0] )
  {
    do
    {
      if ( (unsigned __int64 *)*v67 != v67 + 2 )
        j_j___libc_free_0(*v67);
      v67 += 4;
    }
    while ( v66 != v67 );
    v67 = (unsigned __int64 *)src[0];
  }
  if ( v67 )
    j_j___libc_free_0((unsigned __int64)v67);
  v68 = v112;
  v69 = v111;
  if ( v112 != v111 )
  {
    do
    {
      if ( (unsigned __int64 *)*v69 != v69 + 2 )
        j_j___libc_free_0(*v69);
      v69 += 4;
    }
    while ( v68 != v69 );
    v69 = v111;
  }
  if ( v69 )
    j_j___libc_free_0((unsigned __int64)v69);
}
