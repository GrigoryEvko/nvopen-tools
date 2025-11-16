// Function: sub_C152E0
// Address: 0xc152e0
//
__int64 __fastcall sub_C152E0(__int64 a1)
{
  __int64 v1; // rsi
  _QWORD *v2; // r12
  _QWORD *v3; // r14
  __m128i *v4; // rdi
  __int64 (__fastcall *v5)(__int64); // rax
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rdx
  unsigned __int64 v10; // rdx
  size_t v11; // r14
  unsigned int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // rcx
  _QWORD *v15; // r14
  _QWORD *v16; // r15
  __m128i *v17; // rdi
  __int64 (__fastcall *v18)(__int64); // rax
  __int64 v19; // rdi
  _QWORD *v20; // r15
  unsigned int v21; // eax
  unsigned int v22; // r13d
  __int64 v23; // rdi
  unsigned __int64 *v24; // rsi
  unsigned __int64 v25; // rdx
  __int64 v26; // r12
  char v27; // al
  const __m128i *v28; // rbx
  char *v29; // rax
  __int64 v30; // rdx
  unsigned __int64 v31; // rcx
  __int64 v32; // rsi
  unsigned __int64 v33; // rdi
  __int64 (__fastcall *v34)(__int64); // rcx
  unsigned __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // r12
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 (__fastcall *v40)(__int64, __int64, int); // rax
  __int64 v41; // rax
  unsigned int v42; // r9d
  _QWORD *v43; // r11
  _QWORD *v44; // rcx
  unsigned int v45; // eax
  __int64 *v46; // rax
  __int64 *v47; // rax
  __int64 v48; // rsi
  __int64 v49; // r8
  __int64 v50; // r12
  __int64 v51; // rbx
  _QWORD *v52; // rdi
  __int64 v54; // r14
  __int64 *v55; // r12
  unsigned int v56; // eax
  int v57; // eax
  __int64 v58; // rax
  unsigned __int64 v59; // [rsp+0h] [rbp-2B0h]
  unsigned __int64 v60; // [rsp+8h] [rbp-2A8h]
  unsigned __int64 v61; // [rsp+10h] [rbp-2A0h]
  unsigned __int64 v62; // [rsp+20h] [rbp-290h]
  unsigned __int64 v63; // [rsp+30h] [rbp-280h]
  __int64 *v64; // [rsp+30h] [rbp-280h]
  char *v66; // [rsp+40h] [rbp-270h]
  unsigned __int64 v67; // [rsp+48h] [rbp-268h]
  __int64 *v68; // [rsp+48h] [rbp-268h]
  _QWORD *v69; // [rsp+50h] [rbp-260h]
  __int64 v70; // [rsp+50h] [rbp-260h]
  _QWORD *v71; // [rsp+58h] [rbp-258h]
  unsigned __int64 v72; // [rsp+60h] [rbp-250h]
  const __m128i *v73; // [rsp+60h] [rbp-250h]
  __int64 v74; // [rsp+68h] [rbp-248h]
  unsigned int v75; // [rsp+68h] [rbp-248h]
  void *src; // [rsp+70h] [rbp-240h]
  bool srca; // [rsp+70h] [rbp-240h]
  unsigned __int64 v78; // [rsp+78h] [rbp-238h]
  __int64 v79; // [rsp+80h] [rbp-230h] BYREF
  __int64 v80; // [rsp+88h] [rbp-228h]
  __int64 v81; // [rsp+90h] [rbp-220h]
  __int64 v82; // [rsp+A0h] [rbp-210h] BYREF
  __int64 v83; // [rsp+A8h] [rbp-208h]
  __int64 v84; // [rsp+B0h] [rbp-200h]
  unsigned int v85; // [rsp+B8h] [rbp-1F8h]
  __m128i v86; // [rsp+C0h] [rbp-1F0h] BYREF
  __m128i v87; // [rsp+D0h] [rbp-1E0h]
  __m128i v88; // [rsp+E0h] [rbp-1D0h]
  __m128i v89; // [rsp+F0h] [rbp-1C0h]
  _QWORD v90[2]; // [rsp+100h] [rbp-1B0h] BYREF
  __int64 (__fastcall *v91)(__int64); // [rsp+110h] [rbp-1A0h]
  __int64 v92; // [rsp+118h] [rbp-198h]
  __int64 (__fastcall *v93)(__int64); // [rsp+120h] [rbp-190h]
  __int64 v94; // [rsp+128h] [rbp-188h]
  __int64 (__fastcall *v95)(__int64 *); // [rsp+130h] [rbp-180h]
  __int64 v96; // [rsp+138h] [rbp-178h]
  _QWORD v97[2]; // [rsp+140h] [rbp-170h] BYREF
  __int64 (__fastcall *v98)(__int64); // [rsp+150h] [rbp-160h]
  unsigned __int64 v99; // [rsp+158h] [rbp-158h]
  __int64 (__fastcall *v100)(__int64); // [rsp+160h] [rbp-150h]
  __int64 v101; // [rsp+168h] [rbp-148h]
  __int64 (__fastcall *v102)(_QWORD *); // [rsp+170h] [rbp-140h]
  __int64 v103; // [rsp+178h] [rbp-138h]
  _BYTE *v104; // [rsp+180h] [rbp-130h] BYREF
  size_t n; // [rsp+188h] [rbp-128h]
  unsigned __int64 v106; // [rsp+190h] [rbp-120h]
  _BYTE v107[72]; // [rsp+198h] [rbp-118h] BYREF
  __m128i v108; // [rsp+1E0h] [rbp-D0h] BYREF
  __m128i v109; // [rsp+1F0h] [rbp-C0h] BYREF
  __m128i v110; // [rsp+200h] [rbp-B0h] BYREF
  __m128i v111; // [rsp+210h] [rbp-A0h] BYREF
  unsigned __int64 v112; // [rsp+220h] [rbp-90h]
  unsigned __int64 v113; // [rsp+228h] [rbp-88h]
  unsigned __int64 v114; // [rsp+230h] [rbp-80h]
  unsigned __int64 v115; // [rsp+238h] [rbp-78h]
  unsigned __int64 v116; // [rsp+240h] [rbp-70h]
  unsigned __int64 v117; // [rsp+248h] [rbp-68h]
  unsigned __int64 v118; // [rsp+250h] [rbp-60h]
  unsigned __int64 v119; // [rsp+258h] [rbp-58h]

  v81 = 0x1000000000LL;
  v1 = *(_QWORD *)(a1 + 296);
  v104 = v107;
  v79 = 0;
  v80 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  n = 0;
  v106 = 64;
  sub_BA96F0(&v108, (_QWORD *)v1);
  v60 = v112;
  v86 = _mm_loadu_si128(&v108);
  v62 = v113;
  v87 = _mm_loadu_si128(&v109);
  v67 = v114;
  v88 = _mm_loadu_si128(&v110);
  v78 = v115;
  v89 = _mm_loadu_si128(&v111);
  v59 = v116;
  v61 = v117;
  v63 = v118;
  v72 = v119;
  while ( *(_OWORD *)&v87 != __PAIR128__(v78, v67)
       || *(_OWORD *)&v86 != __PAIR128__(v62, v60)
       || *(_OWORD *)&v89 != __PAIR128__(v72, v63)
       || *(_OWORD *)&v88 != __PAIR128__(v61, v59) )
  {
    v2 = v90;
    v92 = 0;
    v3 = v90;
    v4 = &v86;
    v91 = sub_C141B0;
    v94 = 0;
    v93 = sub_C141D0;
    v96 = 0;
    v95 = sub_C141F0;
    v5 = sub_C14190;
    if ( ((unsigned __int8)sub_C14190 & 1) == 0 )
      goto LABEL_5;
    while ( 1 )
    {
      v5 = *(__int64 (__fastcall **)(__int64))((char *)v5 + v4->m128i_i64[0] - 1);
LABEL_5:
      v6 = v5((__int64)v4);
      if ( v6 )
        break;
      while ( 1 )
      {
        v2 += 2;
        if ( v97 == v2 )
LABEL_104:
          BUG();
        v7 = v3[3];
        v5 = (__int64 (__fastcall *)(__int64))v3[2];
        v3 = v2;
        v4 = (__m128i *)((char *)&v86 + v7);
        if ( ((unsigned __int8)v5 & 1) != 0 )
          break;
        v6 = v5((__int64)v4);
        if ( v6 )
          goto LABEL_9;
      }
    }
LABEL_9:
    v8 = v6;
    if ( (*(_BYTE *)(v6 + 7) & 0x10) == 0 )
      goto LABEL_15;
    n = 0;
    sub_BD5D20(v6);
    v10 = v9 + 1;
    if ( v10 > v106 )
      sub_C8D290(&v104, v107, v10, 1);
    sub_E410F0(&v82, &v104, v8, 0);
    v11 = n;
    src = v104;
    v12 = sub_C92610(v104, n);
    v1 = (__int64)src;
    v13 = sub_C92740(&v79, src, v11, v12);
    v14 = *(_QWORD *)(v79 + 8LL * v13);
    if ( !v14 )
      goto LABEL_60;
    if ( v14 == -8 )
    {
      LODWORD(v81) = v81 - 1;
LABEL_60:
      v71 = (_QWORD *)(v79 + 8LL * v13);
      v75 = v13;
      v41 = sub_C7D670(v11 + 17, 8);
      v42 = v75;
      v43 = v71;
      v44 = (_QWORD *)v41;
      if ( v11 )
      {
        v69 = (_QWORD *)v41;
        memcpy((void *)(v41 + 16), src, v11);
        v42 = v75;
        v43 = v71;
        v44 = v69;
      }
      *((_BYTE *)v44 + v11 + 16) = 0;
      v1 = v42;
      *v44 = v11;
      v44[1] = 0;
      *v43 = v44;
      ++HIDWORD(v80);
      v45 = sub_C929D0(&v79, v42);
      v46 = (__int64 *)(v79 + 8LL * v45);
      v14 = *v46;
      if ( !*v46 || v14 == -8 )
      {
        v47 = v46 + 1;
        do
        {
          do
            v14 = *v47++;
          while ( v14 == -8 );
        }
        while ( !v14 );
      }
    }
    *(_QWORD *)(v14 + 8) = v8;
LABEL_15:
    v15 = v97;
    v99 = 0;
    v101 = 0;
    v16 = v97;
    v17 = &v86;
    v98 = sub_C14100;
    v103 = 0;
    v100 = sub_C14130;
    v102 = sub_C14160;
    v18 = sub_C140D0;
    if ( ((unsigned __int8)sub_C140D0 & 1) == 0 )
      goto LABEL_17;
LABEL_16:
    v18 = *(__int64 (__fastcall **)(__int64))((char *)v18 + v17->m128i_i64[0] - 1);
LABEL_17:
    while ( !(unsigned __int8)v18((__int64)v17) )
    {
      v15 += 2;
      if ( &v104 == v15 )
        goto LABEL_104;
      v19 = v16[3];
      v18 = (__int64 (__fastcall *)(__int64))v16[2];
      v16 = v15;
      v17 = (__m128i *)((char *)&v86 + v19);
      if ( ((unsigned __int8)v18 & 1) != 0 )
        goto LABEL_16;
    }
  }
  v20 = (_QWORD *)a1;
  v68 = *(__int64 **)(a1 + 360);
  v64 = &v68[4 * *(unsigned int *)(a1 + 368)];
  while ( v64 != v68 )
  {
    v1 = *v68;
    v74 = *v68;
    v21 = sub_C14AD0((__int64)v20, *v68);
    if ( v21 == 4 )
    {
      srca = 1;
      v22 = 24;
      goto LABEL_38;
    }
    if ( v21 > 4 )
    {
      v22 = 0;
      srca = 0;
      if ( v21 == 6 )
        v22 = 24;
    }
    else
    {
      v22 = v21 & 0xFFFFFFFD;
      if ( (v21 & 0xFFFFFFFD) != 0 )
      {
        if ( v21 - 2 <= 2 )
        {
          srca = 1;
          v22 = 9;
          goto LABEL_38;
        }
        srca = 0;
        v22 = 9;
      }
      else
      {
        srca = v21 - 2 <= 2;
      }
    }
    v23 = v20[37];
    if ( (*(_BYTE *)(v74 + 8) & 1) != 0 )
    {
      v24 = *(unsigned __int64 **)(v74 - 8);
      v25 = *v24;
      v1 = (__int64)(v24 + 3);
      v26 = sub_BA8B30(v23, v1, v25);
      if ( v26 )
        goto LABEL_31;
    }
    else
    {
      v1 = 0;
      v26 = sub_BA8B30(v23, 0, 0);
      if ( v26 )
        goto LABEL_31;
    }
    v54 = 0;
    if ( (*(_BYTE *)(v74 + 8) & 1) != 0 )
    {
      v55 = *(__int64 **)(v74 - 8);
      v54 = *v55;
      v26 = (__int64)(v55 + 3);
    }
    v56 = sub_C92610(v26, v54);
    v1 = v26;
    v57 = sub_C92860(&v79, v26, v54, v56);
    if ( v57 == -1 )
      goto LABEL_38;
    v58 = v79 + 8LL * v57;
    if ( v58 == v79 + 8LL * (unsigned int)v80 )
      goto LABEL_38;
    v26 = *(_QWORD *)(*(_QWORD *)v58 + 8LL);
    if ( !v26 )
      goto LABEL_38;
LABEL_31:
    if ( !v22 )
    {
      v27 = *(_BYTE *)(v26 + 32) & 0xF;
      if ( !v27 )
      {
        v22 = 9;
        if ( srca )
          goto LABEL_38;
        goto LABEL_37;
      }
      if ( ((v27 + 9) & 0xFu) > 1 )
      {
        if ( ((v27 + 14) & 0xFu) <= 3 )
        {
          v22 = 24;
        }
        else if ( ((v27 + 7) & 0xFu) <= 1 )
        {
          v22 = 24;
        }
      }
      else
      {
        v22 = 17;
      }
    }
    if ( srca || (*(_BYTE *)(v26 + 32) & 0xF) == 1 )
      goto LABEL_38;
LABEL_37:
    srca = !sub_B2FC80(v26);
LABEL_38:
    v28 = (const __m128i *)v68[1];
    v73 = (const __m128i *)v68[2];
    if ( v73 != v28 )
    {
      v29 = "@";
      if ( srca )
        v29 = "@@";
      v66 = v29;
      do
      {
        v86 = _mm_loadu_si128(v28);
        v35 = sub_C931B0(&v86, &unk_3F645A0, 3, 0);
        if ( v35 == -1 )
        {
          v32 = v86.m128i_i64[0];
          v109.m128i_i64[0] = 128;
          v30 = v86.m128i_i64[1];
          v108 = (__m128i)(unsigned __int64)&v109.m128i_u64[1];
        }
        else
        {
          v30 = v86.m128i_i64[1];
          v31 = v35 + 3;
          v32 = v86.m128i_i64[0];
          if ( v35 + 3 > v86.m128i_i64[1] )
          {
            v109.m128i_i64[0] = 128;
            v108 = (__m128i)(unsigned __int64)&v109.m128i_u64[1];
          }
          else
          {
            v109.m128i_i64[0] = 128;
            v108 = (__m128i)(unsigned __int64)&v109.m128i_u64[1];
            v33 = v86.m128i_i64[1] - v31;
            if ( v86.m128i_i64[1] != v31 )
            {
              v34 = (__int64 (__fastcall *)(__int64))(v86.m128i_i64[0] + v31);
              if ( *(_BYTE *)v34 != 64 )
              {
                v90[0] = v86.m128i_i64[0];
                if ( v35 > v86.m128i_i64[1] )
                  v35 = v86.m128i_u64[1];
                v99 = v33;
                LOWORD(v93) = 773;
                v90[1] = v35;
                v98 = v34;
                v91 = (__int64 (__fastcall *)(__int64))v66;
                v97[0] = v90;
                LOWORD(v100) = 1282;
                sub_CA0EC0(v97, &v108);
                v30 = v108.m128i_i64[1];
                v32 = v108.m128i_i64[0];
                v86 = v108;
              }
            }
          }
        }
        v36 = v20[1];
        v97[0] = v32;
        LOWORD(v100) = 261;
        v97[1] = v30;
        v37 = sub_E6C460(v36, v97);
        v38 = sub_E808D0(v74, 0, v20[1], 0);
        v39 = v38;
        if ( srca )
        {
          v70 = v38;
          sub_C14460((__int64)v20, v37);
          v39 = v70;
        }
        v1 = v37;
        sub_E9A490(v20, v37, v39);
        if ( v22 )
        {
          v40 = *(__int64 (__fastcall **)(__int64, __int64, int))(*v20 + 296LL);
          if ( v40 == sub_C14950 )
          {
            if ( v22 == 9 || v22 == 24 )
            {
              v1 = v37;
              sub_C14630((__int64)v20, v37, v22);
            }
          }
          else
          {
            v1 = v37;
            v40((__int64)v20, v37, v22);
          }
        }
        if ( (unsigned __int64 *)v108.m128i_i64[0] != &v109.m128i_u64[1] )
          _libc_free(v108.m128i_i64[0], v1);
        ++v28;
      }
      while ( v73 != v28 );
    }
    v68 += 4;
  }
  if ( v104 != v107 )
    _libc_free(v104, v1);
  v48 = 16LL * v85;
  sub_C7D6A0(v83, v48, 8);
  if ( HIDWORD(v80) )
  {
    v49 = v79;
    if ( (_DWORD)v80 )
    {
      v50 = 8LL * (unsigned int)v80;
      v51 = 0;
      do
      {
        v52 = *(_QWORD **)(v49 + v51);
        if ( v52 != (_QWORD *)-8LL && v52 )
        {
          v48 = *v52 + 17LL;
          sub_C7D6A0(v52, v48, 8);
          v49 = v79;
        }
        v51 += 8;
      }
      while ( v50 != v51 );
    }
  }
  else
  {
    v49 = v79;
  }
  return _libc_free(v49, v48);
}
