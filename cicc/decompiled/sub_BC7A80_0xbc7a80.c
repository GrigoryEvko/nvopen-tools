// Function: sub_BC7A80
// Address: 0xbc7a80
//
__m128i *__fastcall sub_BC7A80(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const char *a7,
        __int64 a8,
        const char *a9,
        __int64 a10,
        const char *a11,
        __int64 a12)
{
  __int64 v15; // r13
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __m128i *v18; // r15
  __m128i *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  __m128i si128; // xmm0
  __int64 v23; // rax
  __int64 v24; // rdx
  __m128i *v26; // rax
  __int64 v27; // rdx
  __m128i v28; // xmm0
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rax
  _BYTE *v35; // rsi
  __int64 v36; // rdx
  _BYTE *v37; // rdi
  _BYTE *v38; // rax
  __int64 v39; // rdi
  __m128i *v40; // rax
  size_t v41; // rax
  __m128i *v42; // r12
  __int64 *v43; // r13
  __m128i *v44; // r15
  int v45; // eax
  __m128i *v46; // rax
  __int64 v47; // rdx
  __m128i v48; // xmm0
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rdx
  __m128i v53; // xmm0
  __int64 v54; // rax
  __int64 v55; // rdx
  size_t v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  __m128i v59; // xmm0
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // [rsp+20h] [rbp-3C0h]
  _QWORD *v63; // [rsp+50h] [rbp-390h] BYREF
  char v64; // [rsp+60h] [rbp-380h]
  void *dest; // [rsp+70h] [rbp-370h]
  size_t v66; // [rsp+78h] [rbp-368h]
  __m128i v67; // [rsp+80h] [rbp-360h] BYREF
  __int64 v68; // [rsp+90h] [rbp-350h] BYREF
  size_t n; // [rsp+98h] [rbp-348h]
  _QWORD v70[2]; // [rsp+A0h] [rbp-340h] BYREF
  __int16 v71; // [rsp+B0h] [rbp-330h]
  _OWORD *v72; // [rsp+C0h] [rbp-320h]
  __int64 v73; // [rsp+C8h] [rbp-318h]
  _OWORD v74[3]; // [rsp+D0h] [rbp-310h] BYREF
  __int128 v75; // [rsp+100h] [rbp-2E0h] BYREF
  __int64 v76; // [rsp+110h] [rbp-2D0h]
  __int64 v77; // [rsp+118h] [rbp-2C8h]
  __int64 v78; // [rsp+120h] [rbp-2C0h]
  char v79; // [rsp+128h] [rbp-2B8h]
  __int128 v80; // [rsp+130h] [rbp-2B0h]
  __int64 v81; // [rsp+140h] [rbp-2A0h]
  _QWORD v82[2]; // [rsp+150h] [rbp-290h] BYREF
  const char *v83; // [rsp+160h] [rbp-280h]
  __int64 v84; // [rsp+168h] [rbp-278h]
  char *v85; // [rsp+170h] [rbp-270h]
  __int64 v86; // [rsp+178h] [rbp-268h]
  _BYTE *v87; // [rsp+180h] [rbp-260h]
  __int64 v88; // [rsp+188h] [rbp-258h]
  _BYTE *v89; // [rsp+190h] [rbp-250h]
  __int64 v90; // [rsp+198h] [rbp-248h]
  __m128i v91; // [rsp+1A0h] [rbp-240h]
  __int64 v92; // [rsp+1B0h] [rbp-230h]
  __int64 v93; // [rsp+1B8h] [rbp-228h]
  __int64 v94; // [rsp+1C0h] [rbp-220h]
  __int64 v95; // [rsp+1C8h] [rbp-218h]
  _BYTE *v96; // [rsp+1D0h] [rbp-210h] BYREF
  __int64 v97; // [rsp+1D8h] [rbp-208h]
  __int64 v98; // [rsp+1E0h] [rbp-200h]
  _BYTE v99[136]; // [rsp+1E8h] [rbp-1F8h] BYREF
  _BYTE *v100; // [rsp+270h] [rbp-170h] BYREF
  __int64 v101; // [rsp+278h] [rbp-168h]
  __int64 v102; // [rsp+280h] [rbp-160h]
  _BYTE v103[136]; // [rsp+288h] [rbp-158h] BYREF
  __m128i v104; // [rsp+310h] [rbp-D0h] BYREF
  __m128i v105; // [rsp+320h] [rbp-C0h] BYREF
  char *v106; // [rsp+330h] [rbp-B0h]
  __int64 v107; // [rsp+338h] [rbp-A8h]
  char v108; // [rsp+340h] [rbp-A0h] BYREF
  char *v109; // [rsp+350h] [rbp-90h]
  __int64 v110; // [rsp+358h] [rbp-88h]
  char v111; // [rsp+360h] [rbp-80h] BYREF
  char v112; // [rsp+370h] [rbp-70h] BYREF

  v15 = a8;
  if ( !byte_4F82770 )
  {
    v62 = a5;
    v45 = sub_2207590(&byte_4F82770);
    a5 = v62;
    if ( v45 )
    {
      qword_4F82790 = -1;
      qword_4F82780 = (__int64)&qword_4F82790;
      qword_4F82788 = 0xC00000003LL;
      dword_4F82798 = -1;
      __cxa_atexit((void (*)(void *))sub_BC4ED0, &qword_4F82780, &qword_4A427C0);
      sub_2207640(&byte_4F82770);
      a5 = v62;
    }
  }
  v104.m128i_i64[0] = a2;
  v104.m128i_i64[1] = a3;
  v16 = _mm_load_si128(&v104);
  v105.m128i_i64[0] = a4;
  v105.m128i_i64[1] = a5;
  v17 = _mm_load_si128(&v105);
  v72 = v74;
  v73 = 0x300000002LL;
  v74[0] = v16;
  v74[1] = v17;
  if ( !byte_4F82728 )
  {
    v42 = &v104;
    if ( (unsigned int)sub_2207590(&byte_4F82728) )
    {
      v104 = (__m128i)(unsigned __int64)&v105;
      v106 = &v108;
      v109 = &v111;
      qword_4F82740 = (__int64)&unk_4F82750;
      qword_4F82748 = 0x100000000LL;
      v105.m128i_i8[0] = 0;
      v107 = 0;
      v108 = 0;
      v110 = 0;
      v111 = 0;
      sub_95D880((__int64)&qword_4F82740, 3);
      v43 = (__int64 *)(qword_4F82740 + 32LL * (unsigned int)qword_4F82748);
      do
      {
        if ( v43 )
        {
          *v43 = (__int64)(v43 + 2);
          sub_BC50C0(v43, v42->m128i_i64[0], v42->m128i_i64[0] + v42->m128i_i64[1]);
        }
        v42 += 2;
        v43 += 4;
      }
      while ( v42 != (__m128i *)&v112 );
      LODWORD(qword_4F82748) = qword_4F82748 + 3;
      v15 = a8;
      v44 = v42;
      __cxa_atexit((void (*)(void *))sub_BC5500, &qword_4F82740, &qword_4A427C0);
      sub_2207640(&byte_4F82728);
      do
      {
        v44 -= 2;
        if ( (__m128i *)v44->m128i_i64[0] != &v44[1] )
          j_j___libc_free_0(v44->m128i_i64[0], v44[1].m128i_i64[0] + 1);
      }
      while ( v44 != &v104 );
    }
  }
  v18 = a1 + 1;
  if ( (unsigned int)sub_BC7640((__int64)&qword_4F82780, (__int64)v72, (unsigned int)v73, (__int64)&qword_4F82740) )
  {
    a1->m128i_i64[0] = (__int64)v18;
    v19 = &v104;
    v104.m128i_i64[0] = 32;
    v20 = sub_22409D0(a1, &v104, 0);
    v21 = v104.m128i_i64[0];
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F566E0);
    a1->m128i_i64[0] = v20;
    a1[1].m128i_i64[0] = v21;
    *(__m128i *)v20 = si128;
    *(__m128i *)(v20 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F566F0);
    v23 = v104.m128i_i64[0];
    v24 = a1->m128i_i64[0];
    a1->m128i_i64[1] = v104.m128i_i64[0];
    *(_BYTE *)(v24 + v23) = 0;
    goto LABEL_5;
  }
  if ( !byte_4F826F0 && (unsigned int)sub_2207590(&byte_4F826F0) )
  {
    sub_C86E60(&qword_4F82700, qword_4F82CA8, qword_4F82CB0, 0, 0);
    __cxa_atexit((void (*)(void *))sub_BC5B10, &qword_4F82700, &qword_4A427C0);
    sub_2207640(&byte_4F826F0);
  }
  if ( (byte_4F82720 & 1) != 0 )
  {
    a1->m128i_i64[0] = (__int64)v18;
    v19 = &v104;
    v104.m128i_i64[0] = 31;
    v26 = (__m128i *)sub_22409D0(a1, &v104, 0);
    v27 = v104.m128i_i64[0];
    v28 = _mm_load_si128((const __m128i *)&xmmword_3F56700);
    a1->m128i_i64[0] = (__int64)v26;
    a1[1].m128i_i64[0] = v27;
    qmemcpy(&v26[1], "iff executable.", 15);
    *v26 = v28;
    v29 = v104.m128i_i64[0];
    v30 = a1->m128i_i64[0];
    a1->m128i_i64[1] = v104.m128i_i64[0];
    *(_BYTE *)(v30 + v29) = 0;
    goto LABEL_5;
  }
  v84 = v15;
  v96 = v99;
  v100 = v103;
  v104 = (__m128i)(unsigned __int64)&v105.m128i_u64[1];
  v82[0] = "--old-line-format=";
  LOWORD(v85) = 1283;
  v83 = a7;
  v97 = 0;
  v98 = 128;
  v101 = 0;
  v102 = 128;
  v105.m128i_i64[0] = 128;
  sub_CA0EC0(v82, &v96);
  v82[0] = "--new-line-format=";
  LOWORD(v85) = 1283;
  v83 = a9;
  v84 = a10;
  sub_CA0EC0(v82, &v100);
  v82[0] = "--unchanged-line-format=";
  LOWORD(v85) = 1283;
  v83 = a11;
  v84 = a12;
  sub_CA0EC0(v82, &v104);
  v84 = 2;
  v86 = 2;
  v82[0] = qword_4F82CA8;
  v82[1] = qword_4F82CB0;
  v83 = "-w";
  v85 = "-d";
  v87 = v96;
  v88 = v97;
  v89 = v100;
  v90 = v101;
  v91 = v104;
  v92 = *(_QWORD *)qword_4F82740;
  v31 = *(_QWORD *)(qword_4F82740 + 8);
  v75 = 0;
  v93 = v31;
  v32 = *(_QWORD *)(qword_4F82740 + 32);
  v76 = 0;
  v94 = v32;
  v95 = *(_QWORD *)(qword_4F82740 + 40);
  v33 = *(_QWORD *)(qword_4F82740 + 64);
  v34 = *(_QWORD *)(qword_4F82740 + 72);
  LOBYTE(v70[0]) = 0;
  v77 = v33;
  v78 = v34;
  v79 = 1;
  v81 = 0;
  v80 = 0;
  if ( (int)sub_C881F0(
              qword_4F82700,
              qword_4F82708,
              (unsigned int)v82,
              8,
              (unsigned int)&v75,
              3,
              v68,
              n,
              0,
              0,
              0,
              0,
              0,
              0) >= 0 )
  {
    v66 = 0;
    v71 = 260;
    dest = &v67;
    v67.m128i_i8[0] = 0;
    v68 = qword_4F82740 + 64;
    sub_C7EA90(&v63, &v68, 0, 1, 0, 0);
    if ( (v64 & 1) != 0 || !v63 )
    {
      a1->m128i_i64[0] = (__int64)v18;
      v19 = (__m128i *)&v68;
      v68 = 22;
      v51 = sub_22409D0(a1, &v68, 0);
      v52 = v68;
      v53 = _mm_load_si128((const __m128i *)&xmmword_3F56720);
      a1->m128i_i64[0] = v51;
      a1[1].m128i_i64[0] = v52;
      *(_DWORD *)(v51 + 16) = 1819636581;
      *(_WORD *)(v51 + 20) = 11892;
      *(__m128i *)v51 = v53;
      v54 = v68;
      v55 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v68;
      *(_BYTE *)(v55 + v54) = 0;
LABEL_47:
      if ( (v64 & 1) == 0 && v63 )
        (*(void (__fastcall **)(_QWORD *))(*v63 + 8LL))(v63);
      if ( dest != &v67 )
      {
        v19 = (__m128i *)(v67.m128i_i64[0] + 1);
        j_j___libc_free_0(dest, v67.m128i_i64[0] + 1);
      }
      goto LABEL_40;
    }
    v35 = (_BYTE *)v63[1];
    if ( v35 )
    {
      v36 = v63[2];
      v68 = (__int64)v70;
      sub_BC5450(&v68, v35, v36);
      v37 = dest;
      v38 = dest;
      if ( (_QWORD *)v68 != v70 )
      {
        if ( dest == &v67 )
        {
          dest = (void *)v68;
          v66 = n;
          v67.m128i_i64[0] = v70[0];
        }
        else
        {
          v39 = v67.m128i_i64[0];
          dest = (void *)v68;
          v66 = n;
          v67.m128i_i64[0] = v70[0];
          if ( v38 )
          {
            v68 = (__int64)v38;
            v70[0] = v39;
LABEL_19:
            n = 0;
            *v38 = 0;
            if ( (_QWORD *)v68 != v70 )
              j_j___libc_free_0(v68, v70[0] + 1LL);
            v19 = (__m128i *)(unsigned int)qword_4F82748;
            if ( (unsigned int)sub_BC5E90(qword_4F82740, (unsigned int)qword_4F82748) )
            {
              a1->m128i_i64[0] = (__int64)v18;
              v19 = (__m128i *)&v68;
              v68 = 32;
              v57 = sub_22409D0(a1, &v68, 0);
              v58 = v68;
              v59 = _mm_load_si128((const __m128i *)&xmmword_3F56730);
              a1->m128i_i64[0] = v57;
              a1[1].m128i_i64[0] = v58;
              *(__m128i *)v57 = v59;
              *(__m128i *)(v57 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F566F0);
              v60 = v68;
              v61 = a1->m128i_i64[0];
              a1->m128i_i64[1] = v68;
              *(_BYTE *)(v61 + v60) = 0;
            }
            else
            {
              v40 = (__m128i *)dest;
              a1->m128i_i64[0] = (__int64)v18;
              if ( v40 == &v67 )
              {
                a1[1] = _mm_load_si128(&v67);
              }
              else
              {
                a1->m128i_i64[0] = (__int64)v40;
                a1[1].m128i_i64[0] = v67.m128i_i64[0];
              }
              v41 = v66;
              dest = &v67;
              v66 = 0;
              a1->m128i_i64[1] = v41;
              v67.m128i_i8[0] = 0;
            }
            goto LABEL_47;
          }
        }
        v68 = (__int64)v70;
        v38 = v70;
        goto LABEL_19;
      }
      v56 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = v70[0];
        else
          memcpy(dest, v70, n);
        v56 = n;
        v37 = dest;
      }
    }
    else
    {
      LOBYTE(v70[0]) = 0;
      v37 = dest;
      v56 = 0;
      v68 = (__int64)v70;
    }
    v66 = v56;
    v37[v56] = 0;
    v38 = (_BYTE *)v68;
    goto LABEL_19;
  }
  a1->m128i_i64[0] = (__int64)v18;
  v19 = (__m128i *)&v68;
  v68 = 28;
  v46 = (__m128i *)sub_22409D0(a1, &v68, 0);
  v47 = v68;
  v48 = _mm_load_si128((const __m128i *)&xmmword_3F56710);
  a1->m128i_i64[0] = (__int64)v46;
  a1[1].m128i_i64[0] = v47;
  qmemcpy(&v46[1], "system diff.", 12);
  *v46 = v48;
  v49 = v68;
  v50 = a1->m128i_i64[0];
  a1->m128i_i64[1] = v68;
  *(_BYTE *)(v50 + v49) = 0;
LABEL_40:
  if ( (unsigned __int64 *)v104.m128i_i64[0] != &v105.m128i_u64[1] )
    _libc_free(v104.m128i_i64[0], v19);
  if ( v100 != v103 )
    _libc_free(v100, v19);
  if ( v96 != v99 )
    _libc_free(v96, v19);
LABEL_5:
  if ( v72 != v74 )
    _libc_free(v72, v19);
  return a1;
}
