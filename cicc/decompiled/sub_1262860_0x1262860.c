// Function: sub_1262860
// Address: 0x1262860
//
__int64 __fastcall sub_1262860(
        const char *a1,
        int a2,
        __int64 *a3,
        const char *a4,
        const char *a5,
        int a6,
        unsigned __int8 a7,
        unsigned __int8 a8,
        unsigned __int8 a9)
{
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r8
  unsigned int v14; // r15d
  size_t v15; // rdx
  size_t v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned int *v21; // r8
  __int64 v22; // r9
  int v23; // r10d
  int v24; // r13d
  unsigned int v25; // r13d
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r9
  __int64 v31; // r8
  __int64 v32; // rbx
  __int64 v33; // rdi
  __int64 v34; // r8
  __int64 v35; // rbx
  __int64 v36; // rdi
  int v38; // eax
  size_t v39; // r15
  _QWORD *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rax
  __m128i *v44; // rdx
  __m128i si128; // xmm0
  __int64 v46; // rax
  size_t v47; // rax
  size_t v48; // r14
  _QWORD *v49; // rdx
  _WORD *v50; // rdi
  _QWORD *v51; // rdi
  unsigned int v53; // [rsp+20h] [rbp-290h]
  _QWORD *src; // [rsp+78h] [rbp-238h]
  char v57; // [rsp+8Ah] [rbp-226h] BYREF
  unsigned __int8 v58; // [rsp+8Bh] [rbp-225h] BYREF
  unsigned int v59; // [rsp+8Ch] [rbp-224h] BYREF
  unsigned int v60; // [rsp+90h] [rbp-220h] BYREF
  unsigned int v61; // [rsp+94h] [rbp-21Ch] BYREF
  __int64 v62; // [rsp+98h] [rbp-218h] BYREF
  _QWORD *v63; // [rsp+A0h] [rbp-210h] BYREF
  __int64 v64; // [rsp+A8h] [rbp-208h]
  _QWORD v65[2]; // [rsp+B0h] [rbp-200h] BYREF
  _QWORD v66[2]; // [rsp+C0h] [rbp-1F0h] BYREF
  _QWORD v67[2]; // [rsp+D0h] [rbp-1E0h] BYREF
  _BYTE *v68; // [rsp+E0h] [rbp-1D0h] BYREF
  __int64 v69; // [rsp+E8h] [rbp-1C8h]
  _QWORD v70[2]; // [rsp+F0h] [rbp-1C0h] BYREF
  unsigned __int64 v71; // [rsp+100h] [rbp-1B0h] BYREF
  size_t n; // [rsp+108h] [rbp-1A8h]
  _QWORD v73[2]; // [rsp+110h] [rbp-1A0h] BYREF
  _WORD *v74; // [rsp+120h] [rbp-190h] BYREF
  __int64 v75; // [rsp+128h] [rbp-188h]
  _WORD v76[64]; // [rsp+130h] [rbp-180h] BYREF
  char v77[8]; // [rsp+1B0h] [rbp-100h] BYREF
  _QWORD v78[2]; // [rsp+1B8h] [rbp-F8h] BYREF
  _QWORD v79[2]; // [rsp+1C8h] [rbp-E8h] BYREF
  _QWORD v80[2]; // [rsp+1D8h] [rbp-D8h] BYREF
  _QWORD v81[2]; // [rsp+1E8h] [rbp-C8h] BYREF
  __int16 v82; // [rsp+1F8h] [rbp-B8h]
  char v83; // [rsp+1FAh] [rbp-B6h]
  int v84; // [rsp+1FCh] [rbp-B4h]
  __int64 v85; // [rsp+200h] [rbp-B0h]
  _QWORD v86[2]; // [rsp+218h] [rbp-98h] BYREF
  _QWORD v87[2]; // [rsp+228h] [rbp-88h] BYREF
  _QWORD *v88; // [rsp+238h] [rbp-78h]
  __int64 v89; // [rsp+240h] [rbp-70h]
  _QWORD v90[2]; // [rsp+248h] [rbp-68h] BYREF
  __int16 v91; // [rsp+258h] [rbp-58h]
  char v92; // [rsp+25Ah] [rbp-56h]
  int v93; // [rsp+25Ch] [rbp-54h]
  __int64 v94; // [rsp+260h] [rbp-50h]

  v11 = sub_22077B0(8);
  v12 = v11;
  if ( v11 )
    sub_1602D10(v11);
  v63 = v65;
  v78[0] = v79;
  v80[0] = v81;
  v88 = v90;
  v82 = 0;
  v86[0] = v87;
  v91 = 0;
  v62 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v64 = 0;
  LOBYTE(v65[0]) = 0;
  v78[1] = 0;
  LOBYTE(v79[0]) = 0;
  v80[1] = 0;
  LOBYTE(v81[0]) = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86[1] = 0;
  LOBYTE(v87[0]) = 0;
  v89 = 0;
  LOBYTE(v90[0]) = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v68 = v70;
  v66[0] = v67;
  v66[1] = 0;
  LOBYTE(v67[0]) = 0;
  v69 = 0;
  LOBYTE(v70[0]) = 0;
  v60 = 0;
  v61 = 0;
  if ( (unsigned int)sub_125FB30(
                       a1,
                       a4,
                       a5,
                       a3,
                       a8,
                       (__int64)v77,
                       (__int64)&v63,
                       &v57,
                       &v58,
                       (__int64)&v68,
                       (__int64)&v60,
                       (__int64)&v61) )
  {
    sub_223E0D0(qword_4FD4BE0, "\n Error processing command line: ", 33);
    v25 = 1;
    v46 = sub_223E0D0(qword_4FD4BE0, v63, v64);
    v17 = (__int64)"\n";
    sub_223E0D0(v46, "\n", 1);
    goto LABEL_12;
  }
  if ( v69 )
  {
    v13 = v61;
    v14 = v60;
    v15 = 0;
    if ( a4 )
    {
      v53 = v61;
      v16 = strlen(a4);
      v13 = v53;
      v15 = v16;
    }
    sub_16DAE00(0, a4, v15, v14, v13);
  }
  v17 = v12;
  v59 = 0;
  v18 = sub_1265340(a2, v12, (unsigned int)v78, (unsigned int)&v62, (unsigned int)&v59, a7, 0, a9, 0);
  v21 = &v59;
  v22 = a7;
  v23 = a9;
  v24 = v18;
  if ( v18 || a9 == 1 )
  {
    if ( !v83 )
    {
      v76[0] = 260;
      v74 = v80;
      sub_16C50A0(&v74, 1);
      LODWORD(v22) = a7;
      v23 = a9;
    }
    v59 = 0;
    sub_1265970(
      (_DWORD)a1,
      v24,
      (unsigned int)v86,
      a6,
      v58,
      (unsigned int)v66,
      (__int64)&v62,
      (__int64)&v59,
      v22,
      a8,
      0,
      v23);
    v17 = v59;
    if ( v59 && !a9 )
    {
      v25 = 1;
      goto LABEL_12;
    }
    if ( !((__int64 (*)(void))sub_16DA870)() )
    {
LABEL_67:
      v25 = 0;
      goto LABEL_12;
    }
    v38 = sub_2241AC0(&v68, "-");
    v71 = (unsigned __int64)v73;
    if ( v38 )
    {
      sub_125C500((__int64 *)&v71, v68, (__int64)&v68[v69]);
      goto LABEL_56;
    }
    if ( !a5 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v47 = strlen(a5);
    v74 = (_WORD *)v47;
    v48 = v47;
    if ( v47 > 0xF )
    {
      v71 = sub_22409D0(&v71, &v74, 0);
      v51 = (_QWORD *)v71;
      v73[0] = v74;
    }
    else
    {
      if ( v47 == 1 )
      {
        LOBYTE(v73[0]) = *a5;
        v49 = v73;
LABEL_73:
        n = v47;
        *((_BYTE *)v49 + v47) = 0;
LABEL_56:
        v39 = n;
        v40 = (_QWORD *)v71;
        v74 = v76;
        v75 = 0x8000000000LL;
        if ( n > 0x80 )
        {
          src = (_QWORD *)v71;
          sub_16CD150(&v74, v76, n, 1);
          v40 = src;
          v50 = (_WORD *)((char *)v74 + (unsigned int)v75);
        }
        else
        {
          if ( !n )
          {
LABEL_58:
            LODWORD(v75) = v39;
            v41 = (unsigned int)v39;
            if ( v40 != v73 )
            {
              j_j___libc_free_0(v40, v73[0] + 1LL);
              v41 = (unsigned int)v75;
            }
            v17 = (__int64)v74;
            sub_16DD960(&v71, v74, v41, "-", 1);
            if ( (v71 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            {
              v71 = v71 & 0xFFFFFFFFFFFFFFFELL | 1;
              v43 = sub_16E8CB0(&v71, v17, v42);
              v44 = *(__m128i **)(v43 + 24);
              if ( *(_QWORD *)(v43 + 16) - (_QWORD)v44 <= 0x2Au )
              {
                v17 = (__int64)"Error: Failed to write time profiler data.\n";
                sub_16E7EE0(v43, "Error: Failed to write time profiler data.\n", 43);
              }
              else
              {
                si128 = _mm_load_si128((const __m128i *)&xmmword_3E9F970);
                qmemcpy(&v44[2], "iler data.\n", 11);
                *v44 = si128;
                v44[1] = _mm_load_si128((const __m128i *)&xmmword_3E9F980);
                *(_QWORD *)(v43 + 24) += 43LL;
              }
            }
            else
            {
              v71 = 0;
            }
            sub_16DB140();
            if ( (v71 & 1) != 0 || (v71 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_16BCAE0(&v71);
            if ( v74 != v76 )
              _libc_free(v74, v17);
            goto LABEL_67;
          }
          v50 = v76;
        }
        memcpy(v50, v40, v39);
        v40 = (_QWORD *)v71;
        LODWORD(v39) = v75 + v39;
        goto LABEL_58;
      }
      if ( !v47 )
      {
        v49 = v73;
        goto LABEL_73;
      }
      v51 = v73;
    }
    memcpy(v51, a5, v48);
    v47 = (size_t)v74;
    v49 = (_QWORD *)v71;
    goto LABEL_73;
  }
  v25 = 1;
  if ( v12 )
  {
    sub_16025D0(v12, v12, v19, v20, &v59, a7);
    v26 = v12;
    v17 = 8;
    v12 = 0;
    j_j___libc_free_0(v26, 8);
  }
LABEL_12:
  v27 = v62;
  if ( v62 )
  {
    j_j___libc_free_0_0(v62);
    v62 = 0;
  }
  if ( sub_16DA870(v27, v17, v19, v20, v21, v22) )
    sub_16DB140();
  if ( v68 != (_BYTE *)v70 )
  {
    v17 = v70[0] + 1LL;
    j_j___libc_free_0(v68, v70[0] + 1LL);
  }
  if ( (_QWORD *)v66[0] != v67 )
  {
    v17 = v67[0] + 1LL;
    j_j___libc_free_0(v66[0], v67[0] + 1LL);
  }
  v31 = v94;
  if ( v94 )
  {
    v28 = (unsigned int)v93;
    if ( v93 <= 0 )
      goto LABEL_27;
    v32 = 0;
    do
    {
      v33 = *(_QWORD *)(v31 + 8 * v32);
      if ( v33 )
      {
        j_j___libc_free_0_0(v33);
        v31 = v94;
      }
      ++v32;
    }
    while ( v93 > (int)v32 );
    if ( v31 )
LABEL_27:
      j_j___libc_free_0_0(v31);
  }
  if ( v88 != v90 )
  {
    v17 = v90[0] + 1LL;
    j_j___libc_free_0(v88, v90[0] + 1LL);
  }
  if ( (_QWORD *)v86[0] != v87 )
  {
    v17 = v87[0] + 1LL;
    j_j___libc_free_0(v86[0], v87[0] + 1LL);
  }
  v34 = v85;
  if ( v85 )
  {
    if ( v84 <= 0 )
      goto LABEL_39;
    v35 = 0;
    do
    {
      v36 = *(_QWORD *)(v34 + 8 * v35);
      if ( v36 )
      {
        j_j___libc_free_0_0(v36);
        v34 = v85;
      }
      ++v35;
    }
    while ( v84 > (int)v35 );
    if ( v34 )
LABEL_39:
      j_j___libc_free_0_0(v34);
  }
  if ( (_QWORD *)v80[0] != v81 )
  {
    v17 = v81[0] + 1LL;
    j_j___libc_free_0(v80[0], v81[0] + 1LL);
  }
  if ( (_QWORD *)v78[0] != v79 )
  {
    v17 = v79[0] + 1LL;
    j_j___libc_free_0(v78[0], v79[0] + 1LL);
  }
  if ( v63 != v65 )
  {
    v17 = v65[0] + 1LL;
    j_j___libc_free_0(v63, v65[0] + 1LL);
  }
  if ( v12 )
  {
    sub_16025D0(v12, v17, v28, v29, v34, v30);
    j_j___libc_free_0(v12, 8);
  }
  return v25;
}
