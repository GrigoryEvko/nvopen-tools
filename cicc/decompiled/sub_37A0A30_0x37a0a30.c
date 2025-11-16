// Function: sub_37A0A30
// Address: 0x37a0a30
//
__int64 __fastcall sub_37A0A30(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r15d
  __int64 v4; // r9
  __int64 v5; // rax
  __int16 v6; // dx
  __int64 v7; // rax
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v9; // rax
  unsigned __int16 v10; // si
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 (__fastcall *v18)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v19; // rcx
  __int64 v20; // rax
  __int16 v21; // r14
  int v22; // edx
  __int64 v23; // rdi
  __int64 *v24; // rax
  unsigned __int64 v25; // rbx
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // r14d
  unsigned int v29; // eax
  _QWORD *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rcx
  int v33; // esi
  const void *v34; // r14
  __int64 v35; // r15
  __int64 v36; // rbx
  __int64 v37; // r12
  __int64 v38; // rdx
  __int64 v39; // r13
  __int64 v40; // rax
  __int64 v41; // rdx
  _QWORD *v42; // rax
  void *v43; // rdi
  __int64 v44; // r12
  unsigned int v45; // r13d
  unsigned int v46; // edx
  unsigned int v47; // r13d
  unsigned __int16 *v48; // rdx
  int v49; // eax
  char *v50; // rax
  _QWORD *v51; // rcx
  char *i; // rdx
  __int64 v53; // rax
  unsigned int v54; // r13d
  __m128i v55; // xmm0
  __int64 v56; // r12
  __int64 v57; // rsi
  __int64 v58; // rbx
  __int128 v59; // rax
  __int64 v60; // r9
  unsigned __int8 *v61; // rax
  char *v62; // rbx
  int v63; // edx
  _QWORD *v64; // rdi
  _QWORD *v65; // rbx
  __int64 v66; // r9
  int v67; // edx
  int v68; // r13d
  __int64 v69; // rax
  char *v70; // rcx
  unsigned __int8 *v71; // rax
  unsigned int v73; // edx
  int v74; // kr00_4
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rdx
  __int64 v78; // r15
  __int64 v79; // rbx
  __int64 v80; // r8
  __int64 v81; // r9
  char *v82; // rax
  int v83; // edx
  int v84; // r14d
  char *v85; // rdx
  char *v86; // rcx
  __int64 v87; // rsi
  __int64 v88; // rcx
  char *v89; // rax
  __int64 v90; // rdx
  char *v91; // rax
  __int64 v92; // rdx
  unsigned int v93; // ecx
  __int64 v94; // rax
  __int64 *v95; // rax
  __int128 v96; // [rsp-10h] [rbp-250h]
  __int128 v97; // [rsp-10h] [rbp-250h]
  __int64 v98; // [rsp+8h] [rbp-238h]
  unsigned int v99; // [rsp+1Ch] [rbp-224h]
  __int64 v100; // [rsp+20h] [rbp-220h]
  char v102; // [rsp+33h] [rbp-20Dh]
  int v103; // [rsp+34h] [rbp-20Ch]
  __int64 v104; // [rsp+38h] [rbp-208h]
  __int128 v106; // [rsp+50h] [rbp-1F0h]
  __int16 v107; // [rsp+62h] [rbp-1DEh]
  unsigned int v108; // [rsp+64h] [rbp-1DCh]
  _QWORD *v109; // [rsp+68h] [rbp-1D8h]
  __int64 v110; // [rsp+C0h] [rbp-180h] BYREF
  __int64 v111; // [rsp+C8h] [rbp-178h]
  __int64 v112; // [rsp+D0h] [rbp-170h] BYREF
  __int64 v113; // [rsp+D8h] [rbp-168h]
  __int64 v114; // [rsp+E0h] [rbp-160h] BYREF
  int v115; // [rsp+E8h] [rbp-158h]
  __int64 v116; // [rsp+F0h] [rbp-150h] BYREF
  int v117; // [rsp+F8h] [rbp-148h]
  void *s; // [rsp+100h] [rbp-140h] BYREF
  __int64 v119; // [rsp+108h] [rbp-138h]
  _QWORD v120[38]; // [rsp+110h] [rbp-130h] BYREF

  v4 = *a1;
  v5 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
  v6 = *(_WORD *)v5;
  v7 = *(_QWORD *)(v5 + 8);
  LOWORD(v110) = v6;
  v111 = v7;
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v4 + 592LL);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v12 = a1[1];
  if ( v8 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&s, v4, *(_QWORD *)(v12 + 64), v10, v11);
    LOWORD(v112) = v119;
    v113 = v120[0];
  }
  else
  {
    LODWORD(v112) = v8(v4, *(_QWORD *)(v12 + 64), v10, v11);
    v113 = v76;
  }
  v13 = *(_QWORD *)(a2 + 80);
  v114 = v13;
  if ( v13 )
    sub_B96E90((__int64)&v114, v13, 1);
  v115 = *(_DWORD *)(a2 + 72);
  v108 = *(_DWORD *)(a2 + 64);
  v14 = *a1;
  sub_2FE6CC0((__int64)&s, *a1, *(_QWORD *)(a1[1] + 64), (unsigned __int16)v110, v111);
  if ( (_BYTE)s != 7 )
  {
    v21 = v112;
    if ( (_WORD)v112 )
      v45 = word_4456340[(unsigned __int16)v112 - 1];
    else
      v45 = sub_3007240((__int64)&v112);
    if ( (_WORD)v110 )
      v19 = word_4456340[(unsigned __int16)v110 - 1];
    else
      v19 = sub_3007240((__int64)&v110);
    v102 = 0;
    v46 = v45 % (unsigned int)v19;
    v47 = v45 / (unsigned int)v19;
    if ( v46 )
      goto LABEL_33;
    v78 = v47;
    v79 = sub_3288990(a1[1], (unsigned int)v110, v111);
    v82 = (char *)v120;
    v84 = v83;
    v119 = 0x1000000000LL;
    v85 = (char *)v120;
    s = v120;
    if ( v47 )
    {
      if ( v47 > 0x10uLL )
      {
        sub_C8D5F0((__int64)&s, v120, v47, 0x10u, v80, v81);
        v85 = (char *)s;
        v82 = (char *)s + 16 * (unsigned int)v119;
      }
      v86 = &v85[16 * v47];
      if ( v86 != v82 )
      {
        do
        {
          if ( v82 )
          {
            *(_QWORD *)v82 = 0;
            *((_DWORD *)v82 + 2) = 0;
          }
          v82 += 16;
        }
        while ( v86 != v82 );
        v85 = (char *)s;
      }
      LODWORD(v119) = v47;
      if ( !v108 )
      {
LABEL_99:
        v93 = v108;
        do
        {
          v94 = v93++;
          v95 = (__int64 *)&v85[16 * v94];
          *v95 = v79;
          *((_DWORD *)v95 + 2) = v84;
          v85 = (char *)s;
        }
        while ( v47 != v93 );
LABEL_101:
        v78 = (unsigned int)v119;
LABEL_102:
        *((_QWORD *)&v97 + 1) = v78;
        *(_QWORD *)&v97 = v85;
        v71 = sub_33FC220((_QWORD *)a1[1], 159, (__int64)&v114, (unsigned int)v112, v113, v81, v97);
        goto LABEL_63;
      }
    }
    else
    {
      v85 = (char *)v120;
      if ( !v108 )
        goto LABEL_102;
    }
    v87 = 0;
    v88 = 0;
    v89 = v85;
    v81 = a2;
    while ( 1 )
    {
      v90 = v88++;
      v91 = &v89[16 * v90];
      v92 = *(_QWORD *)(a2 + 40);
      *(_QWORD *)v91 = *(_QWORD *)(v92 + v87);
      LODWORD(v92) = *(_DWORD *)(v92 + v87 + 8);
      v87 += 40;
      *((_DWORD *)v91 + 2) = v92;
      if ( v108 <= (unsigned int)v88 )
        break;
      v89 = (char *)s;
    }
    v85 = (char *)s;
    if ( v47 == v108 )
      goto LABEL_101;
    goto LABEL_99;
  }
  v17 = a1[1];
  v18 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v18 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&s, *a1, *(_QWORD *)(v17 + 64), v110, v111);
    v14 = (unsigned __int16)v119;
    v20 = v120[0];
  }
  else
  {
    v14 = (unsigned int)v18(*a1, *(_QWORD *)(v17 + 64), v110, v111);
    v20 = v77;
  }
  v21 = v112;
  if ( (_WORD)v112 != (_WORD)v14 )
  {
LABEL_77:
    v102 = 1;
    goto LABEL_33;
  }
  if ( !(_WORD)v14 && v113 != v20 )
  {
    v102 = 1;
    goto LABEL_73;
  }
  v15 = v108;
  if ( v108 <= 1 )
  {
    if ( v108 == 1 )
    {
      v23 = *(_QWORD *)(a2 + 40);
LABEL_108:
      v44 = sub_379AB60((__int64)a1, *(_QWORD *)v23, *(_QWORD *)(v23 + 8));
      goto LABEL_65;
    }
    v21 = v14;
    goto LABEL_77;
  }
  v22 = 1;
  v23 = *(_QWORD *)(a2 + 40);
  v24 = (__int64 *)(v23 + 40);
  while ( 1 )
  {
    v19 = *v24;
    if ( *(_DWORD *)(*v24 + 24) != 51 )
      break;
    ++v22;
    v24 += 5;
    if ( v22 == v108 )
      goto LABEL_108;
  }
  v102 = 1;
  v21 = v14;
  if ( v108 != 2 )
  {
LABEL_33:
    if ( v21 )
    {
      if ( (unsigned __int16)(v21 - 176) > 0x34u )
        goto LABEL_35;
      goto LABEL_80;
    }
LABEL_73:
    if ( !sub_3007100((__int64)&v112) )
    {
LABEL_74:
      v99 = sub_3007130((__int64)&v112, v14);
LABEL_36:
      if ( (_WORD)v110 )
      {
        if ( (unsigned __int16)(v110 - 176) > 0x34u )
          goto LABEL_38;
      }
      else if ( !sub_3007100((__int64)&v110) )
      {
        goto LABEL_69;
      }
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      if ( (_WORD)v110 )
      {
        if ( (unsigned __int16)(v110 - 176) <= 0x34u )
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
LABEL_38:
        v48 = word_4456340;
        v103 = word_4456340[(unsigned __int16)v110 - 1];
        v49 = (unsigned __int16)v112;
        if ( (_WORD)v112 )
        {
LABEL_39:
          v98 = 0;
          v107 = word_4456580[v49 - 1];
LABEL_40:
          v50 = (char *)v120;
          v51 = v120;
          s = v120;
          v119 = 0x1000000000LL;
          if ( v99 )
          {
            if ( v99 > 0x10uLL )
            {
              sub_C8D5F0((__int64)&s, v120, v99, 0x10u, v15, v16);
              v51 = s;
              v50 = (char *)s + 16 * (unsigned int)v119;
            }
            for ( i = (char *)&v51[2 * v99]; i != v50; v50 += 16 )
            {
              if ( v50 )
              {
                *(_QWORD *)v50 = 0;
                *((_DWORD *)v50 + 2) = 0;
              }
            }
            LODWORD(v119) = v99;
          }
          if ( v108 )
          {
            v53 = v108;
            HIWORD(v54) = HIWORD(v3);
            v104 = 0;
            v108 = 0;
            v100 = 40 * v53;
            do
            {
              v55 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v104));
              v106 = (__int128)v55;
              if ( v102 )
              {
                *(_QWORD *)&v106 = sub_379AB60((__int64)a1, v55.m128i_u64[0], v55.m128i_i64[1]);
                *((_QWORD *)&v106 + 1) = v73 | v55.m128i_i64[1] & 0xFFFFFFFF00000000LL;
              }
              if ( v103 )
              {
                v56 = 0;
                do
                {
                  v57 = v56;
                  v58 = v108 + (unsigned int)v56++;
                  v109 = (_QWORD *)a1[1];
                  *(_QWORD *)&v59 = sub_3400EE0((__int64)v109, v57, (__int64)&v114, 0, v55);
                  LOWORD(v54) = v107;
                  v61 = sub_3406EB0(v109, 0x9Eu, (__int64)&v114, v54, v98, v60, v106, v59);
                  v62 = (char *)s + 16 * v58;
                  *(_QWORD *)v62 = v61;
                  *((_DWORD *)v62 + 2) = v63;
                }
                while ( v103 != v56 );
                v108 += v103;
              }
              v104 += 40;
            }
            while ( v100 != v104 );
          }
          LOWORD(v3) = v107;
          v116 = 0;
          v64 = (_QWORD *)a1[1];
          v117 = 0;
          v65 = sub_33F17F0(v64, 51, (__int64)&v116, v3, v98);
          v68 = v67;
          if ( v116 )
            sub_B91220((__int64)&v116, v116);
          if ( v99 > v108 )
          {
            v69 = 16LL * v108;
            do
            {
              v70 = (char *)s;
              *(_QWORD *)((char *)s + v69) = v65;
              *(_DWORD *)&v70[v69 + 8] = v68;
              v69 += 16;
            }
            while ( v69 != 16 * (v108 + (unsigned __int64)(v99 - 1 - v108) + 1) );
          }
          *((_QWORD *)&v96 + 1) = (unsigned int)v119;
          *(_QWORD *)&v96 = s;
          v71 = sub_33FC220((_QWORD *)a1[1], 156, (__int64)&v114, v112, v113, v66, v96);
LABEL_63:
          v43 = s;
          v44 = (__int64)v71;
          if ( s == v120 )
            goto LABEL_65;
          goto LABEL_64;
        }
LABEL_70:
        v74 = sub_3009970((__int64)&v112, v14, (__int64)v48, v19, v15);
        HIWORD(v3) = HIWORD(v74);
        v107 = v74;
        v98 = v75;
        goto LABEL_40;
      }
LABEL_69:
      v103 = sub_3007130((__int64)&v110, v14);
      v49 = (unsigned __int16)v112;
      if ( (_WORD)v112 )
        goto LABEL_39;
      goto LABEL_70;
    }
LABEL_80:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !(_WORD)v112 )
      goto LABEL_74;
    if ( (unsigned __int16)(v112 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_35:
    v99 = word_4456340[(unsigned __int16)v112 - 1];
    goto LABEL_36;
  }
  v25 = (unsigned int)sub_3281500(&v112, v14);
  v28 = sub_3281500(&v110, v14);
  s = v120;
  v119 = 0x1000000000LL;
  if ( (unsigned int)v25 > 0x10 )
  {
    sub_C8D5F0((__int64)&s, v120, v25, 4u, v26, v27);
    memset(s, 255, 4 * v25);
    LODWORD(v119) = v25;
    v30 = s;
  }
  else
  {
    if ( v25 )
    {
      v29 = 4 * v25;
      if ( 4 * v25 )
      {
        if ( v29 >= 8 )
        {
          *(_QWORD *)((char *)&v120[-1] + v29) = -1;
          memset(v120, 0xFFu, 8LL * ((v29 - 1) >> 3));
        }
        else if ( (v29 & 4) != 0 )
        {
          LODWORD(v120[0]) = -1;
          *(_DWORD *)((char *)&v119 + v29 + 4) = -1;
        }
        else if ( v29 )
        {
          LOBYTE(v120[0]) = -1;
        }
      }
    }
    LODWORD(v119) = v25;
    v30 = v120;
  }
  if ( v28 )
  {
    v31 = 0;
    do
    {
      *((_DWORD *)v30 + v31) = v31;
      v32 = (unsigned int)(v28 + v31);
      v33 = v25 + v31++;
      *((_DWORD *)s + v32) = v33;
      v30 = s;
    }
    while ( v28 != v31 );
  }
  v34 = v30;
  v35 = (unsigned int)v119;
  v36 = a1[1];
  v37 = sub_379AB60((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v39 = v38;
  v40 = sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v42 = sub_33FCE10(v36, (unsigned int)v112, v113, (__int64)&v114, v40, v41, a3, v37, v39, v34, v35);
  v43 = s;
  v44 = (__int64)v42;
  if ( s != v120 )
LABEL_64:
    _libc_free((unsigned __int64)v43);
LABEL_65:
  if ( v114 )
    sub_B91220((__int64)&v114, v114);
  return v44;
}
