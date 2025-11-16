// Function: sub_346F960
// Address: 0x346f960
//
__int64 __fastcall sub_346F960(__m128i a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  __int64 v9; // rsi
  __int64 v10; // rax
  __m128i v11; // xmm2
  unsigned int v12; // ebx
  __m128i v13; // xmm3
  __int64 v14; // rax
  unsigned __int16 v15; // r12
  unsigned __int16 *v16; // rdx
  int v17; // eax
  __int64 v18; // rdx
  unsigned __int16 *v19; // rdx
  __int64 v20; // rcx
  unsigned __int64 v21; // rsi
  unsigned __int16 v22; // ax
  __int64 *v23; // rdi
  unsigned int v24; // r12d
  int v25; // eax
  __int64 v26; // rcx
  __int64 v27; // r9
  __int64 v28; // r8
  unsigned int v29; // r15d
  char v30; // al
  __int64 v31; // r9
  __int64 v32; // r8
  unsigned int v33; // r12d
  int v34; // eax
  unsigned int v35; // edx
  unsigned int v36; // r12d
  unsigned __int64 *v37; // r15
  __m128i *v38; // rax
  unsigned int v39; // ebx
  unsigned __int64 v40; // r15
  unsigned int v41; // r12d
  __m128i v42; // rax
  __int64 v43; // r9
  __int64 v44; // rdx
  __int64 v45; // r9
  unsigned __int64 *v46; // rdi
  unsigned __int64 v47; // r8
  __int64 v48; // r10
  __int64 v49; // r11
  const __m128i *v50; // rsi
  __int64 v51; // rcx
  __int64 v52; // rdx
  const __m128i *v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  __int64 v57; // rsi
  __int64 v58; // rcx
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  unsigned __int64 v62; // r8
  __int64 v63; // r12
  unsigned __int64 v64; // rbx
  unsigned __int64 v65; // rdi
  __int64 v67; // rdx
  unsigned __int8 *v68; // rax
  unsigned int v69; // edx
  int v70; // r9d
  unsigned __int8 *v71; // rax
  unsigned int v72; // edx
  __int64 v73; // rdx
  __int64 v74; // rdx
  __int128 v75; // [rsp-10h] [rbp-1B0h]
  __int128 v76; // [rsp-10h] [rbp-1B0h]
  __int128 v77; // [rsp-10h] [rbp-1B0h]
  char v78; // [rsp+17h] [rbp-189h]
  __int64 v79; // [rsp+18h] [rbp-188h]
  __int64 v80; // [rsp+18h] [rbp-188h]
  __int64 v81; // [rsp+20h] [rbp-180h]
  __int64 v82; // [rsp+28h] [rbp-178h]
  unsigned int v83; // [rsp+28h] [rbp-178h]
  unsigned __int8 *v84; // [rsp+30h] [rbp-170h]
  char v85; // [rsp+38h] [rbp-168h]
  unsigned __int64 v86; // [rsp+48h] [rbp-158h]
  __int128 v87; // [rsp+50h] [rbp-150h]
  unsigned int v88; // [rsp+50h] [rbp-150h]
  __int64 v89; // [rsp+60h] [rbp-140h]
  __int64 v90; // [rsp+60h] [rbp-140h]
  __int64 v91; // [rsp+60h] [rbp-140h]
  __int64 v92; // [rsp+68h] [rbp-138h]
  __int64 v93; // [rsp+B0h] [rbp-F0h] BYREF
  int v94; // [rsp+B8h] [rbp-E8h]
  unsigned int v95; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v96; // [rsp+C8h] [rbp-D8h]
  unsigned __int16 v97; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v98; // [rsp+D8h] [rbp-C8h]
  unsigned __int64 v99; // [rsp+E0h] [rbp-C0h] BYREF
  unsigned int v100; // [rsp+E8h] [rbp-B8h]
  __m128i v101; // [rsp+F0h] [rbp-B0h] BYREF
  __m128i v102; // [rsp+100h] [rbp-A0h] BYREF
  __m128i v103; // [rsp+110h] [rbp-90h]
  __int64 v104; // [rsp+120h] [rbp-80h] BYREF
  __int64 v105; // [rsp+128h] [rbp-78h]
  __m128i *v106; // [rsp+130h] [rbp-70h]
  char *v107; // [rsp+138h] [rbp-68h]
  char *m128i_i8; // [rsp+140h] [rbp-60h]
  unsigned __int64 *v109; // [rsp+148h] [rbp-58h]
  __m128i *v110; // [rsp+150h] [rbp-50h]
  __m128i *v111; // [rsp+158h] [rbp-48h]
  __m128i *v112; // [rsp+160h] [rbp-40h]
  unsigned __int64 *v113; // [rsp+168h] [rbp-38h]

  v9 = *(_QWORD *)(a3 + 80);
  v93 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v93, v9, 1);
  v94 = *(_DWORD *)(a3 + 72);
  v10 = *(_QWORD *)(a3 + 40);
  v11 = _mm_loadu_si128((const __m128i *)(v10 + 40));
  v12 = *(_DWORD *)(v10 + 48);
  v13 = _mm_loadu_si128((const __m128i *)(v10 + 80));
  v92 = *(_QWORD *)v10;
  v84 = *(unsigned __int8 **)(v10 + 40);
  v82 = *(_QWORD *)(v10 + 80);
  v81 = *(unsigned int *)(v10 + 8);
  v14 = *(_QWORD *)(*(_QWORD *)v10 + 48LL) + 16 * v81;
  v15 = *(_WORD *)v14;
  v86 = v13.m128i_u64[1];
  v96 = *(_QWORD *)(v14 + 8);
  LOWORD(v95) = v15;
  v16 = (unsigned __int16 *)(*((_QWORD *)v84 + 6) + 16LL * v12);
  v89 = v12;
  v17 = *v16;
  v18 = *((_QWORD *)v16 + 1);
  v97 = v17;
  v98 = v18;
  if ( (_WORD)v17 )
  {
    v19 = word_4456340;
    LOBYTE(a6) = (unsigned __int16)(v17 - 176) <= 0x34u;
    v20 = (unsigned int)a6;
    v21 = word_4456340[v17 - 1];
    if ( v15 )
    {
LABEL_5:
      v79 = 0;
      v22 = word_4456580[v15 - 1];
      goto LABEL_6;
    }
  }
  else
  {
    v21 = sub_3007240((__int64)&v97);
    v20 = HIDWORD(v21);
    a6 = HIDWORD(v21);
    if ( v15 )
      goto LABEL_5;
  }
  v78 = a6;
  v85 = v20;
  v22 = sub_3009970((__int64)&v95, v21, (__int64)v19, v20, a6);
  LOBYTE(a6) = v78;
  v79 = v74;
  LOBYTE(v20) = v85;
LABEL_6:
  v23 = (__int64 *)a4[8];
  LODWORD(v104) = v21;
  v24 = v22;
  BYTE4(v104) = v20;
  if ( (_BYTE)a6 )
  {
    LOWORD(v25) = sub_2D43AD0(v22, v21);
    v28 = 0;
    if ( (_WORD)v25 )
      goto LABEL_8;
  }
  else
  {
    LOWORD(v25) = sub_2D43050(v22, v21);
    v28 = 0;
    if ( (_WORD)v25 )
      goto LABEL_8;
  }
  v25 = sub_3009450(v23, v24, v79, v104, 0, v27);
  HIWORD(v6) = HIWORD(v25);
  v28 = v73;
LABEL_8:
  LOWORD(v6) = v25;
  v29 = (*(_DWORD *)(a3 + 24) != 391) + 213;
  if ( v97 != (_WORD)v25 || !(_WORD)v25 && v98 != v28 )
  {
    v91 = v28;
    v68 = sub_33FAF80((__int64)a4, v29, (__int64)&v93, v6, v28, v27, a1);
    v12 = v69;
    v84 = v68;
    v71 = sub_33FAF80((__int64)a4, v29, (__int64)&v93, v6, v91, v70, a1);
    v28 = v91;
    v26 = 0xFFFFFFFF00000000LL;
    v82 = (__int64)v71;
    v86 = v72 | v13.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v89 = v12;
  }
  v80 = v28;
  *(_QWORD *)&v87 = v84;
  v100 = 1;
  v99 = 0;
  *((_QWORD *)&v87 + 1) = v89 | v11.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v30 = sub_33D1410(v82, (__int64)&v99, 0xFFFFFFFF00000000LL, v26, v28);
  v32 = v80;
  if ( !v30 )
    goto LABEL_13;
  if ( v100 > 0x40 )
  {
    v33 = v100 - 1;
    v34 = sub_C444A0((__int64)&v99);
    v32 = v80;
    if ( v34 == v33 )
      goto LABEL_14;
    goto LABEL_13;
  }
  if ( v99 != 1 )
  {
LABEL_13:
    *((_QWORD *)&v75 + 1) = v86;
    *(_QWORD *)&v75 = v82;
    v84 = sub_3406EB0(a4, 0x3Au, (__int64)&v93, v6, v32, v31, v87, v75);
    v12 = v35;
  }
LABEL_14:
  if ( (_WORD)v95 )
    v88 = word_4456340[(unsigned __int16)v95 - 1];
  else
    v88 = sub_3007240((__int64)&v95);
  if ( v97 )
    v36 = word_4456340[v97 - 1];
  else
    v36 = sub_3007240((__int64)&v97);
  v104 = 0;
  v106 = 0;
  v107 = 0;
  m128i_i8 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v105 = 8;
  v83 = v36 / v88;
  v104 = sub_22077B0(0x40u);
  v37 = (unsigned __int64 *)(v104 + ((4 * v105 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v38 = (__m128i *)sub_22077B0(0x200u);
  v109 = v37;
  *v37 = (unsigned __int64)v38;
  m128i_i8 = v38[32].m128i_i8;
  v112 = v38 + 32;
  v107 = (char *)v38;
  v113 = v37;
  v111 = v38;
  v106 = v38;
  v110 = v38 + 1;
  if ( v38 )
  {
    v38->m128i_i64[1] = v81;
    v38->m128i_i64[0] = v92;
  }
  if ( v36 < v88 )
  {
    v46 = v37;
    goto LABEL_44;
  }
  v90 = v12;
  v39 = 0;
  v40 = *((_QWORD *)&v87 + 1);
  v41 = 0;
  do
  {
    v42.m128i_i64[0] = (__int64)sub_3400EE0((__int64)a4, v41, (__int64)&v93, 0, a1);
    ++v39;
    v103 = v42;
    *((_QWORD *)&v76 + 1) = 2;
    v40 = v90 | v40 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v76 = &v102;
    v102.m128i_i64[0] = (__int64)v84;
    v102.m128i_i64[1] = v40;
    v101.m128i_i64[0] = (__int64)sub_33FC220(a4, 161, (__int64)&v93, v95, v96, v43, v76);
    v101.m128i_i64[1] = v44;
    sub_343FDE0((unsigned __int64 *)&v104, &v101);
    v41 += v88;
  }
  while ( v83 > v39 );
  v46 = v113;
  v37 = v109;
  v38 = v106;
  v47 = (unsigned __int64)v107;
  v48 = (char *)v106 - v107;
  v49 = ((char *)v106 - v107) >> 4;
  if ( (unsigned __int64)(((m128i_i8 - (char *)v106) >> 4) + 32 * (v113 - v109 - 1) + v110 - v111) > 1 )
  {
    do
    {
      v58 = (__int64)v38->m128i_i64 - v47;
      v59 = (__int64)((__int64)v38->m128i_i64 - v47) >> 4;
      if ( (__int64)((__int64)v38->m128i_i64 - v47) >= 0 )
      {
        v50 = v38;
        if ( v58 <= 496 )
          goto LABEL_28;
        v51 = v58 >> 9;
      }
      else
      {
        v51 = ~((unsigned __int64)~v59 >> 5);
      }
      v50 = (const __m128i *)(v37[v51] + 16 * (v59 - 32 * v51));
LABEL_28:
      v102 = _mm_loadu_si128(v50);
      v52 = v59 + 1;
      if ( v52 < 0 )
      {
        v60 = ~((unsigned __int64)~v52 >> 5);
        goto LABEL_39;
      }
      if ( v52 > 31 )
      {
        v60 = v52 >> 5;
LABEL_39:
        v53 = (const __m128i *)(v37[v60] + 16 * (v52 - 32 * v60));
        goto LABEL_31;
      }
      v53 = v38 + 1;
LABEL_31:
      *((_QWORD *)&v77 + 1) = 2;
      *(_QWORD *)&v77 = &v102;
      v103 = _mm_loadu_si128(v53);
      v101.m128i_i64[0] = (__int64)sub_33FC220(a4, 56, (__int64)&v93, v95, v96, v45, v77);
      v101.m128i_i64[1] = v54;
      sub_343FDE0((unsigned __int64 *)&v104, &v101);
      v55 = (__int64)m128i_i8;
      if ( v106 == (__m128i *)(m128i_i8 - 16) )
      {
        j_j___libc_free_0((unsigned __int64)v107);
        v56 = *++v109;
        v55 = *v109 + 512;
        v107 = (char *)v56;
        m128i_i8 = (char *)v55;
        v106 = (__m128i *)v56;
      }
      else
      {
        v56 = (unsigned __int64)&v106[1];
        ++v106;
      }
      if ( v56 == v55 - 16 )
      {
        j_j___libc_free_0((unsigned __int64)v107);
        v49 = 0;
        v48 = 0;
        v57 = 32;
        v37 = v109 + 1;
        v109 = v37;
        v38 = (__m128i *)*v37;
        v61 = *v37 + 512;
        v107 = (char *)v38;
        v47 = (unsigned __int64)v38;
        m128i_i8 = (char *)v61;
        v106 = v38;
      }
      else
      {
        v47 = (unsigned __int64)v107;
        v37 = v109;
        v38 = ++v106;
        v48 = (char *)v106 - v107;
        v57 = (m128i_i8 - (char *)v106) >> 4;
        v49 = ((char *)v106 - v107) >> 4;
      }
      v46 = v113;
    }
    while ( (unsigned __int64)(v57 + v110 - v111 + 32 * (v113 - v37 - 1)) > 1 );
  }
  if ( v48 < 0 )
  {
    v67 = ~((unsigned __int64)~v49 >> 5);
LABEL_55:
    v38 = (__m128i *)(v37[v67] + 16 * (v49 - 32 * v67));
    goto LABEL_44;
  }
  if ( v48 > 496 )
  {
    v67 = v49 >> 5;
    goto LABEL_55;
  }
LABEL_44:
  v62 = v104;
  v63 = v38->m128i_i64[0];
  if ( v104 )
  {
    v64 = (unsigned __int64)(v46 + 1);
    if ( v46 + 1 > v37 )
    {
      do
      {
        v65 = *v37++;
        j_j___libc_free_0(v65);
      }
      while ( v64 > (unsigned __int64)v37 );
      v62 = v104;
    }
    j_j___libc_free_0(v62);
  }
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  if ( v93 )
    sub_B91220((__int64)&v93, v93);
  return v63;
}
