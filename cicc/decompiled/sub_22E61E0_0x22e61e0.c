// Function: sub_22E61E0
// Address: 0x22e61e0
//
__int64 __fastcall sub_22E61E0(__int64 *a1, __int64 a2, int a3)
{
  __int64 v4; // rdi
  __m128i *v5; // rax
  __m128i si128; // xmm0
  __int64 v7; // rdi
  __int64 v8; // rax
  unsigned int v9; // ebx
  __int64 v10; // rdi
  void *v11; // rax
  __int64 v12; // rax
  __m128i *v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // r12
  unsigned __int64 v17; // rsi
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 *v20; // r12
  __int64 *i; // r13
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // rax
  unsigned __int64 *v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int8 *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  unsigned __int64 v33; // rax
  unsigned __int64 *v34; // rsi
  __int64 *v35; // rdi
  __int64 v36; // r8
  __int64 v37; // r9
  const __m128i *v38; // rcx
  const __m128i *v39; // rdx
  unsigned __int64 v40; // rbx
  __m128i *v41; // rax
  __int64 v42; // rcx
  const __m128i *v43; // rax
  const __m128i *v44; // rcx
  unsigned __int64 v45; // rbx
  __int64 v46; // rax
  unsigned __int64 v47; // rdi
  __m128i *v48; // rdx
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rcx
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // r14
  unsigned __int64 v53; // r14
  __int64 v54; // rbx
  __int64 v55; // r12
  unsigned __int64 v56; // rax
  int v57; // edx
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rax
  int v60; // eax
  unsigned int v61; // esi
  __int64 v62; // rdi
  unsigned __int64 *v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // r15
  unsigned __int64 *v68; // rax
  unsigned __int64 v69; // rax
  char v70; // si
  __int64 v71; // rax
  _WORD *v72; // rdx
  char v74; // dl
  __int64 v75; // rax
  _DWORD *v76; // rdx
  __int64 v77; // r12
  _QWORD *v78; // rax
  __int64 v79; // rax
  _WORD *v80; // rdx
  __int64 v81; // rdi
  void *v82; // rax
  __int64 v83; // rax
  _QWORD *v84; // rdx
  __int64 v85; // [rsp+0h] [rbp-270h]
  unsigned int v86; // [rsp+10h] [rbp-260h]
  unsigned int v87; // [rsp+14h] [rbp-25Ch]
  __int64 v88; // [rsp+18h] [rbp-258h]
  __m128i v91; // [rsp+30h] [rbp-240h] BYREF
  char v92; // [rsp+48h] [rbp-228h]
  __int64 v93; // [rsp+50h] [rbp-220h] BYREF
  unsigned __int64 *v94; // [rsp+58h] [rbp-218h]
  __int64 v95; // [rsp+60h] [rbp-210h]
  int v96; // [rsp+68h] [rbp-208h]
  char v97; // [rsp+6Ch] [rbp-204h]
  unsigned __int64 v98; // [rsp+70h] [rbp-200h] BYREF
  _QWORD v99[7]; // [rsp+78h] [rbp-1F8h] BYREF
  __m128i *v100; // [rsp+B0h] [rbp-1C0h] BYREF
  unsigned __int64 v101; // [rsp+B8h] [rbp-1B8h]
  __int8 *v102; // [rsp+C0h] [rbp-1B0h]
  unsigned __int64 v103[16]; // [rsp+D0h] [rbp-1A0h] BYREF
  __m128i v104; // [rsp+150h] [rbp-120h] BYREF
  char v105; // [rsp+168h] [rbp-108h]
  char v106; // [rsp+16Ch] [rbp-104h]
  char v107[64]; // [rsp+170h] [rbp-100h] BYREF
  const __m128i *v108; // [rsp+1B0h] [rbp-C0h]
  const __m128i *v109; // [rsp+1B8h] [rbp-B8h]
  __int8 *v110; // [rsp+1C0h] [rbp-B0h]
  char v111[8]; // [rsp+1C8h] [rbp-A8h] BYREF
  unsigned __int64 v112; // [rsp+1D0h] [rbp-A0h]
  char v113; // [rsp+1E4h] [rbp-8Ch]
  char v114[64]; // [rsp+1E8h] [rbp-88h] BYREF
  const __m128i *v115; // [rsp+228h] [rbp-48h]
  unsigned __int64 v116; // [rsp+230h] [rbp-40h]
  unsigned __int64 v117; // [rsp+238h] [rbp-38h]

  v86 = 2 * a3;
  v4 = sub_CB69B0(a2, 2 * a3);
  v5 = *(__m128i **)(v4 + 32);
  if ( *(_QWORD *)(v4 + 24) - (_QWORD)v5 <= 0x10u )
  {
    v4 = sub_CB6200(v4, (unsigned __int8 *)"subgraph cluster_", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_428CDF0);
    v5[1].m128i_i8[0] = 95;
    *v5 = si128;
    *(_QWORD *)(v4 + 32) += 17LL;
  }
  v7 = sub_CB5A80(v4, (unsigned __int64)a1);
  v8 = *(_QWORD *)(v7 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v7 + 24) - v8) <= 2 )
  {
    sub_CB6200(v7, (unsigned __int8 *)" {\n", 3u);
  }
  else
  {
    *(_BYTE *)(v8 + 2) = 10;
    *(_WORD *)v8 = 31520;
    *(_QWORD *)(v7 + 32) += 3LL;
  }
  v9 = a3 + 1;
  v87 = v86 + 2;
  v10 = sub_CB69B0(a2, v86 + 2);
  v11 = *(void **)(v10 + 32);
  if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 0xBu )
  {
    sub_CB6200(v10, "label = \"\";\n", 0xCu);
  }
  else
  {
    qmemcpy(v11, "label = \"\";\n", 12);
    *(_QWORD *)(v10 + 32) += 12LL;
  }
  if ( (_BYTE)qword_4FDC108 && !sub_22DB7F0(a1) )
  {
    v81 = sub_CB69B0(a2, v87);
    v82 = *(void **)(v81 + 32);
    if ( *(_QWORD *)(v81 + 24) - (_QWORD)v82 <= 0xEu )
    {
      sub_CB6200(v81, "style = solid;\n", 0xFu);
    }
    else
    {
      qmemcpy(v82, "style = solid;\n", 15);
      *(_QWORD *)(v81 + 32) += 15LL;
    }
    v83 = sub_CB69B0(a2, v87);
    v84 = *(_QWORD **)(v83 + 32);
    v16 = v83;
    if ( *(_QWORD *)(v83 + 24) - (_QWORD)v84 <= 7u )
    {
      v16 = sub_CB6200(v83, "color = ", 8u);
    }
    else
    {
      *v84 = 0x203D20726F6C6F63LL;
      *(_QWORD *)(v83 + 32) += 8LL;
    }
    v17 = 2 * (unsigned int)sub_22DADF0((__int64)a1) % 0xC + 2;
  }
  else
  {
    v12 = sub_CB69B0(a2, v87);
    v13 = *(__m128i **)(v12 + 32);
    if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 0xFu )
    {
      sub_CB6200(v12, (unsigned __int8 *)"style = filled;\n", 0x10u);
    }
    else
    {
      *v13 = _mm_load_si128((const __m128i *)&xmmword_428CE00);
      *(_QWORD *)(v12 + 32) += 16LL;
    }
    v14 = sub_CB69B0(a2, v87);
    v15 = *(_QWORD **)(v14 + 32);
    v16 = v14;
    if ( *(_QWORD *)(v14 + 24) - (_QWORD)v15 <= 7u )
    {
      v16 = sub_CB6200(v14, "color = ", 8u);
    }
    else
    {
      *v15 = 0x203D20726F6C6F63LL;
      *(_QWORD *)(v14 + 32) += 8LL;
    }
    v17 = 2 * (unsigned int)sub_22DADF0((__int64)a1) % 0xC + 1;
  }
  v18 = sub_CB59D0(v16, v17);
  v19 = *(_BYTE **)(v18 + 32);
  if ( *(_BYTE **)(v18 + 24) == v19 )
  {
    sub_CB6200(v18, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v19 = 10;
    ++*(_QWORD *)(v18 + 32);
  }
  v20 = (__int64 *)a1[6];
  for ( i = (__int64 *)a1[5]; v20 != i; ++i )
  {
    v22 = *i;
    sub_22E61E0(v22, a2, v9);
  }
  v100 = 0;
  v23 = a1[2];
  v24 = a1[4];
  v101 = 0;
  v102 = 0;
  v88 = v23;
  memset(v103, 0, 0x78u);
  v85 = v24;
  v103[1] = (unsigned __int64)&v103[4];
  v25 = *a1;
  LODWORD(v103[2]) = 8;
  BYTE4(v103[3]) = 1;
  v98 = v25 & 0xFFFFFFFFFFFFFFF8LL;
  v104.m128i_i64[0] = v25 & 0xFFFFFFFFFFFFFFF8LL;
  v94 = &v98;
  v95 = 0x100000008LL;
  v96 = 0;
  v97 = 1;
  v93 = 1;
  v105 = 0;
  sub_22E61A0((__int64)&v100, &v104);
  v26 = &v98;
  if ( &v98 == v99 )
  {
LABEL_115:
    ++HIDWORD(v95);
    v99[0] = v85;
    ++v93;
  }
  else
  {
    while ( v85 != *v26 )
    {
      if ( v99 == ++v26 )
        goto LABEL_115;
    }
  }
  sub_C8CF70((__int64)&v104, v107, 8, (__int64)&v98, (__int64)&v93);
  v27 = (unsigned __int64)v100;
  v100 = 0;
  v108 = (const __m128i *)v27;
  v28 = v101;
  v101 = 0;
  v109 = (const __m128i *)v28;
  v29 = v102;
  v102 = 0;
  v110 = v29;
  sub_C8CF70((__int64)v111, v114, 8, (__int64)&v103[4], (__int64)v103);
  v33 = v103[12];
  memset(&v103[12], 0, 24);
  v115 = (const __m128i *)v33;
  v116 = v103[13];
  v117 = v103[14];
  if ( v100 )
    j_j___libc_free_0((unsigned __int64)v100);
  if ( !v97 )
    _libc_free((unsigned __int64)v94);
  if ( v103[12] )
    j_j___libc_free_0(v103[12]);
  if ( !BYTE4(v103[3]) )
    _libc_free(v103[1]);
  v34 = &v98;
  v35 = &v93;
  sub_C8CD80((__int64)&v93, (__int64)&v98, (__int64)&v104, v30, v31, v32);
  v38 = v109;
  v39 = v108;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v40 = (char *)v109 - (char *)v108;
  if ( v109 == v108 )
  {
    v41 = 0;
  }
  else
  {
    if ( v40 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_126;
    v41 = (__m128i *)sub_22077B0((char *)v109 - (char *)v108);
    v38 = v109;
    v39 = v108;
  }
  v100 = v41;
  v101 = (unsigned __int64)v41;
  v102 = &v41->m128i_i8[v40];
  if ( v39 == v38 )
  {
    v42 = (__int64)v41;
  }
  else
  {
    v42 = (__int64)v41->m128i_i64 + (char *)v38 - (char *)v39;
    do
    {
      if ( v41 )
      {
        *v41 = _mm_loadu_si128(v39);
        v41[1] = _mm_loadu_si128(v39 + 1);
      }
      v41 += 2;
      v39 += 2;
    }
    while ( (__m128i *)v42 != v41 );
  }
  v34 = &v103[4];
  v35 = (__int64 *)v103;
  v101 = v42;
  sub_C8CD80((__int64)v103, (__int64)&v103[4], (__int64)v111, v42, v36, v37);
  v43 = (const __m128i *)v116;
  v44 = v115;
  memset(&v103[12], 0, 24);
  v45 = v116 - (_QWORD)v115;
  if ( (const __m128i *)v116 != v115 )
  {
    if ( v45 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v46 = sub_22077B0(v116 - (_QWORD)v115);
      v44 = v115;
      v47 = v46;
      v43 = (const __m128i *)v116;
      goto LABEL_43;
    }
LABEL_126:
    sub_4261EA(v35, v34, v39);
  }
  v47 = 0;
LABEL_43:
  v103[12] = v47;
  v103[13] = v47;
  v103[14] = v47 + v45;
  if ( v43 == v44 )
  {
    v49 = v47;
  }
  else
  {
    v48 = (__m128i *)v47;
    v49 = v47 + (char *)v43 - (char *)v44;
    do
    {
      if ( v48 )
      {
        *v48 = _mm_loadu_si128(v44);
        v48[1] = _mm_loadu_si128(v44 + 1);
      }
      v48 += 2;
      v44 += 2;
    }
    while ( (__m128i *)v49 != v48 );
  }
  v50 = v101;
  v51 = (unsigned __int64)v100;
  v103[13] = v49;
  if ( v101 - (_QWORD)v100 != v49 - v47 )
    goto LABEL_49;
  while ( v50 != v51 )
  {
    v69 = v47;
    while ( *(_QWORD *)v51 == *(_QWORD *)v69 )
    {
      v70 = *(_BYTE *)(v51 + 24);
      if ( v70 != *(_BYTE *)(v69 + 24) || v70 && *(_DWORD *)(v51 + 16) != *(_DWORD *)(v69 + 16) )
        break;
      v51 += 32LL;
      v69 += 32LL;
      if ( v50 == v51 )
        goto LABEL_76;
    }
    do
    {
LABEL_49:
      v52 = *(_QWORD *)(v50 - 32);
      if ( a1 == (__int64 *)sub_22DBE80(v88, v52) )
      {
        v75 = sub_CB69B0(a2, v87);
        v76 = *(_DWORD **)(v75 + 32);
        v77 = v75;
        if ( *(_QWORD *)(v75 + 24) - (_QWORD)v76 <= 3u )
        {
          v77 = sub_CB6200(v75, (unsigned __int8 *)"Node", 4u);
        }
        else
        {
          *v76 = 1701080910;
          *(_QWORD *)(v75 + 32) += 4LL;
        }
        v78 = sub_22DDF00(*(_QWORD **)(v88 + 32), v52);
        v79 = sub_CB5A80(v77, (unsigned __int64)v78);
        v80 = *(_WORD **)(v79 + 32);
        if ( *(_QWORD *)(v79 + 24) - (_QWORD)v80 <= 1u )
        {
          sub_CB6200(v79, (unsigned __int8 *)";\n", 2u);
        }
        else
        {
          *v80 = 2619;
          *(_QWORD *)(v79 + 32) += 2LL;
        }
      }
      v53 = v101;
      do
      {
        v54 = *(_QWORD *)(v53 - 32);
        v55 = v54 + 48;
        if ( !*(_BYTE *)(v53 - 8) )
        {
          v56 = *(_QWORD *)(v54 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v56 == v55 )
          {
            v58 = 0;
          }
          else
          {
            if ( !v56 )
              BUG();
            v57 = *(unsigned __int8 *)(v56 - 24);
            v58 = v56 - 24;
            if ( (unsigned int)(v57 - 30) >= 0xB )
              v58 = 0;
          }
          *(_QWORD *)(v53 - 24) = v58;
          *(_DWORD *)(v53 - 16) = 0;
          *(_BYTE *)(v53 - 8) = 1;
        }
LABEL_57:
        v59 = *(_QWORD *)(v54 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v55 == v59 )
          goto LABEL_96;
LABEL_58:
        if ( !v59 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v59 - 24) - 30 <= 0xA )
        {
          v60 = sub_B46E30(v59 - 24);
          v61 = *(_DWORD *)(v53 - 16);
          if ( v61 == v60 )
            goto LABEL_97;
          goto LABEL_61;
        }
LABEL_96:
        while ( 1 )
        {
          v61 = *(_DWORD *)(v53 - 16);
          if ( !v61 )
            break;
LABEL_61:
          v62 = *(_QWORD *)(v53 - 24);
          *(_DWORD *)(v53 - 16) = v61 + 1;
          v67 = sub_B46EC0(v62, v61);
          if ( v97 )
          {
            v68 = v94;
            v63 = &v94[HIDWORD(v95)];
            if ( v94 != v63 )
            {
              while ( v67 != *v68 )
              {
                if ( v63 == ++v68 )
                  goto LABEL_65;
              }
              goto LABEL_57;
            }
LABEL_65:
            if ( HIDWORD(v95) < (unsigned int)v95 )
            {
              ++HIDWORD(v95);
              *v63 = v67;
              ++v93;
LABEL_67:
              v91.m128i_i64[0] = v67;
              v92 = 0;
              sub_22E61A0((__int64)&v100, &v91);
              v51 = (unsigned __int64)v100;
              v50 = v101;
              goto LABEL_68;
            }
          }
          sub_C8CC70((__int64)&v93, v67, (__int64)v63, v64, v65, v66);
          if ( v74 )
            goto LABEL_67;
          v59 = *(_QWORD *)(v54 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v55 != v59 )
            goto LABEL_58;
        }
LABEL_97:
        v101 -= 32LL;
        v51 = (unsigned __int64)v100;
        v53 = v101;
      }
      while ( (__m128i *)v101 != v100 );
      v50 = (unsigned __int64)v100;
LABEL_68:
      v47 = v103[12];
    }
    while ( v50 - v51 != v103[13] - v103[12] );
  }
LABEL_76:
  if ( v47 )
    j_j___libc_free_0(v47);
  if ( !BYTE4(v103[3]) )
    _libc_free(v103[1]);
  if ( v100 )
    j_j___libc_free_0((unsigned __int64)v100);
  if ( !v97 )
    _libc_free((unsigned __int64)v94);
  if ( v115 )
    j_j___libc_free_0((unsigned __int64)v115);
  if ( !v113 )
    _libc_free(v112);
  if ( v108 )
    j_j___libc_free_0((unsigned __int64)v108);
  if ( !v106 )
    _libc_free(v104.m128i_u64[1]);
  v71 = sub_CB69B0(a2, v86);
  v72 = *(_WORD **)(v71 + 32);
  if ( *(_QWORD *)(v71 + 24) - (_QWORD)v72 <= 1u )
    return sub_CB6200(v71, "}\n", 2u);
  *v72 = 2685;
  *(_QWORD *)(v71 + 32) += 2LL;
  return 2685;
}
