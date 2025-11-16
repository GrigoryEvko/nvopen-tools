// Function: sub_144DDD0
// Address: 0x144ddd0
//
__int64 __fastcall sub_144DDD0(__int64 *a1, __int64 a2, int a3)
{
  __int64 v5; // rdi
  __m128i *v6; // rax
  __m128i si128; // xmm0
  __int64 v8; // rdi
  __int64 v9; // rax
  unsigned int v10; // ebx
  __int64 v11; // rdi
  void *v12; // rax
  __int64 v13; // rax
  __m128i *v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // r12
  __int64 v18; // rsi
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 *v21; // r12
  __int64 *i; // r13
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rsi
  __int64 *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rdx
  unsigned __int64 v36; // rbx
  __int64 v37; // rax
  __int64 v38; // rcx
  char v39; // si
  __int64 v40; // rax
  __int64 v41; // rcx
  unsigned __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rdx
  __int64 v46; // rax
  char v47; // si
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // r14
  unsigned __int64 v51; // r15
  __int64 v52; // rbx
  __int64 v53; // r12
  __int64 v54; // rax
  __int64 v55; // rdi
  int v56; // eax
  __int64 v57; // rsi
  __int64 v58; // rdi
  __int64 v59; // r15
  __int64 *v60; // rax
  char v61; // dl
  __int64 v62; // rax
  char v63; // r8
  char v64; // si
  __int64 v65; // rax
  _WORD *v66; // rdx
  __int64 *v68; // rsi
  __int64 *v69; // rdi
  __int64 v70; // rax
  _DWORD *v71; // rdx
  __int64 v72; // r12
  __int64 v73; // rax
  __int64 v74; // rax
  _WORD *v75; // rdx
  __int64 *v76; // rdi
  __int64 *v77; // rdx
  __int64 v78; // rdi
  void *v79; // rax
  __int64 v80; // rax
  _QWORD *v81; // rdx
  __int64 v82; // [rsp+8h] [rbp-278h]
  __int64 v83; // [rsp+8h] [rbp-278h]
  unsigned int v84; // [rsp+20h] [rbp-260h]
  unsigned int v85; // [rsp+24h] [rbp-25Ch]
  __int64 v87[3]; // [rsp+30h] [rbp-250h] BYREF
  char v88; // [rsp+48h] [rbp-238h]
  __int64 v89; // [rsp+50h] [rbp-230h] BYREF
  __int64 *v90; // [rsp+58h] [rbp-228h]
  __int64 *v91; // [rsp+60h] [rbp-220h]
  __int64 v92; // [rsp+68h] [rbp-218h]
  int v93; // [rsp+70h] [rbp-210h]
  _QWORD v94[8]; // [rsp+78h] [rbp-208h] BYREF
  __int64 v95; // [rsp+B8h] [rbp-1C8h] BYREF
  __int64 v96; // [rsp+C0h] [rbp-1C0h]
  unsigned __int64 v97; // [rsp+C8h] [rbp-1B8h]
  _QWORD v98[16]; // [rsp+D0h] [rbp-1B0h] BYREF
  _QWORD v99[2]; // [rsp+150h] [rbp-130h] BYREF
  unsigned __int64 v100; // [rsp+160h] [rbp-120h]
  char v101; // [rsp+168h] [rbp-118h]
  char v102[64]; // [rsp+178h] [rbp-108h] BYREF
  __int64 v103; // [rsp+1B8h] [rbp-C8h]
  __int64 v104; // [rsp+1C0h] [rbp-C0h]
  unsigned __int64 v105; // [rsp+1C8h] [rbp-B8h]
  char v106[8]; // [rsp+1D0h] [rbp-B0h] BYREF
  __int64 v107; // [rsp+1D8h] [rbp-A8h]
  unsigned __int64 v108; // [rsp+1E0h] [rbp-A0h]
  char v109[64]; // [rsp+1F8h] [rbp-88h] BYREF
  __int64 v110; // [rsp+238h] [rbp-48h]
  __int64 v111; // [rsp+240h] [rbp-40h]
  __int64 v112; // [rsp+248h] [rbp-38h]

  v84 = 2 * a3;
  v5 = sub_16E8750(a2, (unsigned int)(2 * a3));
  v6 = *(__m128i **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0x10u )
  {
    v5 = sub_16E7EE0(v5, "subgraph cluster_", 17);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_428CDF0);
    v6[1].m128i_i8[0] = 95;
    *v6 = si128;
    *(_QWORD *)(v5 + 24) += 17LL;
  }
  v8 = sub_16E7B40(v5, a1);
  v9 = *(_QWORD *)(v8 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v8 + 16) - v9) <= 2 )
  {
    sub_16E7EE0(v8, " {\n", 3);
  }
  else
  {
    *(_BYTE *)(v9 + 2) = 10;
    *(_WORD *)v9 = 31520;
    *(_QWORD *)(v8 + 24) += 3LL;
  }
  v10 = a3 + 1;
  v85 = v84 + 2;
  v11 = sub_16E8750(a2, v84 + 2);
  v12 = *(void **)(v11 + 24);
  if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 0xBu )
  {
    sub_16E7EE0(v11, "label = \"\";\n", 12);
  }
  else
  {
    qmemcpy(v12, "label = \"\";\n", 12);
    *(_QWORD *)(v11 + 24) += 12LL;
  }
  if ( byte_4F9A460 && !sub_1443980(a1) )
  {
    v78 = sub_16E8750(a2, v85);
    v79 = *(void **)(v78 + 24);
    if ( *(_QWORD *)(v78 + 16) - (_QWORD)v79 <= 0xEu )
    {
      sub_16E7EE0(v78, "style = solid;\n", 15);
    }
    else
    {
      qmemcpy(v79, "style = solid;\n", 15);
      *(_QWORD *)(v78 + 24) += 15LL;
    }
    v80 = sub_16E8750(a2, v85);
    v81 = *(_QWORD **)(v80 + 24);
    v17 = v80;
    if ( *(_QWORD *)(v80 + 16) - (_QWORD)v81 <= 7u )
    {
      v17 = sub_16E7EE0(v80, "color = ", 8);
    }
    else
    {
      *v81 = 0x203D20726F6C6F63LL;
      *(_QWORD *)(v80 + 24) += 8LL;
    }
    v18 = 2 * (unsigned int)sub_1442F90((__int64)a1) % 0xC + 2;
  }
  else
  {
    v13 = sub_16E8750(a2, v85);
    v14 = *(__m128i **)(v13 + 24);
    if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 0xFu )
    {
      sub_16E7EE0(v13, "style = filled;\n", 16);
    }
    else
    {
      *v14 = _mm_load_si128((const __m128i *)&xmmword_428CE00);
      *(_QWORD *)(v13 + 24) += 16LL;
    }
    v15 = sub_16E8750(a2, v85);
    v16 = *(_QWORD **)(v15 + 24);
    v17 = v15;
    if ( *(_QWORD *)(v15 + 16) - (_QWORD)v16 <= 7u )
    {
      v17 = sub_16E7EE0(v15, "color = ", 8);
    }
    else
    {
      *v16 = 0x203D20726F6C6F63LL;
      *(_QWORD *)(v15 + 24) += 8LL;
    }
    v18 = 2 * (unsigned int)sub_1442F90((__int64)a1) % 0xC + 1;
  }
  v19 = sub_16E7A90(v17, v18);
  v20 = *(_BYTE **)(v19 + 24);
  if ( *(_BYTE **)(v19 + 16) == v20 )
  {
    sub_16E7EE0(v19, "\n", 1);
  }
  else
  {
    *v20 = 10;
    ++*(_QWORD *)(v19 + 24);
  }
  v21 = (__int64 *)a1[6];
  for ( i = (__int64 *)a1[5]; v21 != i; ++i )
  {
    v23 = *i;
    sub_144DDD0(v23, a2, v10);
  }
  v95 = 0;
  v96 = 0;
  v24 = a1[2];
  v25 = a1[4];
  v97 = 0;
  v93 = 0;
  v82 = v24;
  memset(v98, 0, sizeof(v98));
  LODWORD(v98[3]) = 8;
  v98[1] = &v98[5];
  v98[2] = &v98[5];
  v26 = *a1;
  v90 = v94;
  v94[0] = v26 & 0xFFFFFFFFFFFFFFF8LL;
  v99[0] = v26 & 0xFFFFFFFFFFFFFFF8LL;
  v91 = v94;
  v92 = 0x100000008LL;
  v89 = 1;
  v101 = 0;
  sub_1446840(&v95, (__int64)v99);
  v27 = v94;
  v76 = &v90[HIDWORD(v92)];
  if ( v90 == v76 )
  {
LABEL_131:
    ++HIDWORD(v92);
    *v76 = v25;
    ++v89;
  }
  else
  {
    v77 = 0;
    while ( v25 != *v27 )
    {
      if ( *v27 == -2 )
        v77 = v27;
      if ( v76 == ++v27 )
      {
        if ( !v77 )
          goto LABEL_131;
        *v77 = v25;
        --v93;
        ++v89;
        break;
      }
    }
  }
  sub_16CCEE0(v99, v102, 8, &v89);
  v28 = v95;
  v95 = 0;
  v103 = v28;
  v29 = v96;
  v96 = 0;
  v104 = v29;
  v30 = v97;
  v97 = 0;
  v105 = v30;
  sub_16CCEE0(v106, v109, 8, v98);
  v31 = v98[13];
  memset(&v98[13], 0, 24);
  v110 = v31;
  v111 = v98[14];
  v112 = v98[15];
  if ( v95 )
    j_j___libc_free_0(v95, v97 - v95);
  if ( v91 != v90 )
    _libc_free((unsigned __int64)v91);
  if ( v98[13] )
    j_j___libc_free_0(v98[13], v98[15] - v98[13]);
  if ( v98[2] != v98[1] )
    _libc_free(v98[2]);
  v32 = v94;
  v33 = &v89;
  sub_16CCCB0(&v89, v94, v99);
  v34 = v104;
  v35 = v103;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v36 = v104 - v103;
  if ( v104 == v103 )
  {
    v37 = 0;
  }
  else
  {
    if ( v36 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_132;
    v37 = sub_22077B0(v104 - v103);
    v34 = v104;
    v35 = v103;
  }
  v95 = v37;
  v96 = v37;
  v97 = v37 + v36;
  if ( v35 == v34 )
  {
    v38 = v37;
  }
  else
  {
    v38 = v37 + v34 - v35;
    do
    {
      if ( v37 )
      {
        *(_QWORD *)v37 = *(_QWORD *)v35;
        v39 = *(_BYTE *)(v35 + 24);
        *(_BYTE *)(v37 + 24) = v39;
        if ( v39 )
          *(__m128i *)(v37 + 8) = _mm_loadu_si128((const __m128i *)(v35 + 8));
      }
      v37 += 32;
      v35 += 32;
    }
    while ( v38 != v37 );
  }
  v32 = &v98[5];
  v33 = v98;
  v96 = v38;
  sub_16CCCB0(v98, &v98[5], v106);
  v40 = v111;
  v41 = v110;
  memset(&v98[13], 0, 24);
  v42 = v111 - v110;
  if ( v111 != v110 )
  {
    if ( v42 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v43 = sub_22077B0(v111 - v110);
      v41 = v110;
      v44 = v43;
      v40 = v111;
      goto LABEL_39;
    }
LABEL_132:
    sub_4261EA(v33, v32, v35);
  }
  v44 = 0;
LABEL_39:
  v98[13] = v44;
  v98[14] = v44;
  v98[15] = v44 + v42;
  if ( v41 == v40 )
  {
    v46 = v44;
  }
  else
  {
    v45 = v44;
    v46 = v44 + v40 - v41;
    do
    {
      if ( v45 )
      {
        *(_QWORD *)v45 = *(_QWORD *)v41;
        v47 = *(_BYTE *)(v41 + 24);
        *(_BYTE *)(v45 + 24) = v47;
        if ( v47 )
          *(__m128i *)(v45 + 8) = _mm_loadu_si128((const __m128i *)(v41 + 8));
      }
      v45 += 32;
      v41 += 32;
    }
    while ( v46 != v45 );
  }
  v48 = v96;
  v49 = v95;
  v98[14] = v46;
  v50 = v82;
  v83 = a2;
  if ( v96 - v95 == v46 - v44 )
    goto LABEL_57;
  do
  {
LABEL_46:
    v51 = *(_QWORD *)(v48 - 32);
    if ( a1 == (__int64 *)sub_1443F20(v50, v51) )
    {
      v70 = sub_16E8750(v83, v85);
      v71 = *(_DWORD **)(v70 + 24);
      v72 = v70;
      if ( *(_QWORD *)(v70 + 16) - (_QWORD)v71 <= 3u )
      {
        v72 = sub_16E7EE0(v70, "Node", 4);
      }
      else
      {
        *v71 = 1701080910;
        *(_QWORD *)(v70 + 24) += 4LL;
      }
      v73 = sub_1444DB0(*(_QWORD **)(v50 + 32), v51);
      v74 = sub_16E7B40(v72, v73);
      v75 = *(_WORD **)(v74 + 24);
      if ( *(_QWORD *)(v74 + 16) - (_QWORD)v75 <= 1u )
      {
        sub_16E7EE0(v74, ";\n", 2);
      }
      else
      {
        *v75 = 2619;
        *(_QWORD *)(v74 + 24) += 2LL;
      }
    }
    v52 = v96;
    do
    {
      v53 = *(_QWORD *)(v52 - 32);
      if ( !*(_BYTE *)(v52 - 8) )
      {
        v54 = sub_157EBA0(*(_QWORD *)(v52 - 32));
        *(_BYTE *)(v52 - 8) = 1;
        *(_QWORD *)(v52 - 24) = v54;
        *(_DWORD *)(v52 - 16) = 0;
      }
      while ( 1 )
      {
        v55 = sub_157EBA0(v53);
        v56 = 0;
        if ( v55 )
          v56 = sub_15F4D60(v55);
        v57 = *(unsigned int *)(v52 - 16);
        if ( (_DWORD)v57 == v56 )
          break;
        v58 = *(_QWORD *)(v52 - 24);
        *(_DWORD *)(v52 - 16) = v57 + 1;
        v59 = sub_15F4DF0(v58, v57);
        v60 = v90;
        if ( v91 != v90 )
          goto LABEL_54;
        v68 = &v90[HIDWORD(v92)];
        if ( v90 == v68 )
        {
LABEL_91:
          if ( HIDWORD(v92) < (unsigned int)v92 )
          {
            ++HIDWORD(v92);
            *v68 = v59;
            ++v89;
LABEL_55:
            v87[0] = v59;
            v88 = 0;
            sub_1446840(&v95, (__int64)v87);
            v49 = v95;
            v48 = v96;
            goto LABEL_56;
          }
LABEL_54:
          sub_16CCBA0(&v89, v59);
          if ( v61 )
            goto LABEL_55;
        }
        else
        {
          v69 = 0;
          while ( v59 != *v60 )
          {
            if ( *v60 == -2 )
            {
              v69 = v60;
              if ( v68 == v60 + 1 )
                goto LABEL_88;
              ++v60;
            }
            else if ( v68 == ++v60 )
            {
              if ( !v69 )
                goto LABEL_91;
LABEL_88:
              *v69 = v59;
              --v93;
              ++v89;
              goto LABEL_55;
            }
          }
        }
      }
      v96 -= 32;
      v49 = v95;
      v52 = v96;
    }
    while ( v96 != v95 );
    v48 = v95;
LABEL_56:
    v44 = v98[13];
  }
  while ( v48 - v49 != v98[14] - v98[13] );
LABEL_57:
  if ( v49 != v48 )
  {
    v62 = v44;
    while ( *(_QWORD *)v49 == *(_QWORD *)v62 )
    {
      v63 = *(_BYTE *)(v49 + 24);
      v64 = *(_BYTE *)(v62 + 24);
      if ( v63 && v64 )
      {
        if ( *(_DWORD *)(v49 + 16) != *(_DWORD *)(v62 + 16) )
          goto LABEL_46;
        v49 += 32;
        v62 += 32;
        if ( v49 == v48 )
          goto LABEL_64;
      }
      else
      {
        if ( v63 != v64 )
          goto LABEL_46;
        v49 += 32;
        v62 += 32;
        if ( v49 == v48 )
          goto LABEL_64;
      }
    }
    goto LABEL_46;
  }
LABEL_64:
  if ( v44 )
    j_j___libc_free_0(v44, v98[15] - v44);
  if ( v98[2] != v98[1] )
    _libc_free(v98[2]);
  if ( v95 )
    j_j___libc_free_0(v95, v97 - v95);
  if ( v91 != v90 )
    _libc_free((unsigned __int64)v91);
  if ( v110 )
    j_j___libc_free_0(v110, v112 - v110);
  if ( v108 != v107 )
    _libc_free(v108);
  if ( v103 )
    j_j___libc_free_0(v103, v105 - v103);
  if ( v100 != v99[1] )
    _libc_free(v100);
  v65 = sub_16E8750(v83, v84);
  v66 = *(_WORD **)(v65 + 24);
  if ( *(_QWORD *)(v65 + 16) - (_QWORD)v66 <= 1u )
    return sub_16E7EE0(v65, "}\n", 2);
  *v66 = 2685;
  *(_QWORD *)(v65 + 24) += 2LL;
  return 2685;
}
