// Function: sub_1AD2230
// Address: 0x1ad2230
//
void *__fastcall sub_1AD2230(__int64 a1, char a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // r15
  char *v7; // rsi
  void *v8; // rdi
  __int64 v9; // r13
  unsigned __int64 v10; // rax
  __m128i *v11; // rsi
  size_t **v12; // r13
  int v13; // r12d
  int v14; // r15d
  char v15; // r14
  size_t **v16; // rbx
  size_t v17; // rax
  int v18; // edx
  void **v19; // r8
  const char *v20; // r9
  size_t v21; // rax
  __int64 v22; // r8
  size_t v23; // rdx
  _QWORD *v24; // rax
  __int64 v25; // rax
  void *v26; // rdx
  _BYTE *v27; // rdi
  _BYTE *v28; // rax
  size_t v29; // rdx
  char *v30; // rsi
  void *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __m128i *v34; // rdx
  __int64 v35; // rdi
  __m128i v36; // xmm0
  __int64 v37; // rdi
  _BYTE *v38; // rax
  int v39; // r14d
  int v40; // r13d
  int v41; // r14d
  void **v42; // rdi
  _BYTE *v43; // rdx
  __int64 v44; // rax
  __m128i *v45; // rdx
  __int64 v46; // rdi
  __m128i v47; // xmm0
  __int64 v48; // r10
  _BYTE *v49; // rax
  __int64 v50; // r15
  __int64 v51; // r12
  __int64 v52; // r12
  __int64 v53; // r12
  char *v54; // rsi
  __int64 v55; // rdx
  void **v56; // rdi
  __int64 v57; // rax
  void *result; // rax
  __int64 v59; // rax
  unsigned __int64 v60; // rax
  __m128i si128; // xmm0
  _BYTE *v62; // rdx
  unsigned __int64 v63; // rsi
  char *v64; // rax
  char *v65; // r9
  unsigned int v66; // eax
  unsigned int v67; // eax
  unsigned int v68; // edi
  __int64 v69; // rcx
  __int64 v70; // rax
  __int64 v71; // [rsp+0h] [rbp-190h]
  __int64 v72; // [rsp+10h] [rbp-180h]
  int v73; // [rsp+10h] [rbp-180h]
  __int64 v74; // [rsp+10h] [rbp-180h]
  char *v76; // [rsp+18h] [rbp-178h]
  __int64 v77; // [rsp+18h] [rbp-178h]
  __int64 v78; // [rsp+18h] [rbp-178h]
  size_t v79; // [rsp+18h] [rbp-178h]
  int v80; // [rsp+28h] [rbp-168h]
  int v81; // [rsp+2Ch] [rbp-164h]
  size_t **v82; // [rsp+30h] [rbp-160h] BYREF
  size_t **v83; // [rsp+38h] [rbp-158h]
  __int64 v84; // [rsp+40h] [rbp-150h]
  char *v85; // [rsp+50h] [rbp-140h] BYREF
  size_t v86; // [rsp+58h] [rbp-138h]
  _QWORD v87[2]; // [rsp+60h] [rbp-130h] BYREF
  char *v88[2]; // [rsp+70h] [rbp-120h] BYREF
  __int64 v89; // [rsp+80h] [rbp-110h] BYREF
  char *v90[2]; // [rsp+90h] [rbp-100h] BYREF
  __int64 v91; // [rsp+A0h] [rbp-F0h] BYREF
  char *v92[2]; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v93; // [rsp+C0h] [rbp-D0h] BYREF
  char *v94[2]; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v95; // [rsp+E0h] [rbp-B0h] BYREF
  char *v96[2]; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v97; // [rsp+100h] [rbp-90h] BYREF
  char *v98[2]; // [rsp+110h] [rbp-80h] BYREF
  __int64 v99; // [rsp+120h] [rbp-70h] BYREF
  void *v100; // [rsp+130h] [rbp-60h] BYREF
  _OWORD *v101; // [rsp+138h] [rbp-58h]
  __int64 v102; // [rsp+140h] [rbp-50h]
  _OWORD *v103; // [rsp+148h] [rbp-48h]
  int v104; // [rsp+150h] [rbp-40h]
  char **v105; // [rsp+158h] [rbp-38h]

  v3 = a1;
  sub_1AD1900(a1);
  v4 = *(_QWORD *)(a1 + 32);
  if ( *(_QWORD *)(a1 + 40) != v4 )
    *(_QWORD *)(a1 + 40) = v4;
  sub_1AD1F10((__int64)&v82, (__int64 **)a1);
  v85 = (char *)v87;
  v86 = 0;
  LOBYTE(v87[0]) = 0;
  sub_2240E30(&v85, 5000);
  v105 = &v85;
  v104 = 1;
  v100 = &unk_49EFBE0;
  v103 = 0;
  v102 = 0;
  v101 = 0;
  v5 = sub_16E7EE0((__int64)&v100, "------- Dumping inliner stats for [", 0x23u);
  v6 = *(_QWORD *)(a1 + 72);
  v7 = *(char **)(a1 + 64);
  v8 = *(void **)(v5 + 24);
  v9 = v5;
  v10 = *(_QWORD *)(v5 + 16) - (_QWORD)v8;
  if ( v6 > v10 )
  {
    v70 = sub_16E7EE0(v9, v7, *(_QWORD *)(v3 + 72));
    v8 = *(void **)(v70 + 24);
    v9 = v70;
    v10 = *(_QWORD *)(v70 + 16) - (_QWORD)v8;
LABEL_5:
    if ( v10 > 9 )
      goto LABEL_6;
LABEL_68:
    sub_16E7EE0(v9, "] -------\n", 0xAu);
    v11 = (__m128i *)v103;
    if ( !a2 )
      goto LABEL_7;
    goto LABEL_69;
  }
  if ( !v6 )
    goto LABEL_5;
  memcpy(v8, v7, *(_QWORD *)(v3 + 72));
  v8 = (void *)(v6 + *(_QWORD *)(v9 + 24));
  v60 = *(_QWORD *)(v9 + 16) - (_QWORD)v8;
  *(_QWORD *)(v9 + 24) = v8;
  if ( v60 <= 9 )
    goto LABEL_68;
LABEL_6:
  qmemcpy(v8, "] -------\n", 10);
  *(_QWORD *)(v9 + 24) += 10LL;
  v11 = (__m128i *)v103;
  if ( !a2 )
    goto LABEL_7;
LABEL_69:
  if ( (unsigned __int64)(v102 - (_QWORD)v11) <= 0x1D )
  {
    sub_16E7EE0((__int64)&v100, "-- List of inlined functions:\n", 0x1Eu);
    v11 = (__m128i *)v103;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42C6010);
    qmemcpy(&v11[1], "ed functions:\n", 14);
    *v11 = si128;
    v11 = (__m128i *)((char *)v103 + 30);
    v103 = (_OWORD *)((char *)v103 + 30);
  }
LABEL_7:
  v12 = v83;
  if ( v83 != v82 )
  {
    v71 = v3;
    v13 = 0;
    v14 = 0;
    v15 = a2;
    v80 = 0;
    v16 = v82;
    v81 = 0;
    while ( 1 )
    {
      v17 = (*v16)[1];
      if ( !*(_DWORD *)(v17 + 80) )
        goto LABEL_10;
      v18 = *(_DWORD *)(v17 + 84) > 0;
      if ( *(_BYTE *)(v17 + 88) )
      {
        ++v14;
        v13 += v18;
        if ( v15 )
          goto LABEL_14;
LABEL_10:
        if ( v12 == ++v16 )
          goto LABEL_33;
      }
      else
      {
        ++v81;
        v80 += v18;
        if ( !v15 )
          goto LABEL_10;
LABEL_14:
        if ( (unsigned __int64)(v102 - (_QWORD)v11) <= 7 )
        {
          v19 = (void **)sub_16E7EE0((__int64)&v100, "Inlined ", 8u);
        }
        else
        {
          v19 = &v100;
          v11->m128i_i64[0] = 0x2064656E696C6E49LL;
          v103 = (_OWORD *)((char *)v103 + 8);
        }
        v20 = "not imported ";
        v72 = (__int64)v19;
        if ( *(_BYTE *)((*v16)[1] + 88) )
          v20 = "imported ";
        v76 = (char *)v20;
        v21 = strlen(v20);
        v22 = v72;
        v23 = v21;
        v24 = *(_QWORD **)(v72 + 24);
        if ( v23 <= *(_QWORD *)(v72 + 16) - (_QWORD)v24 )
        {
          *v24 = *(_QWORD *)v76;
          *(_QWORD *)((char *)v24 + (unsigned int)v23 - 8) = *(_QWORD *)&v76[(unsigned int)v23 - 8];
          v63 = (unsigned __int64)(v24 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          v64 = (char *)v24 - v63;
          v65 = (char *)(v76 - v64);
          v66 = (v23 + (_DWORD)v64) & 0xFFFFFFF8;
          if ( v66 >= 8 )
          {
            v67 = v66 & 0xFFFFFFF8;
            v68 = 0;
            do
            {
              v69 = v68;
              v68 += 8;
              *(_QWORD *)(v63 + v69) = *(_QWORD *)&v65[v69];
            }
            while ( v68 < v67 );
          }
          v26 = (void *)(*(_QWORD *)(v72 + 24) + v23);
          *(_QWORD *)(v72 + 24) = v26;
        }
        else
        {
          v25 = sub_16E7EE0(v72, v76, v23);
          v26 = *(void **)(v25 + 24);
          v22 = v25;
        }
        if ( *(_QWORD *)(v22 + 16) - (_QWORD)v26 <= 9u )
        {
          v59 = sub_16E7EE0(v22, "function [", 0xAu);
          v27 = *(_BYTE **)(v59 + 24);
          v22 = v59;
        }
        else
        {
          qmemcpy(v26, "function [", 10);
          v27 = (_BYTE *)(*(_QWORD *)(v22 + 24) + 10LL);
          *(_QWORD *)(v22 + 24) = v27;
        }
        v28 = *(_BYTE **)(v22 + 16);
        v29 = **v16;
        v30 = (char *)(*v16 + 2);
        if ( v29 > v28 - v27 )
        {
          v22 = sub_16E7EE0(v22, v30, v29);
          v27 = *(_BYTE **)(v22 + 24);
          if ( v27 != *(_BYTE **)(v22 + 16) )
            goto LABEL_26;
        }
        else
        {
          if ( v29 )
          {
            v74 = v22;
            v79 = **v16;
            memcpy(v27, v30, v29);
            v22 = v74;
            v62 = (_BYTE *)(*(_QWORD *)(v74 + 24) + v79);
            v28 = *(_BYTE **)(v74 + 16);
            *(_QWORD *)(v74 + 24) = v62;
            v27 = v62;
          }
          if ( v27 != v28 )
          {
LABEL_26:
            *v27 = 93;
            v31 = (void *)(*(_QWORD *)(v22 + 24) + 1LL);
            v32 = *(_QWORD *)(v22 + 16);
            *(_QWORD *)(v22 + 24) = v31;
            if ( (unsigned __int64)(v32 - (_QWORD)v31) <= 0xC )
              goto LABEL_63;
            goto LABEL_27;
          }
        }
        v22 = sub_16E7EE0(v22, "]", 1u);
        v31 = *(void **)(v22 + 24);
        if ( *(_QWORD *)(v22 + 16) - (_QWORD)v31 <= 0xCu )
        {
LABEL_63:
          v22 = sub_16E7EE0(v22, ": #inlines = ", 0xDu);
          goto LABEL_28;
        }
LABEL_27:
        qmemcpy(v31, ": #inlines = ", 13);
        *(_QWORD *)(v22 + 24) += 13LL;
LABEL_28:
        v33 = sub_16E7AB0(v22, *(int *)((*v16)[1] + 80));
        v34 = *(__m128i **)(v33 + 24);
        v35 = v33;
        if ( *(_QWORD *)(v33 + 16) - (_QWORD)v34 <= 0x20u )
        {
          v35 = sub_16E7EE0(v33, ", #inlines_to_importing_module = ", 0x21u);
        }
        else
        {
          v36 = _mm_load_si128((const __m128i *)&xmmword_42C6020);
          v34[2].m128i_i8[0] = 32;
          *v34 = v36;
          v34[1] = _mm_load_si128((const __m128i *)&xmmword_42C6030);
          *(_QWORD *)(v33 + 24) += 33LL;
        }
        v37 = sub_16E7AB0(v35, *(int *)((*v16)[1] + 84));
        v38 = *(_BYTE **)(v37 + 24);
        if ( *(_BYTE **)(v37 + 16) == v38 )
        {
          sub_16E7EE0(v37, "\n", 1u);
        }
        else
        {
          *v38 = 10;
          ++*(_QWORD *)(v37 + 24);
        }
        v11 = (__m128i *)v103;
        if ( v12 == ++v16 )
        {
LABEL_33:
          v3 = v71;
          v73 = v14 + v81;
          goto LABEL_34;
        }
      }
    }
  }
  v73 = 0;
  v13 = 0;
  v14 = 0;
  v80 = 0;
  v81 = 0;
LABEL_34:
  v39 = *(_DWORD *)(v3 + 60);
  v40 = *(_DWORD *)(v3 + 56) - v39;
  v41 = v39 - v13;
  if ( (unsigned __int64)(v102 - (_QWORD)v11) <= 0xB )
  {
    v42 = (void **)sub_16E7EE0((__int64)&v100, "-- Summary:\n", 0xCu);
    v43 = v42[3];
    if ( (unsigned __int64)((_BYTE *)v42[2] - v43) > 0xE )
      goto LABEL_36;
  }
  else
  {
    v42 = &v100;
    qmemcpy(v11, "-- Summary:\n", 12);
    v43 = (char *)v103 + 12;
    v103 = v43;
    if ( (unsigned __int64)(v102 - (_QWORD)v43) > 0xE )
    {
LABEL_36:
      qmemcpy(v43, "All functions: ", 15);
      v42[3] = (char *)v42[3] + 15;
      goto LABEL_37;
    }
  }
  v42 = (void **)sub_16E7EE0((__int64)v42, "All functions: ", 0xFu);
LABEL_37:
  v44 = sub_16E7AB0((__int64)v42, *(int *)(v3 + 56));
  v45 = *(__m128i **)(v44 + 24);
  v46 = v44;
  if ( *(_QWORD *)(v44 + 16) - (_QWORD)v45 <= 0x15u )
  {
    v46 = sub_16E7EE0(v44, ", imported functions: ", 0x16u);
  }
  else
  {
    v47 = _mm_load_si128((const __m128i *)&xmmword_42C6040);
    v45[1].m128i_i32[0] = 1936617321;
    v45[1].m128i_i16[2] = 8250;
    *v45 = v47;
    *(_QWORD *)(v44 + 24) += 22LL;
  }
  v48 = sub_16E7AB0(v46, *(int *)(v3 + 60));
  v49 = *(_BYTE **)(v48 + 24);
  if ( *(_BYTE **)(v48 + 16) == v49 )
  {
    v48 = sub_16E7EE0(v48, "\n", 1u);
  }
  else
  {
    *v49 = 10;
    ++*(_QWORD *)(v48 + 24);
  }
  v77 = v48;
  sub_1ACF670((__int64)v88, "inlined functions", v73, *(_DWORD *)(v3 + 56), "all functions", 1);
  v78 = sub_16E7EE0(v77, v88[0], (size_t)v88[1]);
  sub_1ACF670((__int64)v90, "imported functions inlined anywhere", v14, *(_DWORD *)(v3 + 60), "imported functions", 1);
  v50 = sub_16E7EE0(v78, v90[0], (size_t)v90[1]);
  sub_1ACF670(
    (__int64)v92,
    "imported functions inlined into importing module",
    v13,
    *(_DWORD *)(v3 + 60),
    "imported functions",
    0);
  v51 = sub_16E7EE0(v50, v92[0], (size_t)v92[1]);
  sub_1ACF670((__int64)v94, ", remaining", v41, *(_DWORD *)(v3 + 60), "imported functions", 1);
  v52 = sub_16E7EE0(v51, v94[0], (size_t)v94[1]);
  sub_1ACF670((__int64)v96, "non-imported functions inlined anywhere", v81, v40, "non-imported functions", 1);
  v53 = sub_16E7EE0(v52, v96[0], (size_t)v96[1]);
  sub_1ACF670(
    (__int64)v98,
    "non-imported functions inlined into importing module",
    v80,
    v40,
    "non-imported functions",
    1);
  v54 = v98[0];
  sub_16E7EE0(v53, v98[0], (size_t)v98[1]);
  if ( (__int64 *)v98[0] != &v99 )
  {
    v54 = (char *)(v99 + 1);
    j_j___libc_free_0(v98[0], v99 + 1);
  }
  if ( (__int64 *)v96[0] != &v97 )
  {
    v54 = (char *)(v97 + 1);
    j_j___libc_free_0(v96[0], v97 + 1);
  }
  if ( (__int64 *)v94[0] != &v95 )
  {
    v54 = (char *)(v95 + 1);
    j_j___libc_free_0(v94[0], v95 + 1);
  }
  if ( (__int64 *)v92[0] != &v93 )
  {
    v54 = (char *)(v93 + 1);
    j_j___libc_free_0(v92[0], v93 + 1);
  }
  if ( (__int64 *)v90[0] != &v91 )
  {
    v54 = (char *)(v91 + 1);
    j_j___libc_free_0(v90[0], v91 + 1);
  }
  v56 = (void **)v88[0];
  if ( (__int64 *)v88[0] != &v89 )
  {
    v54 = (char *)(v89 + 1);
    j_j___libc_free_0(v88[0], v89 + 1);
  }
  if ( v103 != v101 )
  {
    v56 = &v100;
    sub_16E7BA0((__int64 *)&v100);
  }
  v57 = sub_16BA580((__int64)v56, (__int64)v54, v55);
  sub_16E7EE0(v57, v85, v86);
  result = sub_16E7BC0((__int64 *)&v100);
  if ( v85 != (char *)v87 )
    result = (void *)j_j___libc_free_0(v85, v87[0] + 1LL);
  if ( v82 )
    return (void *)j_j___libc_free_0(v82, v84 - (_QWORD)v82);
  return result;
}
