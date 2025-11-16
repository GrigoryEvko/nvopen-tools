// Function: sub_36FDA80
// Address: 0x36fda80
//
void __fastcall sub_36FDA80(__int64 a1, char a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __m128i *v5; // rdx
  __m128i si128; // xmm0
  __int64 *v7; // r15
  void *v8; // rdi
  unsigned __int64 v9; // r14
  unsigned __int8 *v10; // rsi
  unsigned __int64 v11; // rax
  __m128i *v12; // rsi
  size_t **v13; // r13
  signed int v14; // r15d
  size_t **v15; // rbx
  char v16; // r14
  size_t v17; // rax
  int v18; // edx
  __int64 *v19; // r12
  const char *v20; // r9
  size_t v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // rax
  void *v24; // rdx
  _BYTE *v25; // rdi
  _BYTE *v26; // rax
  size_t v27; // rdx
  unsigned __int8 *v28; // rsi
  void *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __m128i *v32; // rdx
  __int64 v33; // rdi
  __m128i v34; // xmm0
  __int64 v35; // rdi
  _BYTE *v36; // rax
  int v37; // r14d
  int v38; // r13d
  signed int v39; // r14d
  __int64 *v40; // rdi
  char *v41; // rdx
  __int64 v42; // rax
  __m128i *v43; // rdx
  __int64 v44; // rdi
  __m128i v45; // xmm0
  __int64 v46; // r10
  _BYTE *v47; // rax
  __int64 v48; // r15
  __int64 v49; // r15
  __int64 v50; // r14
  __int64 v51; // r14
  unsigned __int8 *v52; // rsi
  __int64 *v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned __int64 v56; // rax
  __m128i v57; // xmm0
  _BYTE *v58; // rdx
  unsigned __int64 v59; // rsi
  unsigned __int8 *v60; // rax
  unsigned __int8 *v61; // r9
  unsigned int v62; // eax
  unsigned int v63; // eax
  unsigned int v64; // edi
  __int64 v65; // rcx
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // [rsp+0h] [rbp-1A0h]
  signed int v69; // [rsp+10h] [rbp-190h]
  unsigned __int8 *v71; // [rsp+18h] [rbp-188h]
  __int64 v72; // [rsp+18h] [rbp-188h]
  __int64 v73; // [rsp+18h] [rbp-188h]
  size_t v74; // [rsp+18h] [rbp-188h]
  signed int v75; // [rsp+24h] [rbp-17Ch]
  signed int v76; // [rsp+28h] [rbp-178h]
  signed int v77; // [rsp+2Ch] [rbp-174h]
  unsigned __int64 v78; // [rsp+30h] [rbp-170h] BYREF
  size_t **v79; // [rsp+38h] [rbp-168h]
  unsigned __int8 *v80; // [rsp+50h] [rbp-150h] BYREF
  size_t v81; // [rsp+58h] [rbp-148h]
  _BYTE v82[16]; // [rsp+60h] [rbp-140h] BYREF
  unsigned __int8 *v83[2]; // [rsp+70h] [rbp-130h] BYREF
  __int64 v84; // [rsp+80h] [rbp-120h] BYREF
  unsigned __int8 *v85[2]; // [rsp+90h] [rbp-110h] BYREF
  __int64 v86; // [rsp+A0h] [rbp-100h] BYREF
  unsigned __int8 *v87[2]; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v88; // [rsp+C0h] [rbp-E0h] BYREF
  unsigned __int8 *v89[2]; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v90; // [rsp+E0h] [rbp-C0h] BYREF
  unsigned __int8 *v91[2]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v92; // [rsp+100h] [rbp-A0h] BYREF
  unsigned __int8 *v93[2]; // [rsp+110h] [rbp-90h] BYREF
  __int64 v94; // [rsp+120h] [rbp-80h] BYREF
  __int64 v95[2]; // [rsp+130h] [rbp-70h] BYREF
  char *v96; // [rsp+140h] [rbp-60h]
  __int64 v97; // [rsp+148h] [rbp-58h]
  char *v98; // [rsp+150h] [rbp-50h]
  __int64 v99; // [rsp+158h] [rbp-48h]
  unsigned __int8 **v100; // [rsp+160h] [rbp-40h]

  v3 = a1;
  sub_36FD760(a1);
  v4 = *(_QWORD *)(a1 + 24);
  if ( *(_QWORD *)(a1 + 32) != v4 )
    *(_QWORD *)(a1 + 32) = v4;
  sub_36FD240((__int64 *)&v78, (__int64 **)a1);
  v80 = v82;
  v81 = 0;
  v82[0] = 0;
  sub_2240E30((__int64)&v80, 0x1388u);
  v100 = &v80;
  v95[1] = 0;
  v96 = 0;
  v99 = 0x100000000LL;
  v97 = 0;
  v98 = 0;
  v95[0] = (__int64)&unk_49DD210;
  sub_CB5980((__int64)v95, 0, 0, 0);
  v5 = (__m128i *)v98;
  if ( (unsigned __int64)(v97 - (_QWORD)v98) <= 0x22 )
  {
    v66 = sub_CB6200((__int64)v95, "------- Dumping inliner stats for [", 0x23u);
    v8 = *(void **)(v66 + 32);
    v7 = (__int64 *)v66;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_451EF60);
    v98[34] = 91;
    v7 = v95;
    v5[2].m128i_i16[0] = 8306;
    *v5 = si128;
    v5[1] = _mm_load_si128((const __m128i *)&xmmword_451EF70);
    v8 = v98 + 35;
    v98 += 35;
  }
  v9 = *(_QWORD *)(v3 + 64);
  v10 = *(unsigned __int8 **)(v3 + 56);
  v11 = v7[3] - (_QWORD)v8;
  if ( v9 > v11 )
  {
    v67 = sub_CB6200((__int64)v7, v10, *(_QWORD *)(v3 + 64));
    v8 = *(void **)(v67 + 32);
    v7 = (__int64 *)v67;
    v11 = *(_QWORD *)(v67 + 24) - (_QWORD)v8;
LABEL_7:
    if ( v11 > 9 )
      goto LABEL_8;
LABEL_70:
    sub_CB6200((__int64)v7, "] -------\n", 0xAu);
    v12 = (__m128i *)v98;
    if ( !a2 )
      goto LABEL_9;
    goto LABEL_71;
  }
  if ( !v9 )
    goto LABEL_7;
  memcpy(v8, v10, *(_QWORD *)(v3 + 64));
  v8 = (void *)(v9 + v7[4]);
  v56 = v7[3] - (_QWORD)v8;
  v7[4] = (__int64)v8;
  if ( v56 <= 9 )
    goto LABEL_70;
LABEL_8:
  qmemcpy(v8, "] -------\n", 10);
  v7[4] += 10;
  v12 = (__m128i *)v98;
  if ( !a2 )
    goto LABEL_9;
LABEL_71:
  if ( (unsigned __int64)(v97 - (_QWORD)v12) <= 0x1D )
  {
    sub_CB6200((__int64)v95, "-- List of inlined functions:\n", 0x1Eu);
    v12 = (__m128i *)v98;
  }
  else
  {
    v57 = _mm_load_si128((const __m128i *)&xmmword_42C6010);
    qmemcpy(&v12[1], "ed functions:\n", 14);
    *v12 = v57;
    v12 = (__m128i *)(v98 + 30);
    v98 += 30;
  }
LABEL_9:
  v13 = v79;
  if ( (size_t **)v78 != v79 )
  {
    v68 = v3;
    v14 = 0;
    v15 = (size_t **)v78;
    v16 = a2;
    v75 = 0;
    v77 = 0;
    v76 = 0;
    while ( 1 )
    {
      v17 = (*v15)[1];
      if ( !*(_DWORD *)(v17 + 80) )
        goto LABEL_12;
      v18 = *(_DWORD *)(v17 + 84) > 0;
      if ( *(_BYTE *)(v17 + 88) )
      {
        v77 += v18;
        ++v14;
        if ( v16 )
          goto LABEL_16;
LABEL_12:
        if ( v13 == ++v15 )
          goto LABEL_35;
      }
      else
      {
        ++v76;
        v75 += v18;
        if ( !v16 )
          goto LABEL_12;
LABEL_16:
        if ( (unsigned __int64)(v97 - (_QWORD)v12) <= 7 )
        {
          v19 = (__int64 *)sub_CB6200((__int64)v95, "Inlined ", 8u);
        }
        else
        {
          v19 = v95;
          v12->m128i_i64[0] = 0x2064656E696C6E49LL;
          v98 += 8;
        }
        v20 = "not imported ";
        if ( *(_BYTE *)((*v15)[1] + 88) )
          v20 = "imported ";
        v71 = (unsigned __int8 *)v20;
        v21 = strlen(v20);
        v22 = (_QWORD *)v19[4];
        if ( v21 <= v19[3] - (__int64)v22 )
        {
          *v22 = *(_QWORD *)v71;
          *(_QWORD *)((char *)v22 + (unsigned int)v21 - 8) = *(_QWORD *)&v71[(unsigned int)v21 - 8];
          v59 = (unsigned __int64)(v22 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          v60 = (unsigned __int8 *)v22 - v59;
          v61 = (unsigned __int8 *)(v71 - v60);
          v62 = (v21 + (_DWORD)v60) & 0xFFFFFFF8;
          if ( v62 >= 8 )
          {
            v63 = v62 & 0xFFFFFFF8;
            v64 = 0;
            do
            {
              v65 = v64;
              v64 += 8;
              *(_QWORD *)(v59 + v65) = *(_QWORD *)&v61[v65];
            }
            while ( v64 < v63 );
          }
          v24 = (void *)(v19[4] + v21);
          v19[4] = (__int64)v24;
        }
        else
        {
          v23 = sub_CB6200((__int64)v19, v71, v21);
          v24 = *(void **)(v23 + 32);
          v19 = (__int64 *)v23;
        }
        if ( (unsigned __int64)(v19[3] - (_QWORD)v24) <= 9 )
        {
          v55 = sub_CB6200((__int64)v19, "function [", 0xAu);
          v25 = *(_BYTE **)(v55 + 32);
          v19 = (__int64 *)v55;
        }
        else
        {
          qmemcpy(v24, "function [", 10);
          v25 = (_BYTE *)(v19[4] + 10);
          v19[4] = (__int64)v25;
        }
        v26 = (_BYTE *)v19[3];
        v27 = **v15;
        v28 = (unsigned __int8 *)(*v15 + 2);
        if ( v27 > v26 - v25 )
        {
          v19 = (__int64 *)sub_CB6200((__int64)v19, v28, v27);
          v25 = (_BYTE *)v19[4];
          if ( (_BYTE *)v19[3] != v25 )
            goto LABEL_28;
        }
        else
        {
          if ( v27 )
          {
            v74 = **v15;
            memcpy(v25, v28, v27);
            v58 = (_BYTE *)(v19[4] + v74);
            v19[4] = (__int64)v58;
            v26 = (_BYTE *)v19[3];
            v25 = v58;
          }
          if ( v26 != v25 )
          {
LABEL_28:
            *v25 = 93;
            v29 = (void *)(v19[4] + 1);
            v30 = v19[3];
            v19[4] = (__int64)v29;
            if ( (unsigned __int64)(v30 - (_QWORD)v29) <= 0xC )
              goto LABEL_65;
            goto LABEL_29;
          }
        }
        v19 = (__int64 *)sub_CB6200((__int64)v19, (unsigned __int8 *)"]", 1u);
        v29 = (void *)v19[4];
        if ( (unsigned __int64)(v19[3] - (_QWORD)v29) <= 0xC )
        {
LABEL_65:
          v19 = (__int64 *)sub_CB6200((__int64)v19, ": #inlines = ", 0xDu);
          goto LABEL_30;
        }
LABEL_29:
        qmemcpy(v29, ": #inlines = ", 13);
        v19[4] += 13;
LABEL_30:
        v31 = sub_CB59F0((__int64)v19, *(int *)((*v15)[1] + 80));
        v32 = *(__m128i **)(v31 + 32);
        v33 = v31;
        if ( *(_QWORD *)(v31 + 24) - (_QWORD)v32 <= 0x20u )
        {
          v33 = sub_CB6200(v31, ", #inlines_to_importing_module = ", 0x21u);
        }
        else
        {
          v34 = _mm_load_si128((const __m128i *)&xmmword_42C6020);
          v32[2].m128i_i8[0] = 32;
          *v32 = v34;
          v32[1] = _mm_load_si128((const __m128i *)&xmmword_42C6030);
          *(_QWORD *)(v31 + 32) += 33LL;
        }
        v35 = sub_CB59F0(v33, *(int *)((*v15)[1] + 84));
        v36 = *(_BYTE **)(v35 + 32);
        if ( *(_BYTE **)(v35 + 24) == v36 )
        {
          sub_CB6200(v35, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v36 = 10;
          ++*(_QWORD *)(v35 + 32);
        }
        v12 = (__m128i *)v98;
        if ( v13 == ++v15 )
        {
LABEL_35:
          v3 = v68;
          v69 = v14 + v76;
          goto LABEL_36;
        }
      }
    }
  }
  v69 = 0;
  v14 = 0;
  v75 = 0;
  v77 = 0;
  v76 = 0;
LABEL_36:
  v37 = *(_DWORD *)(v3 + 52);
  v38 = *(_DWORD *)(v3 + 48) - v37;
  v39 = v37 - v77;
  if ( (unsigned __int64)(v97 - (_QWORD)v12) <= 0xB )
  {
    v40 = (__int64 *)sub_CB6200((__int64)v95, "-- Summary:\n", 0xCu);
    v41 = (char *)v40[4];
    if ( (unsigned __int64)(v40[3] - (_QWORD)v41) > 0xE )
      goto LABEL_38;
  }
  else
  {
    v40 = v95;
    qmemcpy(v12, "-- Summary:\n", 12);
    v41 = v98 + 12;
    v98 = v41;
    if ( (unsigned __int64)(v97 - (_QWORD)v41) > 0xE )
    {
LABEL_38:
      qmemcpy(v41, "All functions: ", 15);
      v40[4] += 15;
      goto LABEL_39;
    }
  }
  v40 = (__int64 *)sub_CB6200((__int64)v40, "All functions: ", 0xFu);
LABEL_39:
  v42 = sub_CB59F0((__int64)v40, *(int *)(v3 + 48));
  v43 = *(__m128i **)(v42 + 32);
  v44 = v42;
  if ( *(_QWORD *)(v42 + 24) - (_QWORD)v43 <= 0x15u )
  {
    v44 = sub_CB6200(v42, ", imported functions: ", 0x16u);
  }
  else
  {
    v45 = _mm_load_si128((const __m128i *)&xmmword_42C6040);
    v43[1].m128i_i32[0] = 1936617321;
    v43[1].m128i_i16[2] = 8250;
    *v43 = v45;
    *(_QWORD *)(v42 + 32) += 22LL;
  }
  v46 = sub_CB59F0(v44, *(int *)(v3 + 52));
  v47 = *(_BYTE **)(v46 + 32);
  if ( *(_BYTE **)(v46 + 24) == v47 )
  {
    v46 = sub_CB6200(v46, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v47 = 10;
    ++*(_QWORD *)(v46 + 32);
  }
  v72 = v46;
  sub_36FBAC0((__int64)v83, "inlined functions", v69, *(_DWORD *)(v3 + 48), "all functions", 1);
  v73 = sub_CB6200(v72, v83[0], (size_t)v83[1]);
  sub_36FBAC0((__int64)v85, "imported functions inlined anywhere", v14, *(_DWORD *)(v3 + 52), "imported functions", 1);
  v48 = sub_CB6200(v73, v85[0], (size_t)v85[1]);
  sub_36FBAC0(
    (__int64)v87,
    "imported functions inlined into importing module",
    v77,
    *(_DWORD *)(v3 + 52),
    "imported functions",
    0);
  v49 = sub_CB6200(v48, v87[0], (size_t)v87[1]);
  sub_36FBAC0((__int64)v89, ", remaining", v39, *(_DWORD *)(v3 + 52), "imported functions", 1);
  v50 = sub_CB6200(v49, v89[0], (size_t)v89[1]);
  sub_36FBAC0((__int64)v91, "non-imported functions inlined anywhere", v76, v38, "non-imported functions", 1);
  v51 = sub_CB6200(v50, v91[0], (size_t)v91[1]);
  sub_36FBAC0(
    (__int64)v93,
    "non-imported functions inlined into importing module",
    v75,
    v38,
    "non-imported functions",
    1);
  v52 = v93[0];
  sub_CB6200(v51, v93[0], (size_t)v93[1]);
  if ( (__int64 *)v93[0] != &v94 )
  {
    v52 = (unsigned __int8 *)(v94 + 1);
    j_j___libc_free_0((unsigned __int64)v93[0]);
  }
  if ( (__int64 *)v91[0] != &v92 )
  {
    v52 = (unsigned __int8 *)(v92 + 1);
    j_j___libc_free_0((unsigned __int64)v91[0]);
  }
  if ( (__int64 *)v89[0] != &v90 )
  {
    v52 = (unsigned __int8 *)(v90 + 1);
    j_j___libc_free_0((unsigned __int64)v89[0]);
  }
  if ( (__int64 *)v87[0] != &v88 )
  {
    v52 = (unsigned __int8 *)(v88 + 1);
    j_j___libc_free_0((unsigned __int64)v87[0]);
  }
  if ( (__int64 *)v85[0] != &v86 )
  {
    v52 = (unsigned __int8 *)(v86 + 1);
    j_j___libc_free_0((unsigned __int64)v85[0]);
  }
  v53 = (__int64 *)v83[0];
  if ( (__int64 *)v83[0] != &v84 )
  {
    v52 = (unsigned __int8 *)(v84 + 1);
    j_j___libc_free_0((unsigned __int64)v83[0]);
  }
  if ( v98 != v96 )
  {
    v53 = v95;
    sub_CB5AE0(v95);
  }
  v54 = sub_C5F790((__int64)v53, (__int64)v52);
  sub_CB6200(v54, v80, v81);
  v95[0] = (__int64)&unk_49DD210;
  sub_CB5840((__int64)v95);
  if ( v80 != v82 )
    j_j___libc_free_0((unsigned __int64)v80);
  if ( v78 )
    j_j___libc_free_0(v78);
}
