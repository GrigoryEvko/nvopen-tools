// Function: sub_214E460
// Address: 0x214e460
//
__int64 __fastcall sub_214E460(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  void *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rax
  size_t v18; // rdx
  _BYTE *v19; // rdi
  char *v20; // rsi
  _BYTE *v21; // rax
  char *v22; // rax
  char *v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r12
  __int64 v31; // rdx
  __m128i si128; // xmm0
  _WORD *v33; // rdx
  const char *v34; // rax
  size_t v35; // rdx
  _BYTE *v36; // rdi
  char *v37; // rsi
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  _WORD *v42; // rdx
  __int64 v43; // rdx
  unsigned int v44; // r12d
  __m128i v45; // xmm0
  __int64 v46; // rdi
  _WORD *v47; // rdx
  _BYTE *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rax
  unsigned __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // [rsp+8h] [rbp-68h]
  __int64 v57; // [rsp+10h] [rbp-60h]
  unsigned int v60; // [rsp+28h] [rbp-48h]
  unsigned int v61; // [rsp+2Ch] [rbp-44h]
  __int64 v62; // [rsp+30h] [rbp-40h]
  size_t v63; // [rsp+30h] [rbp-40h]
  __int64 v64; // [rsp+38h] [rbp-38h]
  size_t v65; // [rsp+38h] [rbp-38h]

  v5 = *(void **)(a2 + 24);
  v6 = *(_QWORD *)(a3 - 8LL * *(unsigned int *)(a3 + 8));
  v7 = *a4;
  v57 = *(_QWORD *)(v6 + 136);
  v56 = *(_QWORD *)(v57 + 96);
  v60 = *(_DWORD *)(a1 + 912);
  if ( a4[1] != *a4 )
  {
    v61 = 0;
    v8 = 0;
    while ( 1 )
    {
      v9 = *(_QWORD *)(v7 + 8 * v8);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v5 <= 9u )
      {
        v10 = sub_16E7EE0(a2, ".metadata ", 0xAu);
      }
      else
      {
        v10 = a2;
        qmemcpy(v5, ".metadata ", 10);
        *(_QWORD *)(a2 + 24) += 10LL;
      }
      v11 = *(unsigned int *)(a1 + 912);
      *(_DWORD *)(a1 + 912) = v11 + 1;
      v12 = sub_16E7A90(v10, v11);
      v13 = *(_QWORD *)(v12 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v12 + 16) - v13) <= 2 )
      {
        sub_16E7EE0(v12, " {\n", 3u);
      }
      else
      {
        *(_BYTE *)(v13 + 2) = 10;
        *(_WORD *)v13 = 31520;
        *(_QWORD *)(v12 + 24) += 3LL;
      }
      v14 = *(_QWORD *)(a2 + 24);
      if ( (_DWORD)v56 != -1 )
        break;
LABEL_23:
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v14) <= 2 )
      {
        sub_16E7EE0(a2, "}\n\n", 3u);
        v5 = *(void **)(a2 + 24);
      }
      else
      {
        *(_BYTE *)(v14 + 2) = 10;
        *(_WORD *)v14 = 2685;
        v5 = (void *)(*(_QWORD *)(a2 + 24) + 3LL);
        *(_QWORD *)(a2 + 24) = v5;
      }
      v8 = ++v61;
      v7 = *a4;
      if ( v61 >= (unsigned __int64)((a4[1] - *a4) >> 3) )
        goto LABEL_26;
    }
    v15 = 0;
    v64 = (unsigned int)(v56 + 1);
    while ( 1 )
    {
      v24 = *(_QWORD *)(v9 + 8 * (v15 - *(unsigned int *)(v9 + 8)));
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v14) > 1 )
      {
        v16 = a2;
        *(_WORD *)v14 = 8713;
        *(_QWORD *)(a2 + 24) += 2LL;
      }
      else
      {
        v62 = *(_QWORD *)(v9 + 8 * (v15 - *(unsigned int *)(v9 + 8)));
        v25 = sub_16E7EE0(a2, "\t\"", 2u);
        v24 = v62;
        v16 = v25;
      }
      v17 = sub_161E970(v24);
      v19 = *(_BYTE **)(v16 + 24);
      v20 = (char *)v17;
      v21 = *(_BYTE **)(v16 + 16);
      if ( v18 > v21 - v19 )
      {
        v16 = sub_16E7EE0(v16, v20, v18);
        v21 = *(_BYTE **)(v16 + 16);
        v19 = *(_BYTE **)(v16 + 24);
      }
      else if ( v18 )
      {
        v63 = v18;
        memcpy(v19, v20, v18);
        v21 = *(_BYTE **)(v16 + 16);
        v19 = (_BYTE *)(v63 + *(_QWORD *)(v16 + 24));
        *(_QWORD *)(v16 + 24) = v19;
      }
      if ( v19 == v21 )
      {
        sub_16E7EE0(v16, "\"", 1u);
        v22 = *(char **)(a2 + 16);
        v23 = *(char **)(a2 + 24);
        if ( (_DWORD)v56 != (_DWORD)v15 )
        {
LABEL_15:
          if ( (unsigned __int64)(v22 - v23) <= 1 )
          {
            sub_16E7EE0(a2, ",\n", 2u);
            v14 = *(_QWORD *)(a2 + 24);
          }
          else
          {
            *(_WORD *)v23 = 2604;
            v14 = *(_QWORD *)(a2 + 24) + 2LL;
            *(_QWORD *)(a2 + 24) = v14;
          }
          goto LABEL_17;
        }
      }
      else
      {
        *v19 = 34;
        ++*(_QWORD *)(v16 + 24);
        v22 = *(char **)(a2 + 16);
        v23 = *(char **)(a2 + 24);
        if ( (_DWORD)v56 != (_DWORD)v15 )
          goto LABEL_15;
      }
      if ( v23 == v22 )
      {
        sub_16E7EE0(a2, "\n", 1u);
        v14 = *(_QWORD *)(a2 + 24);
LABEL_17:
        if ( v64 == ++v15 )
          goto LABEL_23;
      }
      else
      {
        *v23 = 10;
        ++v15;
        v14 = *(_QWORD *)(a2 + 24) + 1LL;
        *(_QWORD *)(a2 + 24) = v14;
        if ( v64 == v15 )
          goto LABEL_23;
      }
    }
  }
LABEL_26:
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v5 <= 9u )
  {
    v26 = sub_16E7EE0(a2, ".metadata ", 0xAu);
  }
  else
  {
    v26 = a2;
    qmemcpy(v5, ".metadata ", 10);
    *(_QWORD *)(a2 + 24) += 10LL;
  }
  v27 = *(unsigned int *)(a1 + 912);
  *(_DWORD *)(a1 + 912) = v27 + 1;
  v28 = sub_16E7A90(v26, v27);
  v29 = *(_QWORD *)(v28 + 24);
  v30 = v28;
  if ( (unsigned __int64)(*(_QWORD *)(v28 + 16) - v29) <= 2 )
  {
    v55 = sub_16E7EE0(v28, " {\n", 3u);
    v31 = *(_QWORD *)(v55 + 24);
    v30 = v55;
  }
  else
  {
    *(_BYTE *)(v29 + 2) = 10;
    *(_WORD *)v29 = 31520;
    v31 = *(_QWORD *)(v28 + 24) + 3LL;
    *(_QWORD *)(v28 + 24) = v31;
  }
  if ( (unsigned __int64)(*(_QWORD *)(v30 + 16) - v31) <= 0x16 )
  {
    v54 = sub_16E7EE0(v30, "\t\"cl_kernel_arg_info\",\n", 0x17u);
    v33 = *(_WORD **)(v54 + 24);
    v30 = v54;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4327100);
    *(_DWORD *)(v31 + 16) = 1868983913;
    *(_WORD *)(v31 + 20) = 11298;
    *(_BYTE *)(v31 + 22) = 10;
    *(__m128i *)v31 = si128;
    v33 = (_WORD *)(*(_QWORD *)(v30 + 24) + 23LL);
    *(_QWORD *)(v30 + 24) = v33;
  }
  if ( *(_QWORD *)(v30 + 16) - (_QWORD)v33 <= 1u )
  {
    v30 = sub_16E7EE0(v30, "\t\"", 2u);
  }
  else
  {
    *v33 = 8713;
    *(_QWORD *)(v30 + 24) += 2LL;
  }
  v34 = sub_1649960(v57);
  v36 = *(_BYTE **)(v30 + 24);
  v37 = (char *)v34;
  v38 = *(_QWORD *)(v30 + 16) - (_QWORD)v36;
  if ( v38 < v35 )
  {
    v53 = sub_16E7EE0(v30, v37, v35);
    v36 = *(_BYTE **)(v53 + 24);
    v30 = v53;
    v38 = *(_QWORD *)(v53 + 16) - (_QWORD)v36;
LABEL_36:
    if ( v38 > 2 )
      goto LABEL_37;
LABEL_59:
    v30 = sub_16E7EE0(v30, "\",\n", 3u);
    v39 = *(_QWORD *)(v30 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v30 + 16) - v39) > 5 )
      goto LABEL_38;
    goto LABEL_60;
  }
  if ( !v35 )
    goto LABEL_36;
  v65 = v35;
  memcpy(v36, v37, v35);
  v36 = (_BYTE *)(v65 + *(_QWORD *)(v30 + 24));
  v52 = *(_QWORD *)(v30 + 16) - (_QWORD)v36;
  *(_QWORD *)(v30 + 24) = v36;
  if ( v52 <= 2 )
    goto LABEL_59;
LABEL_37:
  v36[2] = 10;
  *(_WORD *)v36 = 11298;
  v39 = *(_QWORD *)(v30 + 24) + 3LL;
  v40 = *(_QWORD *)(v30 + 16);
  *(_QWORD *)(v30 + 24) = v39;
  if ( (unsigned __int64)(v40 - v39) > 5 )
  {
LABEL_38:
    *(_DWORD *)v39 = 862072329;
    *(_WORD *)(v39 + 4) = 8242;
    *(_QWORD *)(v30 + 24) += 6LL;
    goto LABEL_39;
  }
LABEL_60:
  v30 = sub_16E7EE0(v30, "\t.b32 ", 6u);
LABEL_39:
  v41 = sub_16E7A90(v30, (unsigned int)v56);
  v42 = *(_WORD **)(v41 + 24);
  if ( *(_QWORD *)(v41 + 16) - (_QWORD)v42 <= 1u )
  {
    sub_16E7EE0(v41, ",\n", 2u);
  }
  else
  {
    *v42 = 2604;
    *(_QWORD *)(v41 + 24) += 2LL;
  }
  v43 = *(_QWORD *)(a2 + 24);
  v44 = v60;
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v43) <= 0x10 )
    goto LABEL_47;
LABEL_42:
  v45 = _mm_load_si128((const __m128i *)&xmmword_4327110);
  *(_BYTE *)(v43 + 16) = 32;
  v46 = a2;
  *(__m128i *)v43 = v45;
  *(_QWORD *)(a2 + 24) += 17LL;
  while ( 1 )
  {
    sub_16E7A90(v46, v44);
    if ( v44 + 1 == v60 + 5 )
      break;
    v47 = *(_WORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v47 <= 1u )
    {
      sub_16E7EE0(a2, ",\n", 2u);
      v43 = *(_QWORD *)(a2 + 24);
    }
    else
    {
      *v47 = 2604;
      v43 = *(_QWORD *)(a2 + 24) + 2LL;
      *(_QWORD *)(a2 + 24) = v43;
    }
    ++v44;
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v43) > 0x10 )
      goto LABEL_42;
LABEL_47:
    v46 = sub_16E7EE0(a2, "\t.metadata_index ", 0x11u);
  }
  v48 = *(_BYTE **)(a2 + 24);
  if ( *(_BYTE **)(a2 + 16) == v48 )
  {
    sub_16E7EE0(a2, "\n", 1u);
    v49 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v49) > 2 )
      goto LABEL_57;
  }
  else
  {
    *v48 = 10;
    v49 = *(_QWORD *)(a2 + 24) + 1LL;
    v50 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a2 + 24) = v49;
    if ( (unsigned __int64)(v50 - v49) > 2 )
    {
LABEL_57:
      *(_BYTE *)(v49 + 2) = 10;
      *(_WORD *)v49 = 2685;
      *(_QWORD *)(a2 + 24) += 3LL;
      return 2685;
    }
  }
  return sub_16E7EE0(a2, "}\n\n", 3u);
}
