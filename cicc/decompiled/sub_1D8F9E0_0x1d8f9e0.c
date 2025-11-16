// Function: sub_1D8F9E0
// Address: 0x1d8f9e0
//
__int64 __fastcall sub_1D8F9E0(__int64 a1, __int64 a2)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r12
  void *v10; // rdx
  const char *v11; // rax
  size_t v12; // rdx
  _WORD *v13; // rdi
  char *v14; // rsi
  unsigned __int64 v15; // rax
  int *v16; // r13
  int *i; // r12
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdi
  _BYTE *v23; // rax
  __int64 v24; // r12
  __m128i *v25; // rdx
  __m128i si128; // xmm0
  const char *v27; // rax
  size_t v28; // rdx
  _WORD *v29; // rdi
  char *v30; // rsi
  unsigned __int64 v31; // rax
  __int64 v32; // r13
  __int64 v33; // r12
  _BYTE *v34; // rax
  _WORD *v35; // rdi
  _BYTE *v36; // rax
  size_t *v37; // rsi
  size_t v38; // r15
  void *v39; // rsi
  __int64 v40; // rax
  char *v41; // rsi
  size_t v42; // rdx
  _QWORD *v43; // rax
  __int64 v44; // rax
  void *v45; // rdx
  int *v46; // r15
  int *v47; // r12
  __int64 v48; // rdi
  _BYTE *v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rdi
  _BYTE *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  unsigned __int64 v55; // rdi
  char *v56; // rax
  char *v57; // rsi
  unsigned int v58; // eax
  char *v59; // r8
  unsigned int v60; // eax
  unsigned int v61; // ecx
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  _WORD *v66; // rdx
  __int64 v67; // rax
  _WORD *v68; // rdx
  __int64 j; // [rsp-48h] [rbp-48h]
  size_t v70; // [rsp-48h] [rbp-48h]
  size_t v71; // [rsp-48h] [rbp-48h]
  __int64 *v72; // [rsp-40h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 19) & 0x40) != 0 )
    return 0;
  v4 = *(__int64 **)(a1 + 8);
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_74:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4FC3606 )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_74;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4FC3606);
  v8 = sub_1D8F610(v7, a2);
  v9 = *(_QWORD *)(a1 + 160);
  v72 = (__int64 *)v8;
  v10 = *(void **)(v9 + 24);
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 0xCu )
  {
    v9 = sub_16E7EE0(v9, "GC roots for ", 0xDu);
  }
  else
  {
    qmemcpy(v10, "GC roots for ", 13);
    *(_QWORD *)(v9 + 24) += 13LL;
  }
  v11 = sub_1649960(*v72);
  v13 = *(_WORD **)(v9 + 24);
  v14 = (char *)v11;
  v15 = *(_QWORD *)(v9 + 16) - (_QWORD)v13;
  if ( v15 >= v12 )
  {
    if ( v12 )
    {
      v71 = v12;
      memcpy(v13, v14, v12);
      v67 = *(_QWORD *)(v9 + 16);
      v68 = (_WORD *)(*(_QWORD *)(v9 + 24) + v71);
      *(_QWORD *)(v9 + 24) = v68;
      v13 = v68;
      v15 = v67 - (_QWORD)v68;
    }
    if ( v15 > 1 )
      goto LABEL_13;
LABEL_70:
    sub_16E7EE0(v9, ":\n", 2u);
    goto LABEL_14;
  }
  v63 = sub_16E7EE0(v9, v14, v12);
  v13 = *(_WORD **)(v63 + 24);
  v9 = v63;
  if ( *(_QWORD *)(v63 + 16) - (_QWORD)v13 <= 1u )
    goto LABEL_70;
LABEL_13:
  *v13 = 2618;
  *(_QWORD *)(v9 + 24) += 2LL;
LABEL_14:
  v16 = (int *)v72[4];
  for ( i = (int *)v72[3]; v16 != i; i += 4 )
  {
    while ( 1 )
    {
      v22 = *(_QWORD *)(a1 + 160);
      v23 = *(_BYTE **)(v22 + 24);
      if ( *(_BYTE **)(v22 + 16) == v23 )
      {
        v22 = sub_16E7EE0(v22, "\t", 1u);
      }
      else
      {
        *v23 = 9;
        ++*(_QWORD *)(v22 + 24);
      }
      v18 = sub_16E7AB0(v22, *i);
      v19 = *(_BYTE **)(v18 + 24);
      if ( *(_BYTE **)(v18 + 16) == v19 )
      {
        v18 = sub_16E7EE0(v18, "\t", 1u);
      }
      else
      {
        *v19 = 9;
        ++*(_QWORD *)(v18 + 24);
      }
      v20 = sub_16E7AB0(v18, i[1]);
      v21 = *(_QWORD *)(v20 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v20 + 16) - v21) <= 4 )
        break;
      i += 4;
      *(_DWORD *)v21 = 1567650651;
      *(_BYTE *)(v21 + 4) = 10;
      *(_QWORD *)(v20 + 24) += 5LL;
      if ( v16 == i )
        goto LABEL_25;
    }
    sub_16E7EE0(v20, "[sp]\n", 5u);
  }
LABEL_25:
  v24 = *(_QWORD *)(a1 + 160);
  v25 = *(__m128i **)(v24 + 24);
  if ( *(_QWORD *)(v24 + 16) - (_QWORD)v25 <= 0x12u )
  {
    v24 = sub_16E7EE0(*(_QWORD *)(a1 + 160), "GC safe points for ", 0x13u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42E9390);
    v25[1].m128i_i8[2] = 32;
    v25[1].m128i_i16[0] = 29295;
    *v25 = si128;
    *(_QWORD *)(v24 + 24) += 19LL;
  }
  v27 = sub_1649960(*v72);
  v29 = *(_WORD **)(v24 + 24);
  v30 = (char *)v27;
  v31 = *(_QWORD *)(v24 + 16) - (_QWORD)v29;
  if ( v28 > v31 )
  {
    v64 = sub_16E7EE0(v24, v30, v28);
    v29 = *(_WORD **)(v64 + 24);
    v24 = v64;
    v31 = *(_QWORD *)(v64 + 16) - (_QWORD)v29;
  }
  else if ( v28 )
  {
    v70 = v28;
    memcpy(v29, v30, v28);
    v65 = *(_QWORD *)(v24 + 16);
    v66 = (_WORD *)(*(_QWORD *)(v24 + 24) + v70);
    *(_QWORD *)(v24 + 24) = v66;
    v29 = v66;
    v31 = v65 - (_QWORD)v66;
  }
  if ( v31 <= 1 )
  {
    sub_16E7EE0(v24, ":\n", 2u);
  }
  else
  {
    *v29 = 2618;
    *(_QWORD *)(v24 + 24) += 2LL;
  }
  v32 = v72[6];
  for ( j = v72[7]; j != v32; v32 += 24 )
  {
    v33 = *(_QWORD *)(a1 + 160);
    v34 = *(_BYTE **)(v33 + 24);
    if ( *(_BYTE **)(v33 + 16) == v34 )
    {
      v54 = sub_16E7EE0(*(_QWORD *)(a1 + 160), "\t", 1u);
      v35 = *(_WORD **)(v54 + 24);
      v33 = v54;
    }
    else
    {
      *v34 = 9;
      v35 = (_WORD *)(*(_QWORD *)(v33 + 24) + 1LL);
      *(_QWORD *)(v33 + 24) = v35;
    }
    v36 = *(_BYTE **)(v32 + 8);
    if ( (*v36 & 4) != 0 )
    {
      v37 = (size_t *)*((_QWORD *)v36 - 1);
      v38 = *v37;
      v39 = v37 + 2;
      if ( v38 <= *(_QWORD *)(v33 + 16) - (_QWORD)v35 )
      {
        if ( v38 )
        {
          memcpy(v35, v39, v38);
          v35 = (_WORD *)(v38 + *(_QWORD *)(v33 + 24));
          *(_QWORD *)(v33 + 24) = v35;
        }
      }
      else
      {
        v40 = sub_16E7EE0(v33, (char *)v39, v38);
        v35 = *(_WORD **)(v40 + 24);
        v33 = v40;
      }
    }
    if ( *(_QWORD *)(v33 + 16) - (_QWORD)v35 <= 1u )
    {
      v33 = sub_16E7EE0(v33, ": ", 2u);
    }
    else
    {
      *v35 = 8250;
      *(_QWORD *)(v33 + 24) += 2LL;
    }
    v41 = "post-call";
    v42 = 9LL - (*(_DWORD *)v32 == 0);
    if ( !*(_DWORD *)v32 )
      v41 = "pre-call";
    v43 = *(_QWORD **)(v33 + 24);
    if ( *(_QWORD *)(v33 + 16) - (_QWORD)v43 >= v42 )
    {
      *v43 = *(_QWORD *)v41;
      *(_QWORD *)((char *)v43 + (unsigned int)v42 - 8) = *(_QWORD *)&v41[(unsigned int)v42 - 8];
      v55 = (unsigned __int64)(v43 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      v56 = (char *)v43 - v55;
      v57 = (char *)(v41 - v56);
      v58 = (v42 + (_DWORD)v56) & 0xFFFFFFF8;
      v59 = v57;
      if ( v58 >= 8 )
      {
        v60 = v58 & 0xFFFFFFF8;
        v61 = 0;
        do
        {
          v62 = v61;
          v61 += 8;
          *(_QWORD *)(v55 + v62) = *(_QWORD *)&v59[v62];
        }
        while ( v61 < v60 );
      }
      v45 = (void *)(*(_QWORD *)(v33 + 24) + v42);
      *(_QWORD *)(v33 + 24) = v45;
    }
    else
    {
      v44 = sub_16E7EE0(v33, v41, v42);
      v45 = *(void **)(v44 + 24);
      v33 = v44;
    }
    if ( *(_QWORD *)(v33 + 16) - (_QWORD)v45 <= 9u )
    {
      sub_16E7EE0(v33, ", live = {", 0xAu);
    }
    else
    {
      qmemcpy(v45, ", live = {", 10);
      *(_QWORD *)(v33 + 24) += 10LL;
    }
    v46 = (int *)v72[3];
    v47 = (int *)v72[4];
    while ( 1 )
    {
      v48 = *(_QWORD *)(a1 + 160);
      v49 = *(_BYTE **)(v48 + 24);
      if ( *(_BYTE **)(v48 + 16) == v49 )
      {
        v48 = sub_16E7EE0(v48, " ", 1u);
      }
      else
      {
        *v49 = 32;
        ++*(_QWORD *)(v48 + 24);
      }
      v50 = *v46;
      v46 += 4;
      sub_16E7AB0(v48, v50);
      v51 = *(_QWORD *)(a1 + 160);
      if ( v47 == v46 )
        break;
      v52 = *(_BYTE **)(v51 + 24);
      if ( *(_BYTE **)(v51 + 16) == v52 )
      {
        sub_16E7EE0(v51, ",", 1u);
      }
      else
      {
        *v52 = 44;
        ++*(_QWORD *)(v51 + 24);
      }
    }
    v53 = *(_QWORD *)(v51 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v51 + 16) - v53) <= 2 )
    {
      sub_16E7EE0(v51, " }\n", 3u);
    }
    else
    {
      *(_BYTE *)(v53 + 2) = 10;
      *(_WORD *)v53 = 32032;
      *(_QWORD *)(v51 + 24) += 3LL;
    }
  }
  return 0;
}
