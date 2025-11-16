// Function: sub_39EDC00
// Address: 0x39edc00
//
__int64 __fastcall sub_39EDC00(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        unsigned int a7,
        _BYTE *a8,
        __int64 a9,
        unsigned int a10,
        unsigned int a11,
        void *a12,
        size_t a13)
{
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // r8
  __m128i *v21; // rdx
  __int64 v22; // rdi
  void *v23; // rdx
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // rdi
  _BYTE *v27; // rax
  unsigned __int64 v28; // r13
  __int64 v30; // rdi
  _BYTE *v31; // rax
  __int64 v32; // r14
  char *v33; // rsi
  size_t v34; // rdx
  void *v35; // rdi
  __int64 v36; // rax
  __int64 v37; // r15
  char *v38; // rsi
  size_t v39; // rdx
  unsigned __int64 v40; // rax
  _BYTE *v41; // rdi
  unsigned __int64 v42; // rax
  _BYTE *v43; // rdi
  __int64 v44; // rdi
  _BYTE *v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rax
  _BYTE *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rax
  size_t v53; // [rsp+10h] [rbp-60h]

  v14 = *(_QWORD *)(a1 + 272);
  v15 = *(_QWORD *)(v14 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v14 + 16) - v15) <= 5 )
  {
    v14 = sub_16E7EE0(v14, "\t.loc\t", 6u);
  }
  else
  {
    *(_DWORD *)v15 = 1869360649;
    *(_WORD *)(v15 + 4) = 2403;
    *(_QWORD *)(v14 + 24) += 6LL;
  }
  v16 = sub_16E7A90(v14, a2);
  v17 = *(_BYTE **)(v16 + 24);
  if ( *(_BYTE **)(v16 + 16) == v17 )
  {
    v16 = sub_16E7EE0(v16, " ", 1u);
  }
  else
  {
    *v17 = 32;
    ++*(_QWORD *)(v16 + 24);
  }
  v18 = sub_16E7A90(v16, a3);
  v19 = *(_BYTE **)(v18 + 24);
  if ( *(_BYTE **)(v18 + 16) == v19 )
  {
    v18 = sub_16E7EE0(v18, " ", 1u);
  }
  else
  {
    *v19 = 32;
    ++*(_QWORD *)(v18 + 24);
  }
  sub_16E7A90(v18, a4);
  v20 = *(_QWORD *)(a1 + 272);
  v21 = *(__m128i **)(v20 + 24);
  if ( *(_QWORD *)(v20 + 16) - (_QWORD)v21 <= 0xFu )
  {
    v20 = sub_16E7EE0(*(_QWORD *)(a1 + 272), ", function_name ", 0x10u);
  }
  else
  {
    *v21 = _mm_load_si128((const __m128i *)&xmmword_4534D60);
    *(_QWORD *)(v20 + 24) += 16LL;
  }
  sub_38E2490(a8, v20, 0);
  v22 = *(_QWORD *)(a1 + 272);
  v23 = *(void **)(v22 + 24);
  if ( *(_QWORD *)(v22 + 16) - (_QWORD)v23 <= 0xCu )
  {
    v22 = sub_16E7EE0(v22, ", inlined_at ", 0xDu);
  }
  else
  {
    qmemcpy(v23, ", inlined_at ", 13);
    *(_QWORD *)(v22 + 24) += 13LL;
  }
  v24 = sub_16E7A90(v22, a5);
  v25 = *(_BYTE **)(v24 + 24);
  if ( *(_BYTE **)(v24 + 16) == v25 )
  {
    v24 = sub_16E7EE0(v24, " ", 1u);
  }
  else
  {
    *v25 = 32;
    ++*(_QWORD *)(v24 + 24);
  }
  v26 = sub_16E7A90(v24, a6);
  v27 = *(_BYTE **)(v26 + 24);
  if ( *(_BYTE **)(v26 + 16) == v27 )
  {
    v26 = sub_16E7EE0(v26, " ", 1u);
  }
  else
  {
    *v27 = 32;
    ++*(_QWORD *)(v26 + 24);
  }
  sub_16E7A90(v26, a7);
  if ( !*(_BYTE *)(*(_QWORD *)(a1 + 280) + 360LL) )
    goto LABEL_22;
  if ( (a9 & 2) == 0 )
  {
    if ( (a9 & 4) == 0 )
      goto LABEL_18;
LABEL_53:
    sub_1263B40(*(_QWORD *)(a1 + 272), " prologue_end");
    if ( (a9 & 8) == 0 )
      goto LABEL_19;
LABEL_54:
    sub_1263B40(*(_QWORD *)(a1 + 272), " epilogue_begin");
    goto LABEL_19;
  }
  sub_1263B40(*(_QWORD *)(a1 + 272), " basic_block");
  if ( (a9 & 4) != 0 )
    goto LABEL_53;
LABEL_18:
  if ( (a9 & 8) != 0 )
    goto LABEL_54;
LABEL_19:
  if ( (((unsigned __int8)a9 ^ *(_BYTE *)(*(_QWORD *)(a1 + 8) + 1034LL)) & 1) != 0 )
  {
    sub_1263B40(*(_QWORD *)(a1 + 272), " is_stmt ");
    v47 = *(_QWORD *)(a1 + 272);
    if ( (a9 & 1) != 0 )
      sub_1263B40(v47, "1");
    else
      sub_1263B40(v47, "0");
  }
  if ( !a10 )
  {
    if ( !a11 )
      goto LABEL_22;
    goto LABEL_51;
  }
  v48 = sub_1263B40(*(_QWORD *)(a1 + 272), " isa ");
  sub_16E7A90(v48, a10);
  if ( a11 )
  {
LABEL_51:
    v46 = sub_1263B40(*(_QWORD *)(a1 + 272), " discriminator ");
    sub_16E7A90(v46, a11);
  }
LABEL_22:
  if ( (*(_BYTE *)(a1 + 680) & 1) == 0 )
    goto LABEL_23;
  sub_16BE270(*(_QWORD *)(a1 + 272), 40);
  v36 = *(_QWORD *)(a1 + 280);
  v37 = *(_QWORD *)(a1 + 272);
  v38 = *(char **)(v36 + 48);
  v39 = *(_QWORD *)(v36 + 56);
  v40 = *(_QWORD *)(v37 + 16);
  v41 = *(_BYTE **)(v37 + 24);
  if ( v39 > v40 - (unsigned __int64)v41 )
  {
    v51 = sub_16E7EE0(*(_QWORD *)(a1 + 272), v38, v39);
    v41 = *(_BYTE **)(v51 + 24);
    v37 = v51;
    v40 = *(_QWORD *)(v51 + 16);
  }
  else if ( v39 )
  {
    v53 = v39;
    memcpy(v41, v38, v39);
    v49 = (_BYTE *)(*(_QWORD *)(v37 + 24) + v53);
    *(_QWORD *)(v37 + 24) = v49;
    v40 = *(_QWORD *)(v37 + 16);
    v41 = v49;
  }
  if ( v40 <= (unsigned __int64)v41 )
  {
    v37 = sub_16E7DE0(v37, 32);
  }
  else
  {
    *(_QWORD *)(v37 + 24) = v41 + 1;
    *v41 = 32;
  }
  v42 = *(_QWORD *)(v37 + 16);
  v43 = *(_BYTE **)(v37 + 24);
  if ( v42 - (unsigned __int64)v43 >= a13 )
  {
    if ( a13 )
    {
      memcpy(v43, a12, a13);
      v43 = (_BYTE *)(*(_QWORD *)(v37 + 24) + a13);
      *(_QWORD *)(v37 + 24) = v43;
      v42 = *(_QWORD *)(v37 + 16);
    }
    if ( v42 > (unsigned __int64)v43 )
      goto LABEL_43;
LABEL_61:
    v37 = sub_16E7DE0(v37, 58);
    goto LABEL_44;
  }
  v50 = sub_16E7EE0(v37, (char *)a12, a13);
  v43 = *(_BYTE **)(v50 + 24);
  v37 = v50;
  if ( *(_QWORD *)(v50 + 16) <= (unsigned __int64)v43 )
    goto LABEL_61;
LABEL_43:
  *(_QWORD *)(v37 + 24) = v43 + 1;
  *v43 = 58;
LABEL_44:
  v44 = sub_16E7A90(v37, a3);
  v45 = *(_BYTE **)(v44 + 24);
  if ( (unsigned __int64)v45 >= *(_QWORD *)(v44 + 16) )
  {
    v44 = sub_16E7DE0(v44, 58);
  }
  else
  {
    *(_QWORD *)(v44 + 24) = v45 + 1;
    *v45 = 58;
  }
  sub_16E7A90(v44, a4);
LABEL_23:
  v28 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v32 = *(_QWORD *)(a1 + 272);
    v33 = *(char **)(a1 + 304);
    v34 = *(unsigned int *)(a1 + 312);
    v35 = *(void **)(v32 + 24);
    if ( v28 > *(_QWORD *)(v32 + 16) - (_QWORD)v35 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v33, v34);
    }
    else
    {
      memcpy(v35, v33, v34);
      *(_QWORD *)(v32 + 24) += v28;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
  {
    sub_39E0440(a1);
  }
  else
  {
    v30 = *(_QWORD *)(a1 + 272);
    v31 = *(_BYTE **)(v30 + 24);
    if ( (unsigned __int64)v31 >= *(_QWORD *)(v30 + 16) )
    {
      sub_16E7DE0(v30, 10);
    }
    else
    {
      *(_QWORD *)(v30 + 24) = v31 + 1;
      *v31 = 10;
    }
  }
  return sub_38DBB60(a1, a2, a3, a4, a9, a10, a11);
}
