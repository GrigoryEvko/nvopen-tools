// Function: sub_39EE280
// Address: 0x39ee280
//
__int64 __fastcall sub_39EE280(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        char a5,
        unsigned int a6,
        unsigned int a7,
        void *a8,
        size_t a9)
{
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v18; // rdi
  _BYTE *v19; // rax
  size_t v20; // rdx
  __int64 v22; // rdi
  _BYTE *v23; // rax
  __int64 v24; // r8
  char *v25; // rsi
  void *v26; // rdi
  __int64 v27; // rax
  __int64 v28; // r8
  char *v29; // rsi
  size_t v30; // rdx
  unsigned __int64 v31; // rax
  _BYTE *v32; // rdi
  unsigned __int64 v33; // rax
  _BYTE *v34; // rdi
  __int64 v35; // rdi
  _BYTE *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rax
  _BYTE *v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // [rsp+8h] [rbp-68h]
  __int64 v44; // [rsp+10h] [rbp-60h]
  size_t v45; // [rsp+10h] [rbp-60h]
  __int64 v46; // [rsp+18h] [rbp-58h]
  __int64 v47; // [rsp+20h] [rbp-50h]

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
  if ( !*(_BYTE *)(*(_QWORD *)(a1 + 280) + 360LL) )
    goto LABEL_14;
  if ( (a5 & 2) == 0 )
  {
    if ( (a5 & 4) == 0 )
      goto LABEL_10;
LABEL_41:
    sub_1263B40(*(_QWORD *)(a1 + 272), " prologue_end");
    if ( (a5 & 8) == 0 )
      goto LABEL_11;
LABEL_42:
    sub_1263B40(*(_QWORD *)(a1 + 272), " epilogue_begin");
    goto LABEL_11;
  }
  sub_1263B40(*(_QWORD *)(a1 + 272), " basic_block");
  if ( (a5 & 4) != 0 )
    goto LABEL_41;
LABEL_10:
  if ( (a5 & 8) != 0 )
    goto LABEL_42;
LABEL_11:
  if ( (((unsigned __int8)a5 ^ *(_BYTE *)(*(_QWORD *)(a1 + 8) + 1034LL)) & 1) != 0 )
  {
    sub_1263B40(*(_QWORD *)(a1 + 272), " is_stmt ");
    v38 = *(_QWORD *)(a1 + 272);
    if ( (a5 & 1) != 0 )
      sub_1263B40(v38, "1");
    else
      sub_1263B40(v38, "0");
  }
  if ( !a6 )
  {
    if ( !a7 )
      goto LABEL_14;
    goto LABEL_39;
  }
  v39 = sub_1263B40(*(_QWORD *)(a1 + 272), " isa ");
  sub_16E7A90(v39, a6);
  if ( a7 )
  {
LABEL_39:
    v37 = sub_1263B40(*(_QWORD *)(a1 + 272), " discriminator ");
    sub_16E7A90(v37, a7);
  }
LABEL_14:
  if ( (*(_BYTE *)(a1 + 680) & 1) == 0 )
    goto LABEL_15;
  sub_16BE270(*(_QWORD *)(a1 + 272), 40);
  v27 = *(_QWORD *)(a1 + 280);
  v28 = *(_QWORD *)(a1 + 272);
  v29 = *(char **)(v27 + 48);
  v30 = *(_QWORD *)(v27 + 56);
  v31 = *(_QWORD *)(v28 + 16);
  v32 = *(_BYTE **)(v28 + 24);
  if ( v30 > v31 - (unsigned __int64)v32 )
  {
    v42 = sub_16E7EE0(*(_QWORD *)(a1 + 272), v29, v30);
    v32 = *(_BYTE **)(v42 + 24);
    v28 = v42;
    v31 = *(_QWORD *)(v42 + 16);
  }
  else if ( v30 )
  {
    v43 = *(_QWORD *)(a1 + 272);
    v45 = v30;
    memcpy(v32, v29, v30);
    v28 = v43;
    v40 = (_BYTE *)(*(_QWORD *)(v43 + 24) + v45);
    v31 = *(_QWORD *)(v43 + 16);
    *(_QWORD *)(v43 + 24) = v40;
    v32 = v40;
  }
  if ( (unsigned __int64)v32 >= v31 )
  {
    v28 = sub_16E7DE0(v28, 32);
  }
  else
  {
    *(_QWORD *)(v28 + 24) = v32 + 1;
    *v32 = 32;
  }
  v33 = *(_QWORD *)(v28 + 16);
  v34 = *(_BYTE **)(v28 + 24);
  if ( v33 - (unsigned __int64)v34 >= a9 )
  {
    if ( a9 )
    {
      v44 = v28;
      memcpy(v34, a8, a9);
      v28 = v44;
      v34 = (_BYTE *)(*(_QWORD *)(v44 + 24) + a9);
      v33 = *(_QWORD *)(v44 + 16);
      *(_QWORD *)(v44 + 24) = v34;
    }
    if ( v33 > (unsigned __int64)v34 )
      goto LABEL_35;
LABEL_49:
    v28 = sub_16E7DE0(v28, 58);
    goto LABEL_36;
  }
  v41 = sub_16E7EE0(v28, (char *)a8, a9);
  v34 = *(_BYTE **)(v41 + 24);
  v28 = v41;
  if ( *(_QWORD *)(v41 + 16) <= (unsigned __int64)v34 )
    goto LABEL_49;
LABEL_35:
  *(_QWORD *)(v28 + 24) = v34 + 1;
  *v34 = 58;
LABEL_36:
  v35 = sub_16E7A90(v28, a3);
  v36 = *(_BYTE **)(v35 + 24);
  if ( (unsigned __int64)v36 >= *(_QWORD *)(v35 + 16) )
  {
    v35 = sub_16E7DE0(v35, 58);
  }
  else
  {
    *(_QWORD *)(v35 + 24) = v36 + 1;
    *v36 = 58;
  }
  sub_16E7A90(v35, a4);
LABEL_15:
  v20 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v24 = *(_QWORD *)(a1 + 272);
    v25 = *(char **)(a1 + 304);
    v26 = *(void **)(v24 + 24);
    if ( v20 > *(_QWORD *)(v24 + 16) - (_QWORD)v26 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v25, v20);
    }
    else
    {
      v46 = *(_QWORD *)(a1 + 272);
      v47 = *(unsigned int *)(a1 + 312);
      memcpy(v26, v25, v20);
      *(_QWORD *)(v46 + 24) += v47;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
  {
    sub_39E0440(a1);
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 272);
    v23 = *(_BYTE **)(v22 + 24);
    if ( (unsigned __int64)v23 >= *(_QWORD *)(v22 + 16) )
    {
      sub_16E7DE0(v22, 10);
    }
    else
    {
      *(_QWORD *)(v22 + 24) = v23 + 1;
      *v23 = 10;
    }
  }
  return sub_38DBB60(a1, a2, a3, a4, a5, a6, a7);
}
