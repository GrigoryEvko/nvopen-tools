// Function: sub_39ED450
// Address: 0x39ed450
//
_QWORD *__fastcall sub_39ED450(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        char a6,
        char a7,
        char *a8,
        unsigned __int64 a9,
        unsigned __int64 a10)
{
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __int64 v19; // rdi
  _BYTE *v20; // rax
  size_t v21; // r14
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // r8
  char *v28; // rsi
  size_t v29; // rdx
  unsigned __int64 v30; // rax
  _BYTE *v31; // rdi
  unsigned __int64 v32; // rax
  _BYTE *v33; // rdi
  __int64 v34; // rdi
  _BYTE *v35; // rax
  __int64 v36; // r15
  char *v37; // rsi
  void *v38; // rdi
  __int64 v39; // rdi
  _BYTE *v40; // rax
  _BYTE *v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // [rsp+8h] [rbp-68h]
  __int64 v45; // [rsp+10h] [rbp-60h]
  size_t v46; // [rsp+10h] [rbp-60h]

  v13 = *(_QWORD *)(a1 + 272);
  v14 = *(_QWORD *)(v13 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v13 + 16) - v14) <= 8 )
  {
    v13 = sub_16E7EE0(v13, "\t.cv_loc\t", 9u);
  }
  else
  {
    *(_BYTE *)(v14 + 8) = 9;
    *(_QWORD *)v14 = 0x636F6C5F76632E09LL;
    *(_QWORD *)(v13 + 24) += 9LL;
  }
  v15 = sub_16E7A90(v13, a2);
  v16 = *(_BYTE **)(v15 + 24);
  if ( *(_BYTE **)(v15 + 16) == v16 )
  {
    v15 = sub_16E7EE0(v15, " ", 1u);
  }
  else
  {
    *v16 = 32;
    ++*(_QWORD *)(v15 + 24);
  }
  v17 = sub_16E7A90(v15, a3);
  v18 = *(_BYTE **)(v17 + 24);
  if ( *(_BYTE **)(v17 + 16) == v18 )
  {
    v17 = sub_16E7EE0(v17, " ", 1u);
  }
  else
  {
    *v18 = 32;
    ++*(_QWORD *)(v17 + 24);
  }
  v19 = sub_16E7A90(v17, a4);
  v20 = *(_BYTE **)(v19 + 24);
  if ( *(_BYTE **)(v19 + 16) == v20 )
  {
    v19 = sub_16E7EE0(v19, " ", 1u);
  }
  else
  {
    *v20 = 32;
    ++*(_QWORD *)(v19 + 24);
  }
  sub_16E7A90(v19, a5);
  if ( a6 )
    sub_1263B40(*(_QWORD *)(a1 + 272), " prologue_end");
  if ( a7 == ((*(_BYTE *)(sub_38BE350(*(_QWORD *)(a1 + 8)) + 14) & 2) != 0) )
  {
LABEL_12:
    if ( (*(_BYTE *)(a1 + 680) & 1) == 0 )
      goto LABEL_13;
    goto LABEL_21;
  }
  v23 = *(_QWORD *)(a1 + 272);
  v24 = *(_QWORD *)(v23 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v23 + 16) - v24) <= 8 )
  {
    sub_16E7EE0(v23, " is_stmt ", 9u);
  }
  else
  {
    *(_BYTE *)(v24 + 8) = 32;
    *(_QWORD *)v24 = 0x746D74735F736920LL;
    *(_QWORD *)(v23 + 24) += 9LL;
  }
  v25 = *(_QWORD *)(a1 + 272);
  if ( !a7 )
  {
    sub_1263B40(v25, "0");
    goto LABEL_12;
  }
  sub_1263B40(v25, "1");
  if ( (*(_BYTE *)(a1 + 680) & 1) == 0 )
  {
LABEL_13:
    v21 = *(unsigned int *)(a1 + 312);
    if ( !*(_DWORD *)(a1 + 312) )
      goto LABEL_14;
LABEL_34:
    v36 = *(_QWORD *)(a1 + 272);
    v37 = *(char **)(a1 + 304);
    v38 = *(void **)(v36 + 24);
    if ( v21 > *(_QWORD *)(v36 + 16) - (_QWORD)v38 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v37, v21);
    }
    else
    {
      memcpy(v38, v37, v21);
      *(_QWORD *)(v36 + 24) += v21;
    }
    goto LABEL_14;
  }
LABEL_21:
  sub_16BE270(*(_QWORD *)(a1 + 272), 40);
  v26 = *(_QWORD *)(a1 + 280);
  v27 = *(_QWORD *)(a1 + 272);
  v28 = *(char **)(v26 + 48);
  v29 = *(_QWORD *)(v26 + 56);
  v30 = *(_QWORD *)(v27 + 16);
  v31 = *(_BYTE **)(v27 + 24);
  if ( v29 > v30 - (unsigned __int64)v31 )
  {
    v43 = sub_16E7EE0(*(_QWORD *)(a1 + 272), v28, v29);
    v31 = *(_BYTE **)(v43 + 24);
    v27 = v43;
    v30 = *(_QWORD *)(v43 + 16);
  }
  else if ( v29 )
  {
    v44 = *(_QWORD *)(a1 + 272);
    v46 = v29;
    memcpy(v31, v28, v29);
    v27 = v44;
    v41 = (_BYTE *)(*(_QWORD *)(v44 + 24) + v46);
    v30 = *(_QWORD *)(v44 + 16);
    *(_QWORD *)(v44 + 24) = v41;
    v31 = v41;
  }
  if ( (unsigned __int64)v31 >= v30 )
  {
    v27 = sub_16E7DE0(v27, 32);
  }
  else
  {
    *(_QWORD *)(v27 + 24) = v31 + 1;
    *v31 = 32;
  }
  v32 = *(_QWORD *)(v27 + 16);
  v33 = *(_BYTE **)(v27 + 24);
  if ( v32 - (unsigned __int64)v33 < a9 )
  {
    v42 = sub_16E7EE0(v27, a8, a9);
    v33 = *(_BYTE **)(v42 + 24);
    v27 = v42;
    v32 = *(_QWORD *)(v42 + 16);
  }
  else if ( a9 )
  {
    v45 = v27;
    memcpy(v33, a8, a9);
    v27 = v45;
    v33 = (_BYTE *)(*(_QWORD *)(v45 + 24) + a9);
    v32 = *(_QWORD *)(v45 + 16);
    *(_QWORD *)(v45 + 24) = v33;
  }
  if ( (unsigned __int64)v33 >= v32 )
  {
    v27 = sub_16E7DE0(v27, 58);
  }
  else
  {
    *(_QWORD *)(v27 + 24) = v33 + 1;
    *v33 = 58;
  }
  v34 = sub_16E7A90(v27, a4);
  v35 = *(_BYTE **)(v34 + 24);
  if ( (unsigned __int64)v35 >= *(_QWORD *)(v34 + 16) )
  {
    v34 = sub_16E7DE0(v34, 58);
  }
  else
  {
    *(_QWORD *)(v34 + 24) = v35 + 1;
    *v35 = 58;
  }
  sub_16E7A90(v34, a5);
  v21 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
    goto LABEL_34;
LABEL_14:
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
  {
    sub_39E0440(a1);
  }
  else
  {
    v39 = *(_QWORD *)(a1 + 272);
    v40 = *(_BYTE **)(v39 + 24);
    if ( (unsigned __int64)v40 >= *(_QWORD *)(v39 + 16) )
    {
      sub_16E7DE0(v39, 10);
    }
    else
    {
      *(_QWORD *)(v39 + 24) = v40 + 1;
      *v40 = 10;
    }
  }
  return sub_38DC620(a1, a2, a3, a4, a5, a6, a7, (int)a8, a9, a10);
}
