// Function: sub_12636E0
// Address: 0x12636e0
//
__int64 __fastcall sub_12636E0(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rcx
  char *v9; // r15
  size_t v10; // rax
  _QWORD *v11; // rdi
  size_t v12; // r13
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  const char *v16; // r15
  size_t v17; // rax
  _WORD *v18; // rdi
  size_t v19; // r13
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  const char *v24; // r15
  size_t v25; // rax
  _BYTE *v26; // rdi
  size_t v27; // r13
  _BYTE *v28; // rax
  _BYTE *v29; // rax
  __int64 v30; // rbx
  __int64 v31; // rax
  char *v32; // r15
  size_t v33; // rax
  size_t v34; // r13
  __int64 v35; // r12
  _BYTE *v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rdi
  _BYTE *v39; // rax
  __m128i *v40; // rdx
  __m128i si128; // xmm0
  __int64 v42; // rax
  __int64 v43; // rdx

  v5 = sub_16E8C20(a1, a2, a3, a4);
  v6 = *(_QWORD *)(v5 + 24);
  v7 = v5;
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v6) <= 2 )
  {
    a2 = "[ \"";
    v7 = sub_16E7EE0(v5, "[ \"", 3);
  }
  else
  {
    v8 = 8283;
    *(_BYTE *)(v6 + 2) = 34;
    *(_WORD *)v6 = 8283;
    *(_QWORD *)(v5 + 24) += 3LL;
  }
  v9 = *(char **)(a3 + 16);
  if ( !v9 )
    goto LABEL_26;
  v10 = strlen(*(const char **)(a3 + 16));
  v11 = *(_QWORD **)(v7 + 24);
  v12 = v10;
  v13 = *(_QWORD *)(v7 + 16) - (_QWORD)v11;
  if ( v12 > v13 )
  {
    a2 = v9;
    v7 = sub_16E7EE0(v7, v9, v12);
LABEL_26:
    v11 = *(_QWORD **)(v7 + 24);
    v13 = *(_QWORD *)(v7 + 16) - (_QWORD)v11;
    goto LABEL_27;
  }
  if ( v12 )
  {
    a2 = v9;
    memcpy(v11, v9, v12);
    v11 = (_QWORD *)(v12 + *(_QWORD *)(v7 + 24));
    v14 = *(_QWORD *)(v7 + 16) - (_QWORD)v11;
    *(_QWORD *)(v7 + 24) = v11;
    if ( v14 <= 7 )
      goto LABEL_7;
    goto LABEL_28;
  }
LABEL_27:
  if ( v13 <= 7 )
  {
LABEL_7:
    a2 = "\" -llc \"";
    v15 = sub_16E7EE0(v7, "\" -llc \"", 8);
    v16 = *(const char **)(a3 + 32);
    v7 = v15;
    if ( v16 )
      goto LABEL_8;
LABEL_29:
    v18 = *(_WORD **)(v7 + 24);
    v20 = *(_QWORD *)(v7 + 16) - (_QWORD)v18;
    goto LABEL_30;
  }
LABEL_28:
  *v11 = 0x2220636C6C2D2022LL;
  *(_QWORD *)(v7 + 24) += 8LL;
  v16 = *(const char **)(a3 + 32);
  if ( !v16 )
    goto LABEL_29;
LABEL_8:
  v17 = strlen(v16);
  v18 = *(_WORD **)(v7 + 24);
  v19 = v17;
  v20 = *(_QWORD *)(v7 + 16) - (_QWORD)v18;
  if ( v19 > v20 )
  {
    a2 = (char *)v16;
    v7 = sub_16E7EE0(v7, v16, v19);
    goto LABEL_29;
  }
  if ( v19 )
  {
    a2 = (char *)v16;
    memcpy(v18, v16, v19);
    v18 = (_WORD *)(v19 + *(_QWORD *)(v7 + 24));
    v21 = *(_QWORD *)(v7 + 16) - (_QWORD)v18;
    *(_QWORD *)(v7 + 24) = v18;
    if ( v21 <= 5 )
      goto LABEL_11;
    goto LABEL_31;
  }
LABEL_30:
  if ( v20 <= 5 )
  {
LABEL_11:
    a2 = "\" -o \"";
    v22 = sub_16E7EE0(v7, "\" -o \"", 6);
    v24 = *(const char **)(a3 + 8);
    v7 = v22;
    if ( v24 )
      goto LABEL_12;
LABEL_32:
    v28 = *(_BYTE **)(v7 + 16);
    v26 = *(_BYTE **)(v7 + 24);
    goto LABEL_33;
  }
LABEL_31:
  v23 = 8736;
  *(_DWORD *)v18 = 1865228322;
  v18[2] = 8736;
  *(_QWORD *)(v7 + 24) += 6LL;
  v24 = *(const char **)(a3 + 8);
  if ( !v24 )
    goto LABEL_32;
LABEL_12:
  v25 = strlen(v24);
  v26 = *(_BYTE **)(v7 + 24);
  v27 = v25;
  v28 = *(_BYTE **)(v7 + 16);
  v23 = v28 - v26;
  if ( v27 > v28 - v26 )
  {
    a2 = (char *)v24;
    v7 = sub_16E7EE0(v7, v24, v27);
    goto LABEL_32;
  }
  if ( v27 )
  {
    a2 = (char *)v24;
    memcpy(v26, v24, v27);
    v29 = *(_BYTE **)(v7 + 16);
    v26 = (_BYTE *)(v27 + *(_QWORD *)(v7 + 24));
    *(_QWORD *)(v7 + 24) = v26;
    if ( v29 == v26 )
      goto LABEL_15;
    goto LABEL_34;
  }
LABEL_33:
  if ( v28 == v26 )
  {
LABEL_15:
    a2 = "\"";
    v26 = (_BYTE *)v7;
    v30 = 0;
    sub_16E7EE0(v7, "\"", 1);
    if ( *(int *)(a3 + 52) <= 1 )
      goto LABEL_35;
    goto LABEL_23;
  }
LABEL_34:
  *v26 = 34;
  v30 = 0;
  ++*(_QWORD *)(v7 + 24);
  if ( *(int *)(a3 + 52) <= 1 )
    goto LABEL_35;
  do
  {
LABEL_23:
    v35 = sub_16E8C20(v26, a2, v23, v8);
    v36 = *(_BYTE **)(v35 + 24);
    if ( (unsigned __int64)v36 < *(_QWORD *)(v35 + 16) )
    {
      v23 = (__int64)(v36 + 1);
      *(_QWORD *)(v35 + 24) = v36 + 1;
      *v36 = 32;
    }
    else
    {
      v26 = (_BYTE *)v35;
      a2 = (char *)32;
      v35 = sub_16E7DE0(v35, 32);
    }
    v31 = *(_QWORD *)(a3 + 80);
    ++v30;
    v32 = *(char **)(v31 + 8 * v30);
    if ( v32 )
    {
      v33 = strlen(*(const char **)(v31 + 8 * v30));
      v26 = *(_BYTE **)(v35 + 24);
      v34 = v33;
      if ( v33 > *(_QWORD *)(v35 + 16) - (_QWORD)v26 )
      {
        a2 = v32;
        v26 = (_BYTE *)v35;
        sub_16E7EE0(v35, v32, v33);
      }
      else if ( v33 )
      {
        a2 = v32;
        memcpy(v26, v32, v33);
        *(_QWORD *)(v35 + 24) += v34;
      }
    }
  }
  while ( *(_DWORD *)(a3 + 52) > (int)v30 + 1 );
LABEL_35:
  v38 = sub_16E8C20(v26, a2, v23, v8);
  v39 = *(_BYTE **)(v38 + 24);
  if ( (unsigned __int64)v39 >= *(_QWORD *)(v38 + 16) )
  {
    a2 = (char *)32;
    v38 = sub_16E7DE0(v38, 32);
  }
  else
  {
    *(_QWORD *)(v38 + 24) = v39 + 1;
    *v39 = 32;
  }
  v40 = *(__m128i **)(v38 + 24);
  if ( *(_QWORD *)(v38 + 16) - (_QWORD)v40 <= 0x13u )
  {
    a2 = "-nvvm-version=nvvm70";
    sub_16E7EE0(v38, "-nvvm-version=nvvm70", 20);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3C23BC0);
    v40[1].m128i_i32[0] = 808938870;
    *v40 = si128;
    *(_QWORD *)(v38 + 24) += 20LL;
  }
  v42 = sub_16E8C20(v38, a2, v40, v37);
  v43 = *(_QWORD *)(v42 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v42 + 16) - v43) <= 2 )
    return sub_16E7EE0(v42, " ]\n", 3);
  *(_BYTE *)(v43 + 2) = 10;
  *(_WORD *)v43 = 23840;
  *(_QWORD *)(v42 + 24) += 3LL;
  return 23840;
}
