// Function: sub_3710150
// Address: 0x3710150
//
__int64 *__fastcall sub_3710150(__int64 *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r14
  bool v7; // zf
  unsigned __int64 v8; // rax
  __int64 *v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // rdx
  const char *v13; // r8
  _QWORD *v14; // rax
  size_t v15; // r15
  __int64 *v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rdx
  const char *v19; // r9
  _QWORD *v20; // rax
  size_t v21; // r15
  unsigned __int64 v22; // rcx
  __int8 *v23; // rsi
  __m128i *v24; // rax
  int v25; // eax
  __int64 *v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // rdx
  const char *v29; // r10
  _QWORD *v30; // rax
  size_t v31; // r8
  __int64 v32; // rcx
  unsigned int v33; // r8d
  __int64 v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // rdi
  __int64 v37; // rax
  _QWORD *v38; // rdi
  unsigned __int64 v39; // rax
  __int64 v40; // rax
  _QWORD *v41; // rdi
  const char *v42; // [rsp+0h] [rbp-1C0h]
  const char *src; // [rsp+8h] [rbp-1B8h]
  size_t n; // [rsp+10h] [rbp-1B0h]
  const char *v45; // [rsp+18h] [rbp-1A8h]
  unsigned __int16 v47; // [rsp+3Eh] [rbp-182h] BYREF
  __int64 v48[2]; // [rsp+40h] [rbp-180h] BYREF
  _QWORD v49[2]; // [rsp+50h] [rbp-170h] BYREF
  __int64 v50[2]; // [rsp+60h] [rbp-160h] BYREF
  _QWORD v51[2]; // [rsp+70h] [rbp-150h] BYREF
  unsigned __int64 v52[2]; // [rsp+80h] [rbp-140h] BYREF
  _QWORD v53[2]; // [rsp+90h] [rbp-130h] BYREF
  __int64 v54[2]; // [rsp+A0h] [rbp-120h] BYREF
  char v55; // [rsp+B0h] [rbp-110h] BYREF
  __m128i v56; // [rsp+C0h] [rbp-100h] BYREF
  __m128i v57; // [rsp+D0h] [rbp-F0h] BYREF
  __int16 v58; // [rsp+E0h] [rbp-E0h]
  __m128i v59; // [rsp+F0h] [rbp-D0h] BYREF
  __int64 v60; // [rsp+100h] [rbp-C0h]
  _DWORD v61[46]; // [rsp+108h] [rbp-B8h] BYREF

  v4 = a2 + 2;
  v7 = a2[9] == 0;
  v59.m128i_i64[0] = (__int64)v61;
  v60 = 128;
  qmemcpy(v61, "Attrs: ", 7);
  v59.m128i_i64[1] = 7;
  if ( v7 || a2[7] || a2[8] )
    goto LABEL_3;
  v10 = sub_3707A40();
  v13 = sub_370CA30(a2 + 2, *(_DWORD *)(a4 + 8) & 0x1F, (unsigned __int8 *)v10, v11);
  v14 = v49;
  v15 = v12;
  v48[0] = (__int64)v49;
  if ( &v13[v12] && !v13 )
    goto LABEL_48;
  v56.m128i_i64[0] = v12;
  if ( v12 > 0xF )
  {
    v45 = v13;
    v37 = sub_22409D0((__int64)v48, (unsigned __int64 *)&v56, 0);
    v13 = v45;
    v48[0] = v37;
    v38 = (_QWORD *)v37;
    v49[0] = v56.m128i_i64[0];
LABEL_64:
    memcpy(v38, v13, v15);
    v15 = v56.m128i_i64[0];
    v14 = (_QWORD *)v48[0];
    goto LABEL_20;
  }
  if ( v12 != 1 )
  {
    if ( !v12 )
      goto LABEL_20;
    v38 = v49;
    goto LABEL_64;
  }
  LOBYTE(v49[0]) = *v13;
LABEL_20:
  v48[1] = v15;
  *((_BYTE *)v14 + v15) = 0;
  sub_8FD6D0((__int64)&v56, "[ Type: ", v48);
  sub_C58CA0(&v59, v56.m128i_i64[0], (_BYTE *)(v56.m128i_i64[0] + v56.m128i_i64[1]));
  sub_2240A30((unsigned __int64 *)&v56);
  v16 = sub_3707A50();
  v19 = sub_370CA30(v4, *(_BYTE *)(a4 + 8) >> 5, (unsigned __int8 *)v16, v17);
  v20 = v51;
  v21 = v18;
  v50[0] = (__int64)v51;
  if ( &v19[v18] && !v19 )
    goto LABEL_48;
  v56.m128i_i64[0] = v18;
  if ( v18 > 0xF )
  {
    v42 = v19;
    v40 = sub_22409D0((__int64)v50, (unsigned __int64 *)&v56, 0);
    v19 = v42;
    v50[0] = v40;
    v41 = (_QWORD *)v40;
    v51[0] = v56.m128i_i64[0];
LABEL_80:
    memcpy(v41, v19, v21);
    v21 = v56.m128i_i64[0];
    v20 = (_QWORD *)v50[0];
    goto LABEL_25;
  }
  if ( v18 != 1 )
  {
    if ( !v18 )
      goto LABEL_25;
    v41 = v51;
    goto LABEL_80;
  }
  LOBYTE(v51[0]) = *v19;
LABEL_25:
  v50[1] = v21;
  *((_BYTE *)v20 + v21) = 0;
  sub_8FD6D0((__int64)&v56, ", Mode: ", v50);
  sub_C58CA0(&v59, v56.m128i_i64[0], (_BYTE *)(v56.m128i_i64[0] + v56.m128i_i64[1]));
  sub_2240A30((unsigned __int64 *)&v56);
  v22 = (unsigned __int8)(*(_DWORD *)(a4 + 8) >> 13);
  if ( (unsigned __int8)(*(_DWORD *)(a4 + 8) >> 13) )
  {
    v23 = &v57.m128i_i8[5];
    do
    {
      *--v23 = v22 % 0xA + 48;
      v39 = v22;
      v22 /= 0xAu;
    }
    while ( v39 > 9 );
  }
  else
  {
    v57.m128i_i8[4] = 48;
    v23 = &v57.m128i_i8[4];
  }
  v54[0] = (__int64)&v55;
  sub_370CBD0(v54, v23, (__int64)v57.m128i_i64 + 5);
  v24 = (__m128i *)sub_2241130((unsigned __int64 *)v54, 0, 0, ", SizeOf: ", 0xAu);
  v56.m128i_i64[0] = (__int64)&v57;
  if ( (__m128i *)v24->m128i_i64[0] == &v24[1] )
  {
    v57 = _mm_loadu_si128(v24 + 1);
  }
  else
  {
    v56.m128i_i64[0] = v24->m128i_i64[0];
    v57.m128i_i64[0] = v24[1].m128i_i64[0];
  }
  v56.m128i_i64[1] = v24->m128i_i64[1];
  v24->m128i_i64[0] = (__int64)v24[1].m128i_i64;
  v24->m128i_i64[1] = 0;
  v24[1].m128i_i8[0] = 0;
  sub_C58CA0(&v59, v56.m128i_i64[0], (_BYTE *)(v56.m128i_i64[0] + v56.m128i_i64[1]));
  sub_2240A30((unsigned __int64 *)&v56);
  sub_2240A30((unsigned __int64 *)v54);
  v25 = *(_DWORD *)(a4 + 8);
  if ( (v25 & 0x100) != 0 )
  {
    sub_C58CA0(&v59, ", isFlat", "");
    v25 = *(_DWORD *)(a4 + 8);
  }
  if ( (v25 & 0x400) != 0 )
  {
    sub_C58CA0(&v59, ", isConst", "");
    v25 = *(_DWORD *)(a4 + 8);
  }
  if ( (v25 & 0x200) != 0 )
  {
    sub_C58CA0(&v59, ", isVolatile", "");
    v25 = *(_DWORD *)(a4 + 8);
  }
  if ( (v25 & 0x800) != 0 )
  {
    sub_C58CA0(&v59, ", isUnaligned", "");
    v25 = *(_DWORD *)(a4 + 8);
  }
  if ( (v25 & 0x1000) != 0 )
  {
    sub_C58CA0(&v59, ", isRestricted", "");
    v25 = *(_DWORD *)(a4 + 8);
  }
  if ( (v25 & 0x100000) != 0 )
  {
    sub_C58CA0(&v59, ", isThisPtr&", "");
    v25 = *(_DWORD *)(a4 + 8);
  }
  if ( (v25 & 0x200000) != 0 )
    sub_C58CA0(&v59, ", isThisPtr&&", "");
  sub_C58CA0(&v59, " ]", "");
  sub_2240A30((unsigned __int64 *)v50);
  sub_2240A30((unsigned __int64 *)v48);
LABEL_3:
  v56.m128i_i64[0] = (__int64)"PointeeType";
  v58 = 259;
  sub_37011E0((unsigned __int64 *)v54, v4, (unsigned int *)(a4 + 2), v56.m128i_i64);
  v8 = v54[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v54[0] & 0xFFFFFFFFFFFFFFFELL) != 0
    || (v54[0] = 0,
        sub_9C66B0(v54),
        v58 = 261,
        v56 = v59,
        sub_370BDF0((unsigned __int64 *)v54, v4, (unsigned int *)(a4 + 8), &v56),
        v8 = v54[0] & 0xFFFFFFFFFFFFFFFELL,
        (v54[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
  {
    v54[0] = 0;
    *a1 = v8 | 1;
    sub_9C66B0(v54);
    goto LABEL_11;
  }
  v54[0] = 0;
  sub_9C66B0(v54);
  if ( (unsigned __int8)((*(_BYTE *)(a4 + 8) >> 5) - 2) > 1u )
    goto LABEL_45;
  if ( a2[7] && !a2[9] && !a2[8] )
  {
    *(_BYTE *)(a4 + 18) = 1;
    *(_DWORD *)(a4 + 12) = 0;
    *(_WORD *)(a4 + 16) = 0;
  }
  v56.m128i_i64[0] = (__int64)"ClassType";
  v58 = 259;
  sub_37011E0((unsigned __int64 *)v54, v4, (unsigned int *)(a4 + 12), v56.m128i_i64);
  if ( (v54[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v54[0] = v54[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    *a1 = 0;
    sub_9C6670(a1, v54);
    sub_9C66B0(v54);
    goto LABEL_11;
  }
  v54[0] = 0;
  sub_9C66B0(v54);
  v26 = sub_3707A60();
  v29 = sub_370CAA0(v4, *(_WORD *)(a4 + 16), v26, v27);
  v30 = v53;
  v31 = v28;
  v52[0] = (unsigned __int64)v53;
  if ( &v29[v28] && !v29 )
LABEL_48:
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v56.m128i_i64[0] = v28;
  if ( v28 > 0xF )
  {
    src = v29;
    n = v28;
    v35 = sub_22409D0((__int64)v52, (unsigned __int64 *)&v56, 0);
    v31 = n;
    v29 = src;
    v52[0] = v35;
    v36 = (_QWORD *)v35;
    v53[0] = v56.m128i_i64[0];
LABEL_62:
    memcpy(v36, v29, v31);
    v31 = v56.m128i_i64[0];
    v30 = (_QWORD *)v52[0];
    goto LABEL_52;
  }
  if ( v28 != 1 )
  {
    if ( !v28 )
      goto LABEL_52;
    v36 = v53;
    goto LABEL_62;
  }
  LOBYTE(v53[0]) = *v29;
LABEL_52:
  v52[1] = v31;
  *((_BYTE *)v30 + v31) = 0;
  sub_8FD6D0((__int64)v54, "Representation: ", v52);
  v7 = a2[9] == 0;
  v56.m128i_i64[0] = (__int64)v54;
  v58 = 260;
  if ( !v7 && !a2[7] && !a2[8] )
  {
LABEL_82:
    if ( !a2[7] )
      v47 = *(_WORD *)(a4 + 16);
    goto LABEL_57;
  }
  if ( (unsigned int)sub_3700ED0((__int64)v4, (__int64)"Representation: ", 260, v32, v33) <= 1 )
  {
    sub_370CCD0((unsigned __int64 *)v48, 2u);
    goto LABEL_59;
  }
  v34 = a2[9];
  if ( a2[8] )
  {
    if ( !v34 )
      goto LABEL_82;
  }
  else if ( v34 )
  {
    goto LABEL_82;
  }
LABEL_57:
  sub_370BC10((unsigned __int64 *)v50, v4, &v47, &v56);
  if ( (v50[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v48[0] = 0;
    v50[0] = v50[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    sub_9C6670(v48, v50);
    sub_9C66B0(v50);
  }
  else
  {
    v50[0] = 0;
    sub_9C66B0(v50);
    if ( a2[7] && !a2[9] && !a2[8] )
      *(_WORD *)(a4 + 16) = v47;
    v48[0] = 1;
    v50[0] = 0;
    sub_9C66B0(v50);
  }
LABEL_59:
  sub_2240A30((unsigned __int64 *)v54);
  if ( (v48[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    v48[0] = 0;
    sub_9C66B0(v48);
    sub_2240A30(v52);
LABEL_45:
    v56.m128i_i64[0] = 0;
    *a1 = 1;
    sub_9C66B0(v56.m128i_i64);
    goto LABEL_11;
  }
  v48[0] = v48[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  *a1 = 0;
  sub_9C6670(a1, v48);
  sub_9C66B0(v48);
  sub_2240A30(v52);
LABEL_11:
  if ( (_DWORD *)v59.m128i_i64[0] != v61 )
    _libc_free(v59.m128i_u64[0]);
  return a1;
}
