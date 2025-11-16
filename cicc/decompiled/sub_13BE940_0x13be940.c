// Function: sub_13BE940
// Address: 0x13be940
//
__int64 __fastcall sub_13BE940(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  _BYTE *v6; // rax
  __int64 v7; // rdx
  _BYTE *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rcx
  __m128i *v14; // rax
  __m128i *v15; // rcx
  __m128i *v16; // rdx
  __int64 v17; // rcx
  char *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rax
  _DWORD *v29; // rdx
  __int64 v30; // rax
  _BYTE *v31; // rdx
  __int64 v32; // rdx
  _BYTE *v33; // rsi
  __int64 v34; // rcx
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rdi
  char *v37; // rcx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 *v40; // rcx
  __int64 v41; // rcx
  const char *v42; // rsi
  __m128i *v43; // rax
  __m128i *v44; // rdx
  __int64 v45; // rdi
  __int64 v46; // rdi
  __m128i *v47; // rdx
  __int64 v48; // rdi
  __int64 v49; // rdi
  _BYTE *v50; // rax
  __int64 v52; // rax
  __m128i si128; // xmm0
  __int64 v54; // [rsp+38h] [rbp-198h] BYREF
  int v55; // [rsp+40h] [rbp-190h] BYREF
  __int64 v56; // [rsp+48h] [rbp-188h]
  __m128i *v57; // [rsp+50h] [rbp-180h]
  __int64 v58; // [rsp+58h] [rbp-178h]
  __m128i v59; // [rsp+60h] [rbp-170h] BYREF
  _QWORD *v60; // [rsp+70h] [rbp-160h] BYREF
  __int64 v61; // [rsp+78h] [rbp-158h]
  _QWORD v62[2]; // [rsp+80h] [rbp-150h] BYREF
  _QWORD *v63; // [rsp+90h] [rbp-140h] BYREF
  __int64 v64; // [rsp+98h] [rbp-138h]
  _QWORD v65[2]; // [rsp+A0h] [rbp-130h] BYREF
  _BYTE *v66; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v67; // [rsp+B8h] [rbp-118h]
  _QWORD v68[2]; // [rsp+C0h] [rbp-110h] BYREF
  __m128i *v69; // [rsp+D0h] [rbp-100h] BYREF
  __int64 v70; // [rsp+D8h] [rbp-F8h]
  __m128i v71; // [rsp+E0h] [rbp-F0h] BYREF
  __m128i **v72; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v73; // [rsp+F8h] [rbp-D8h]
  _QWORD v74[2]; // [rsp+100h] [rbp-D0h] BYREF
  _QWORD *v75; // [rsp+110h] [rbp-C0h] BYREF
  __int64 *v76; // [rsp+118h] [rbp-B8h]
  _QWORD v77[2]; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v78; // [rsp+130h] [rbp-A0h] BYREF
  __int64 v79; // [rsp+138h] [rbp-98h]
  __m128i v80; // [rsp+140h] [rbp-90h] BYREF
  _BYTE v81[128]; // [rsp+150h] [rbp-80h] BYREF

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_81:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F99CCC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_81;
  }
  v54 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F99CCC)
      + 160;
  v6 = (_BYTE *)sub_1649960(a2);
  if ( v6 )
  {
    v63 = v65;
    sub_13B5840((__int64 *)&v63, v6, (__int64)&v6[v7]);
  }
  else
  {
    LOBYTE(v65[0]) = 0;
    v63 = v65;
    v64 = 0;
  }
  v8 = (_BYTE *)a1[20];
  v9 = a1[21];
  v60 = v62;
  sub_13B5790((__int64 *)&v60, v8, (__int64)&v8[v9]);
  if ( v61 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_80:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v60, ".", 1, v10);
  v11 = 15;
  v12 = 15;
  if ( v60 != v62 )
    v12 = v62[0];
  v13 = v61 + v64;
  if ( v61 + v64 > v12 )
  {
    if ( v63 != v65 )
      v11 = v65[0];
    if ( v13 <= v11 )
    {
      v14 = (__m128i *)sub_2241130(&v63, 0, 0, v60, v61);
      v69 = &v71;
      v15 = (__m128i *)v14->m128i_i64[0];
      v16 = v14 + 1;
      if ( (__m128i *)v14->m128i_i64[0] != &v14[1] )
        goto LABEL_15;
LABEL_77:
      v71 = _mm_loadu_si128(v14 + 1);
      goto LABEL_16;
    }
  }
  v14 = (__m128i *)sub_2241490(&v60, v63, v64, v13);
  v69 = &v71;
  v15 = (__m128i *)v14->m128i_i64[0];
  v16 = v14 + 1;
  if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
    goto LABEL_77;
LABEL_15:
  v69 = v15;
  v71.m128i_i64[0] = v14[1].m128i_i64[0];
LABEL_16:
  v17 = v14->m128i_i64[1];
  v70 = v17;
  v14->m128i_i64[0] = (__int64)v16;
  v14->m128i_i64[1] = 0;
  v14[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v70) <= 3 )
    goto LABEL_80;
  v18 = ".dot";
  v19 = sub_2241490(&v69, ".dot", 4, v17);
  v21 = v19 + 16;
  v57 = &v59;
  if ( *(_QWORD *)v19 == v19 + 16 )
  {
    v59 = _mm_loadu_si128((const __m128i *)(v19 + 16));
  }
  else
  {
    v57 = *(__m128i **)v19;
    v59.m128i_i64[0] = *(_QWORD *)(v19 + 16);
  }
  v22 = *(_QWORD *)(v19 + 8);
  *(_BYTE *)(v19 + 16) = 0;
  v58 = v22;
  *(_QWORD *)v19 = v21;
  *(_QWORD *)(v19 + 8) = 0;
  if ( v69 != &v71 )
  {
    v18 = (char *)(v71.m128i_i64[0] + 1);
    j_j___libc_free_0(v69, v71.m128i_i64[0] + 1);
  }
  if ( v60 != v62 )
  {
    v18 = (char *)(v62[0] + 1LL);
    j_j___libc_free_0(v60, v62[0] + 1LL);
  }
  v23 = (__int64)v63;
  if ( v63 != v65 )
  {
    v18 = (char *)(v65[0] + 1LL);
    j_j___libc_free_0(v63, v65[0] + 1LL);
  }
  v55 = 0;
  v56 = sub_2241E40(v23, v18, v21, v22, v20);
  v25 = sub_16E8CB0(v23, v18, v24);
  v26 = *(_QWORD *)(v25 + 24);
  v27 = v25;
  if ( (unsigned __int64)(*(_QWORD *)(v25 + 16) - v26) <= 8 )
  {
    v27 = sub_16E7EE0(v25, "Writing '", 9);
  }
  else
  {
    *(_BYTE *)(v26 + 8) = 39;
    *(_QWORD *)v26 = 0x20676E6974697257LL;
    *(_QWORD *)(v25 + 24) += 9LL;
  }
  v28 = sub_16E7EE0(v27, v57->m128i_i8, v58);
  v29 = *(_DWORD **)(v28 + 24);
  if ( *(_QWORD *)(v28 + 16) - (_QWORD)v29 <= 3u )
  {
    sub_16E7EE0(v28, "'...", 4);
  }
  else
  {
    *v29 = 774778407;
    *(_QWORD *)(v28 + 24) += 4LL;
  }
  sub_16E8AF0(v81, v57, v58, &v55, 1);
  v66 = v68;
  v78 = 19;
  v30 = sub_22409D0(&v66, &v78, 0);
  v66 = (_BYTE *)v30;
  v68[0] = v78;
  *(__m128i *)v30 = _mm_load_si128((const __m128i *)&xmmword_4289C10);
  v31 = v66;
  *(_WORD *)(v30 + 16) = 25970;
  *(_BYTE *)(v30 + 18) = 101;
  v67 = v78;
  v31[v78] = 0;
  v33 = (_BYTE *)sub_1649960(a2);
  if ( v33 )
  {
    v75 = v77;
    sub_13B5840((__int64 *)&v75, v33, (__int64)&v33[v32]);
  }
  else
  {
    LOBYTE(v77[0]) = 0;
    v75 = v77;
    v76 = 0;
  }
  v72 = (__m128i **)v74;
  sub_13B5790((__int64 *)&v72, v66, (__int64)&v66[v67]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v73) <= 5 )
    goto LABEL_80;
  sub_2241490(&v72, " for '", 6, v34);
  v35 = 15;
  v36 = 15;
  if ( v72 != v74 )
    v36 = v74[0];
  v37 = (char *)v76 + v73;
  if ( (unsigned __int64)v76 + v73 <= v36 )
    goto LABEL_38;
  if ( v75 != v77 )
    v35 = v77[0];
  if ( (unsigned __int64)v37 <= v35 )
  {
    v38 = sub_2241130(&v75, 0, 0, v72, v73);
    v39 = v38 + 16;
    v78 = (__int64)&v80;
    v40 = *(__int64 **)v38;
    if ( *(_QWORD *)v38 != v38 + 16 )
      goto LABEL_39;
  }
  else
  {
LABEL_38:
    v38 = sub_2241490(&v72, v75, v76, v37);
    v39 = v38 + 16;
    v78 = (__int64)&v80;
    v40 = *(__int64 **)v38;
    if ( *(_QWORD *)v38 != v38 + 16 )
    {
LABEL_39:
      v78 = (__int64)v40;
      v80.m128i_i64[0] = *(_QWORD *)(v38 + 16);
      goto LABEL_40;
    }
  }
  v80 = _mm_loadu_si128((const __m128i *)(v38 + 16));
LABEL_40:
  v41 = *(_QWORD *)(v38 + 8);
  v79 = v41;
  *(_QWORD *)v38 = v39;
  *(_QWORD *)(v38 + 8) = 0;
  *(_BYTE *)(v38 + 16) = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v79) <= 9 )
    goto LABEL_80;
  v42 = "' function";
  v43 = (__m128i *)sub_2241490(&v78, "' function", 10, v41);
  v69 = &v71;
  v44 = v43 + 1;
  if ( (__m128i *)v43->m128i_i64[0] == &v43[1] )
  {
    v71 = _mm_loadu_si128(v43 + 1);
  }
  else
  {
    v69 = (__m128i *)v43->m128i_i64[0];
    v71.m128i_i64[0] = v43[1].m128i_i64[0];
  }
  v70 = v43->m128i_i64[1];
  v43->m128i_i64[0] = (__int64)v44;
  v43->m128i_i64[1] = 0;
  v43[1].m128i_i8[0] = 0;
  if ( (__m128i *)v78 != &v80 )
  {
    v42 = (const char *)(v80.m128i_i64[0] + 1);
    j_j___libc_free_0(v78, v80.m128i_i64[0] + 1);
  }
  if ( v72 != v74 )
  {
    v42 = (const char *)(v74[0] + 1LL);
    j_j___libc_free_0(v72, v74[0] + 1LL);
  }
  v45 = (__int64)v75;
  if ( v75 != v77 )
  {
    v42 = (const char *)(v77[0] + 1LL);
    j_j___libc_free_0(v75, v77[0] + 1LL);
  }
  if ( v55 )
  {
    v52 = sub_16E8CB0(v45, v42, v44);
    v47 = *(__m128i **)(v52 + 24);
    v48 = v52;
    if ( *(_QWORD *)(v52 + 16) - (_QWORD)v47 <= 0x20u )
    {
      v42 = "  error opening file for writing!";
      sub_16E7EE0(v52, "  error opening file for writing!", 33);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
      v47[2].m128i_i8[0] = 33;
      *v47 = si128;
      v47[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
      *(_QWORD *)(v52 + 24) += 33LL;
    }
  }
  else
  {
    LOWORD(v74[0]) = 260;
    v72 = &v69;
    v76 = &v54;
    v75 = v81;
    LOBYTE(v77[0]) = 1;
    sub_16E2FC0(&v78, &v72);
    v42 = (const char *)&v78;
    sub_13B8450((__int64 *)&v75, (__int64 **)&v78);
    sub_13BB090((__int64)&v75);
    v46 = (__int64)v75;
    v47 = (__m128i *)v75[3];
    if ( v75[2] - (_QWORD)v47 <= 1u )
    {
      v42 = "}\n";
      sub_16E7EE0(v75, "}\n", 2);
    }
    else
    {
      v47->m128i_i16[0] = 2685;
      *(_QWORD *)(v46 + 24) += 2LL;
    }
    v48 = v78;
    if ( (__m128i *)v78 != &v80 )
    {
      v42 = (const char *)(v80.m128i_i64[0] + 1);
      j_j___libc_free_0(v78, v80.m128i_i64[0] + 1);
    }
  }
  v49 = sub_16E8CB0(v48, v42, v47);
  v50 = *(_BYTE **)(v49 + 24);
  if ( *(_BYTE **)(v49 + 16) == v50 )
  {
    sub_16E7EE0(v49, "\n", 1);
  }
  else
  {
    *v50 = 10;
    ++*(_QWORD *)(v49 + 24);
  }
  if ( v69 != &v71 )
    j_j___libc_free_0(v69, v71.m128i_i64[0] + 1);
  if ( v66 != (_BYTE *)v68 )
    j_j___libc_free_0(v66, v68[0] + 1LL);
  sub_16E7C30(v81);
  if ( v57 != &v59 )
    j_j___libc_free_0(v57, v59.m128i_i64[0] + 1);
  return 0;
}
