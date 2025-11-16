// Function: sub_13BD710
// Address: 0x13bd710
//
__int64 __fastcall sub_13BD710(_QWORD *a1, __int64 a2)
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
  __int64 v30; // rdx
  _BYTE *v31; // rsi
  __int64 v32; // rcx
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdi
  char *v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rdx
  __m128i *v38; // rcx
  __int64 v39; // rcx
  const char *v40; // rsi
  __m128i *v41; // rax
  __m128i *v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // rdi
  __m128i *v45; // rdx
  _QWORD *m128i_i64; // rdi
  __int64 v47; // rdi
  _BYTE *v48; // rax
  __int64 v50; // rax
  __m128i si128; // xmm0
  __int64 v52; // [rsp+38h] [rbp-198h] BYREF
  int v53; // [rsp+40h] [rbp-190h] BYREF
  __int64 v54; // [rsp+48h] [rbp-188h]
  __m128i *v55; // [rsp+50h] [rbp-180h]
  __int64 v56; // [rsp+58h] [rbp-178h]
  __m128i v57; // [rsp+60h] [rbp-170h] BYREF
  _QWORD *v58; // [rsp+70h] [rbp-160h] BYREF
  __int64 v59; // [rsp+78h] [rbp-158h]
  _QWORD v60[2]; // [rsp+80h] [rbp-150h] BYREF
  _QWORD *v61; // [rsp+90h] [rbp-140h] BYREF
  __int64 v62; // [rsp+98h] [rbp-138h]
  _QWORD v63[2]; // [rsp+A0h] [rbp-130h] BYREF
  _BYTE *v64; // [rsp+B0h] [rbp-120h]
  __int64 v65; // [rsp+B8h] [rbp-118h]
  _QWORD v66[2]; // [rsp+C0h] [rbp-110h] BYREF
  __m128i *v67; // [rsp+D0h] [rbp-100h] BYREF
  __int64 v68; // [rsp+D8h] [rbp-F8h]
  __m128i v69; // [rsp+E0h] [rbp-F0h] BYREF
  __m128i **v70; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v71; // [rsp+F8h] [rbp-D8h]
  _QWORD v72[2]; // [rsp+100h] [rbp-D0h] BYREF
  _QWORD *v73; // [rsp+110h] [rbp-C0h] BYREF
  __int64 *v74; // [rsp+118h] [rbp-B8h]
  _QWORD v75[2]; // [rsp+120h] [rbp-B0h] BYREF
  __m128i *v76; // [rsp+130h] [rbp-A0h] BYREF
  __int64 v77; // [rsp+138h] [rbp-98h]
  __m128i v78; // [rsp+140h] [rbp-90h] BYREF
  _BYTE v79[128]; // [rsp+150h] [rbp-80h] BYREF

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_81:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9E06C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_81;
  }
  v52 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9E06C)
      + 160;
  v6 = (_BYTE *)sub_1649960(a2);
  if ( v6 )
  {
    v61 = v63;
    sub_13B5840((__int64 *)&v61, v6, (__int64)&v6[v7]);
  }
  else
  {
    LOBYTE(v63[0]) = 0;
    v61 = v63;
    v62 = 0;
  }
  v8 = (_BYTE *)a1[20];
  v9 = a1[21];
  v58 = v60;
  sub_13B5790((__int64 *)&v58, v8, (__int64)&v8[v9]);
  if ( v59 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_80:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v58, ".", 1, v10);
  v11 = 15;
  v12 = 15;
  if ( v58 != v60 )
    v12 = v60[0];
  v13 = v59 + v62;
  if ( v59 + v62 > v12 )
  {
    if ( v61 != v63 )
      v11 = v63[0];
    if ( v13 <= v11 )
    {
      v14 = (__m128i *)sub_2241130(&v61, 0, 0, v58, v59);
      v67 = &v69;
      v15 = (__m128i *)v14->m128i_i64[0];
      v16 = v14 + 1;
      if ( (__m128i *)v14->m128i_i64[0] != &v14[1] )
        goto LABEL_15;
LABEL_77:
      v69 = _mm_loadu_si128(v14 + 1);
      goto LABEL_16;
    }
  }
  v14 = (__m128i *)sub_2241490(&v58, v61, v62, v13);
  v67 = &v69;
  v15 = (__m128i *)v14->m128i_i64[0];
  v16 = v14 + 1;
  if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
    goto LABEL_77;
LABEL_15:
  v67 = v15;
  v69.m128i_i64[0] = v14[1].m128i_i64[0];
LABEL_16:
  v17 = v14->m128i_i64[1];
  v68 = v17;
  v14->m128i_i64[0] = (__int64)v16;
  v14->m128i_i64[1] = 0;
  v14[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v68) <= 3 )
    goto LABEL_80;
  v18 = ".dot";
  v19 = sub_2241490(&v67, ".dot", 4, v17);
  v21 = v19 + 16;
  v55 = &v57;
  if ( *(_QWORD *)v19 == v19 + 16 )
  {
    v57 = _mm_loadu_si128((const __m128i *)(v19 + 16));
  }
  else
  {
    v55 = *(__m128i **)v19;
    v57.m128i_i64[0] = *(_QWORD *)(v19 + 16);
  }
  v22 = *(_QWORD *)(v19 + 8);
  *(_BYTE *)(v19 + 16) = 0;
  v56 = v22;
  *(_QWORD *)v19 = v21;
  *(_QWORD *)(v19 + 8) = 0;
  if ( v67 != &v69 )
  {
    v18 = (char *)(v69.m128i_i64[0] + 1);
    j_j___libc_free_0(v67, v69.m128i_i64[0] + 1);
  }
  if ( v58 != v60 )
  {
    v18 = (char *)(v60[0] + 1LL);
    j_j___libc_free_0(v58, v60[0] + 1LL);
  }
  v23 = (__int64)v61;
  if ( v61 != v63 )
  {
    v18 = (char *)(v63[0] + 1LL);
    j_j___libc_free_0(v61, v63[0] + 1LL);
  }
  v53 = 0;
  v54 = sub_2241E40(v23, v18, v21, v22, v20);
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
  v28 = sub_16E7EE0(v27, v55->m128i_i8, v56);
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
  sub_16E8AF0(v79, v55, v56, &v53, 1);
  v64 = v66;
  strcpy((char *)v66, "Dominator tree");
  v65 = 14;
  v31 = (_BYTE *)sub_1649960(a2);
  if ( v31 )
  {
    v73 = v75;
    sub_13B5840((__int64 *)&v73, v31, (__int64)&v31[v30]);
  }
  else
  {
    LOBYTE(v75[0]) = 0;
    v73 = v75;
    v74 = 0;
  }
  v70 = (__m128i **)v72;
  sub_13B5790((__int64 *)&v70, v64, (__int64)&v64[v65]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v71) <= 5 )
    goto LABEL_80;
  sub_2241490(&v70, " for '", 6, v32);
  v33 = 15;
  v34 = 15;
  if ( v70 != v72 )
    v34 = v72[0];
  v35 = (char *)v74 + v71;
  if ( (unsigned __int64)v74 + v71 <= v34 )
    goto LABEL_38;
  if ( v73 != v75 )
    v33 = v75[0];
  if ( (unsigned __int64)v35 <= v33 )
  {
    v36 = sub_2241130(&v73, 0, 0, v70, v71);
    v37 = v36 + 16;
    v76 = &v78;
    v38 = *(__m128i **)v36;
    if ( *(_QWORD *)v36 != v36 + 16 )
      goto LABEL_39;
  }
  else
  {
LABEL_38:
    v36 = sub_2241490(&v70, v73, v74, v35);
    v37 = v36 + 16;
    v76 = &v78;
    v38 = *(__m128i **)v36;
    if ( *(_QWORD *)v36 != v36 + 16 )
    {
LABEL_39:
      v76 = v38;
      v78.m128i_i64[0] = *(_QWORD *)(v36 + 16);
      goto LABEL_40;
    }
  }
  v78 = _mm_loadu_si128((const __m128i *)(v36 + 16));
LABEL_40:
  v39 = *(_QWORD *)(v36 + 8);
  v77 = v39;
  *(_QWORD *)v36 = v37;
  *(_QWORD *)(v36 + 8) = 0;
  *(_BYTE *)(v36 + 16) = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v77) <= 9 )
    goto LABEL_80;
  v40 = "' function";
  v41 = (__m128i *)sub_2241490(&v76, "' function", 10, v39);
  v67 = &v69;
  v42 = v41 + 1;
  if ( (__m128i *)v41->m128i_i64[0] == &v41[1] )
  {
    v69 = _mm_loadu_si128(v41 + 1);
  }
  else
  {
    v67 = (__m128i *)v41->m128i_i64[0];
    v69.m128i_i64[0] = v41[1].m128i_i64[0];
  }
  v68 = v41->m128i_i64[1];
  v41->m128i_i64[0] = (__int64)v42;
  v41->m128i_i64[1] = 0;
  v41[1].m128i_i8[0] = 0;
  if ( v76 != &v78 )
  {
    v40 = (const char *)(v78.m128i_i64[0] + 1);
    j_j___libc_free_0(v76, v78.m128i_i64[0] + 1);
  }
  if ( v70 != v72 )
  {
    v40 = (const char *)(v72[0] + 1LL);
    j_j___libc_free_0(v70, v72[0] + 1LL);
  }
  v43 = (__int64)v73;
  if ( v73 != v75 )
  {
    v40 = (const char *)(v75[0] + 1LL);
    j_j___libc_free_0(v73, v75[0] + 1LL);
  }
  if ( v53 )
  {
    v50 = sub_16E8CB0(v43, v40, v42);
    v45 = *(__m128i **)(v50 + 24);
    m128i_i64 = (_QWORD *)v50;
    if ( *(_QWORD *)(v50 + 16) - (_QWORD)v45 <= 0x20u )
    {
      v40 = "  error opening file for writing!";
      sub_16E7EE0(v50, "  error opening file for writing!", 33);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
      v45[2].m128i_i8[0] = 33;
      *v45 = si128;
      v45[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
      *(_QWORD *)(v50 + 24) += 33LL;
    }
  }
  else
  {
    LOWORD(v72[0]) = 260;
    v70 = &v67;
    v74 = &v52;
    v73 = v79;
    LOBYTE(v75[0]) = 1;
    sub_16E2FC0(&v76, &v70);
    v40 = (const char *)&v76;
    sub_13B8800((__int64 *)&v73, (_QWORD **)&v76);
    sub_13BA9D0((__int64)&v73);
    v44 = (__int64)v73;
    v45 = (__m128i *)v73[3];
    if ( v73[2] - (_QWORD)v45 <= 1u )
    {
      v40 = "}\n";
      sub_16E7EE0(v73, "}\n", 2);
    }
    else
    {
      v45->m128i_i16[0] = 2685;
      *(_QWORD *)(v44 + 24) += 2LL;
    }
    m128i_i64 = v76->m128i_i64;
    if ( v76 != &v78 )
    {
      v40 = (const char *)(v78.m128i_i64[0] + 1);
      j_j___libc_free_0(v76, v78.m128i_i64[0] + 1);
    }
  }
  v47 = sub_16E8CB0(m128i_i64, v40, v45);
  v48 = *(_BYTE **)(v47 + 24);
  if ( *(_BYTE **)(v47 + 16) == v48 )
  {
    sub_16E7EE0(v47, "\n", 1);
  }
  else
  {
    *v48 = 10;
    ++*(_QWORD *)(v47 + 24);
  }
  if ( v67 != &v69 )
    j_j___libc_free_0(v67, v69.m128i_i64[0] + 1);
  if ( v64 != (_BYTE *)v66 )
    j_j___libc_free_0(v64, v66[0] + 1LL);
  sub_16E7C30(v79);
  if ( v55 != &v57 )
    j_j___libc_free_0(v55, v57.m128i_i64[0] + 1);
  return 0;
}
