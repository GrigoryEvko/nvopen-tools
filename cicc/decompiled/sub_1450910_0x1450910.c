// Function: sub_1450910
// Address: 0x1450910
//
__int64 __fastcall sub_1450910(_QWORD *a1, __int64 a2)
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
  _BYTE *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rcx
  __m128i *v36; // rax
  __m128i *v37; // rsi
  __m128i *v38; // rdx
  __m128i *v39; // rax
  __int64 *v40; // rsi
  __int64 v41; // rdi
  _BYTE *v42; // rdi
  __m128i *v43; // rdx
  __int64 v44; // rdi
  _BYTE *v45; // rax
  __int64 v47; // rax
  __m128i si128; // xmm0
  __int64 v49; // [rsp+28h] [rbp-198h] BYREF
  unsigned int v50; // [rsp+30h] [rbp-190h] BYREF
  __int64 v51; // [rsp+38h] [rbp-188h]
  __m128i *v52; // [rsp+40h] [rbp-180h]
  __int64 v53; // [rsp+48h] [rbp-178h]
  __m128i v54; // [rsp+50h] [rbp-170h] BYREF
  _QWORD *v55; // [rsp+60h] [rbp-160h] BYREF
  __int64 v56; // [rsp+68h] [rbp-158h]
  _QWORD v57[2]; // [rsp+70h] [rbp-150h] BYREF
  _QWORD *v58; // [rsp+80h] [rbp-140h] BYREF
  __int64 v59; // [rsp+88h] [rbp-138h]
  _QWORD v60[2]; // [rsp+90h] [rbp-130h] BYREF
  _BYTE *v61[2]; // [rsp+A0h] [rbp-120h] BYREF
  _QWORD v62[2]; // [rsp+B0h] [rbp-110h] BYREF
  __m128i *v63; // [rsp+C0h] [rbp-100h] BYREF
  __int64 *v64; // [rsp+C8h] [rbp-F8h]
  __m128i v65; // [rsp+D0h] [rbp-F0h] BYREF
  _QWORD *v66; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v67; // [rsp+E8h] [rbp-D8h]
  _QWORD v68[2]; // [rsp+F0h] [rbp-D0h] BYREF
  _QWORD *v69; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v70; // [rsp+108h] [rbp-B8h]
  _QWORD v71[2]; // [rsp+110h] [rbp-B0h] BYREF
  __m128i *v72; // [rsp+120h] [rbp-A0h] BYREF
  __int64 v73; // [rsp+128h] [rbp-98h]
  __m128i v74; // [rsp+130h] [rbp-90h] BYREF
  _BYTE v75[128]; // [rsp+140h] [rbp-80h] BYREF

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_77:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9A04C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_77;
  }
  v49 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9A04C)
      + 160;
  v6 = (_BYTE *)sub_1649960(a2);
  if ( v6 )
  {
    v58 = v60;
    sub_144C6E0((__int64 *)&v58, v6, (__int64)&v6[v7]);
  }
  else
  {
    LOBYTE(v60[0]) = 0;
    v58 = v60;
    v59 = 0;
  }
  v8 = (_BYTE *)a1[20];
  v9 = a1[21];
  v55 = v57;
  sub_144C790((__int64 *)&v55, v8, (__int64)&v8[v9]);
  if ( v56 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_76:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v55, ".", 1, v10);
  v11 = 15;
  v12 = 15;
  if ( v55 != v57 )
    v12 = v57[0];
  v13 = v56 + v59;
  if ( v56 + v59 > v12 )
  {
    if ( v58 != v60 )
      v11 = v60[0];
    if ( v13 <= v11 )
    {
      v14 = (__m128i *)sub_2241130(&v58, 0, 0, v55, v56);
      v63 = &v65;
      v15 = (__m128i *)v14->m128i_i64[0];
      v16 = v14 + 1;
      if ( (__m128i *)v14->m128i_i64[0] != &v14[1] )
        goto LABEL_15;
LABEL_74:
      v65 = _mm_loadu_si128(v14 + 1);
      goto LABEL_16;
    }
  }
  v14 = (__m128i *)sub_2241490(&v55, v58, v59, v13);
  v63 = &v65;
  v15 = (__m128i *)v14->m128i_i64[0];
  v16 = v14 + 1;
  if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
    goto LABEL_74;
LABEL_15:
  v63 = v15;
  v65.m128i_i64[0] = v14[1].m128i_i64[0];
LABEL_16:
  v17 = v14->m128i_i64[1];
  v64 = (__int64 *)v17;
  v14->m128i_i64[0] = (__int64)v16;
  v14->m128i_i64[1] = 0;
  v14[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)v64) <= 3 )
    goto LABEL_76;
  v18 = ".dot";
  v19 = sub_2241490(&v63, ".dot", 4, v17);
  v21 = v19 + 16;
  v52 = &v54;
  if ( *(_QWORD *)v19 == v19 + 16 )
  {
    v54 = _mm_loadu_si128((const __m128i *)(v19 + 16));
  }
  else
  {
    v52 = *(__m128i **)v19;
    v54.m128i_i64[0] = *(_QWORD *)(v19 + 16);
  }
  v22 = *(_QWORD *)(v19 + 8);
  *(_BYTE *)(v19 + 16) = 0;
  v53 = v22;
  *(_QWORD *)v19 = v21;
  *(_QWORD *)(v19 + 8) = 0;
  if ( v63 != &v65 )
  {
    v18 = (char *)(v65.m128i_i64[0] + 1);
    j_j___libc_free_0(v63, v65.m128i_i64[0] + 1);
  }
  if ( v55 != v57 )
  {
    v18 = (char *)(v57[0] + 1LL);
    j_j___libc_free_0(v55, v57[0] + 1LL);
  }
  v23 = (__int64)v58;
  if ( v58 != v60 )
  {
    v18 = (char *)(v60[0] + 1LL);
    j_j___libc_free_0(v58, v60[0] + 1LL);
  }
  v50 = 0;
  v51 = sub_2241E40(v23, v18, v21, v22, v20);
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
  v28 = sub_16E7EE0(v27, v52->m128i_i8, v53);
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
  sub_16E8AF0(v75, v52, v53, &v50, 1);
  v61[0] = v62;
  sub_144C6E0((__int64 *)v61, "Region Graph", (__int64)"");
  v30 = (_BYTE *)sub_1649960(a2);
  if ( v30 )
  {
    v69 = v71;
    sub_144C6E0((__int64 *)&v69, v30, (__int64)&v30[v31]);
  }
  else
  {
    LOBYTE(v71[0]) = 0;
    v69 = v71;
    v70 = 0;
  }
  v66 = v68;
  sub_144C790((__int64 *)&v66, v61[0], (__int64)&v61[0][(unsigned __int64)v61[1]]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v67) <= 5 )
    goto LABEL_76;
  sub_2241490(&v66, " for '", 6, v32);
  v33 = 15;
  v34 = 15;
  if ( v66 != v68 )
    v34 = v68[0];
  v35 = v67 + v70;
  if ( v67 + v70 <= v34 )
    goto LABEL_38;
  if ( v69 != v71 )
    v33 = v71[0];
  if ( v35 <= v33 )
  {
    v36 = (__m128i *)sub_2241130(&v69, 0, 0, v66, v67);
    v72 = &v74;
    v37 = (__m128i *)v36->m128i_i64[0];
    v38 = v36 + 1;
    if ( (__m128i *)v36->m128i_i64[0] != &v36[1] )
      goto LABEL_39;
  }
  else
  {
LABEL_38:
    v36 = (__m128i *)sub_2241490(&v66, v69, v70, v35);
    v72 = &v74;
    v37 = (__m128i *)v36->m128i_i64[0];
    v38 = v36 + 1;
    if ( (__m128i *)v36->m128i_i64[0] != &v36[1] )
    {
LABEL_39:
      v72 = v37;
      v74.m128i_i64[0] = v36[1].m128i_i64[0];
      goto LABEL_40;
    }
  }
  v74 = _mm_loadu_si128(v36 + 1);
LABEL_40:
  v73 = v36->m128i_i64[1];
  v36->m128i_i64[0] = (__int64)v38;
  v36->m128i_i64[1] = 0;
  v36[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v73) <= 9 )
    goto LABEL_76;
  v39 = (__m128i *)sub_2241490(&v72, "' function", 10, &v74);
  v63 = &v65;
  if ( (__m128i *)v39->m128i_i64[0] == &v39[1] )
  {
    v65 = _mm_loadu_si128(v39 + 1);
  }
  else
  {
    v63 = (__m128i *)v39->m128i_i64[0];
    v65.m128i_i64[0] = v39[1].m128i_i64[0];
  }
  v40 = (__int64 *)v39->m128i_i64[1];
  v64 = v40;
  v39->m128i_i64[0] = (__int64)v39[1].m128i_i64;
  v39->m128i_i64[1] = 0;
  v39[1].m128i_i8[0] = 0;
  if ( v72 != &v74 )
  {
    v40 = (__int64 *)(v74.m128i_i64[0] + 1);
    j_j___libc_free_0(v72, v74.m128i_i64[0] + 1);
  }
  if ( v66 != v68 )
  {
    v40 = (__int64 *)(v68[0] + 1LL);
    j_j___libc_free_0(v66, v68[0] + 1LL);
  }
  v41 = (__int64)v69;
  if ( v69 != v71 )
  {
    v40 = (__int64 *)(v71[0] + 1LL);
    j_j___libc_free_0(v69, v71[0] + 1LL);
  }
  if ( v50 )
  {
    v47 = sub_16E8CB0(v41, v40, v50);
    v43 = *(__m128i **)(v47 + 24);
    v42 = (_BYTE *)v47;
    if ( *(_QWORD *)(v47 + 16) - (_QWORD)v43 <= 0x20u )
    {
      v40 = (__int64 *)"  error opening file for writing!";
      sub_16E7EE0(v47, "  error opening file for writing!", 33);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
      v43[2].m128i_i8[0] = 33;
      *v43 = si128;
      v43[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
      *(_QWORD *)(v47 + 24) += 33LL;
    }
  }
  else
  {
    v42 = v75;
    v74.m128i_i16[0] = 260;
    v40 = &v49;
    v72 = (__m128i *)&v63;
    sub_14507E0((__int64)v75, &v49, 0, (__int64)&v72);
  }
  v44 = sub_16E8CB0(v42, v40, v43);
  v45 = *(_BYTE **)(v44 + 24);
  if ( *(_BYTE **)(v44 + 16) == v45 )
  {
    sub_16E7EE0(v44, "\n", 1);
  }
  else
  {
    *v45 = 10;
    ++*(_QWORD *)(v44 + 24);
  }
  if ( v63 != &v65 )
    j_j___libc_free_0(v63, v65.m128i_i64[0] + 1);
  if ( (_QWORD *)v61[0] != v62 )
    j_j___libc_free_0(v61[0], v62[0] + 1LL);
  sub_16E7C30(v75);
  if ( v52 != &v54 )
    j_j___libc_free_0(v52, v54.m128i_i64[0] + 1);
  return 0;
}
