// Function: sub_315CC60
// Address: 0x315cc60
//
void __fastcall sub_315CC60(__int64 *a1, unsigned __int64 a2)
{
  __int64 *v4; // r14
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // eax
  int v8; // esi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  size_t v12; // rax
  unsigned __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  size_t v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned int v25; // edx
  int v26; // eax
  unsigned __int64 v27; // r13
  __int64 v28; // rdi
  __m128i *v29; // rdx
  __m128i si128; // xmm0
  __int64 v31; // rdx
  __int64 v32; // rax
  __m128i v33; // xmm0
  __int64 v34; // rax
  _WORD *v35; // rdx
  __int64 v36; // r13
  char *v37; // rax
  __int64 v38; // rdx
  unsigned __int8 *v39; // rsi
  size_t v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdi
  _WORD *v44; // rdx
  unsigned __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rdx
  unsigned __int64 v48; // rax
  __int64 v49; // r14
  int v50; // r13d
  unsigned int v51; // r15d
  unsigned int v52; // r8d
  __int64 v53; // rdx
  __int64 v54; // rdi
  _BYTE *v55; // rax
  __int64 v56; // rdx
  char *v57; // rsi
  __int64 v58; // rdi
  _WORD *v59; // rdx
  int v60; // r15d
  unsigned int v61; // r8d
  __int64 v62; // rdi
  _BYTE *v63; // rax
  __int64 v64; // rdi
  _BYTE *v65; // rax
  __int64 v66; // rdi
  _BYTE *v67; // rax
  int v68; // eax
  int v69; // r8d
  unsigned __int8 *v70; // [rsp+20h] [rbp-B0h] BYREF
  size_t v71; // [rsp+28h] [rbp-A8h]
  _BYTE v72[16]; // [rsp+30h] [rbp-A0h] BYREF
  __m128i *v73; // [rsp+40h] [rbp-90h] BYREF
  size_t v74; // [rsp+48h] [rbp-88h]
  __m128i v75; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int8 *v76; // [rsp+60h] [rbp-70h] BYREF
  size_t v77; // [rsp+68h] [rbp-68h]
  _QWORD v78[12]; // [rsp+70h] [rbp-60h] BYREF

  v4 = *(__int64 **)a1[1];
  v71 = 0;
  v70 = v72;
  v72[0] = 0;
  if ( sub_3158140(*v4, a2) )
  {
    if ( 0x3FFFFFFFFFFFFFFFLL - v71 <= 0x1A )
      goto LABEL_119;
    sub_2241490((unsigned __int64 *)&v70, "style=filled,fillcolor=gray", 0x1Bu);
  }
  v5 = v4[1];
  if ( !v5 )
    goto LABEL_15;
  v6 = *(_QWORD *)(v5 + 8);
  v7 = *(_DWORD *)(v5 + 24);
  if ( !v7 )
    goto LABEL_15;
  v8 = v7 - 1;
  v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
  {
LABEL_5:
    if ( !*((_BYTE *)v10 + 8) )
      goto LABEL_15;
    v12 = v71;
    if ( v71 )
    {
      LOBYTE(v78[0]) = 44;
      v12 = 1;
    }
    v76 = (unsigned __int8 *)v78;
    v77 = v12;
    *((_BYTE *)v78 + v12) = 0;
    if ( 0x3FFFFFFFFFFFFFFFLL - v77 > 8 )
    {
      v13 = sub_2241490((unsigned __int64 *)&v76, "color=red", 9u);
      v73 = &v75;
      if ( (unsigned __int64 *)*v13 == v13 + 2 )
      {
        v75 = _mm_loadu_si128((const __m128i *)v13 + 1);
      }
      else
      {
        v73 = (__m128i *)*v13;
        v75.m128i_i64[0] = v13[2];
      }
      v74 = v13[1];
      *v13 = (unsigned __int64)(v13 + 2);
      v13[1] = 0;
      *((_BYTE *)v13 + 16) = 0;
      sub_2241490((unsigned __int64 *)&v70, v73->m128i_i8, v74);
      if ( v73 != &v75 )
        j_j___libc_free_0((unsigned __int64)v73);
      if ( v76 != (unsigned __int8 *)v78 )
        j_j___libc_free_0((unsigned __int64)v76);
      goto LABEL_15;
    }
LABEL_119:
    sub_4262D8((__int64)"basic_string::append");
  }
  v68 = 1;
  while ( v11 != -4096 )
  {
    v69 = v68 + 1;
    v9 = v8 & (v68 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      goto LABEL_5;
    v68 = v69;
  }
LABEL_15:
  v14 = *a1;
  v15 = *(_QWORD *)(*a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v15) <= 4 )
  {
    v14 = sub_CB6200(v14, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v15 = 1685016073;
    *(_BYTE *)(v15 + 4) = 101;
    *(_QWORD *)(v14 + 32) += 5LL;
  }
  v16 = sub_CB5A80(v14, a2);
  v17 = *(_QWORD **)(v16 + 32);
  if ( *(_QWORD *)(v16 + 24) - (_QWORD)v17 <= 7u )
  {
    sub_CB6200(v16, " [shape=", 8u);
  }
  else
  {
    *v17 = 0x3D65706168735B20LL;
    *(_QWORD *)(v16 + 32) += 8LL;
  }
  v18 = *a1;
  v19 = *(_QWORD *)(*a1 + 32);
  v20 = *(_QWORD *)(*a1 + 24) - v19;
  if ( *((_BYTE *)a1 + 16) )
  {
    if ( v20 <= 4 )
    {
      sub_CB6200(v18, (unsigned __int8 *)"none,", 5u);
    }
    else
    {
      *(_DWORD *)v19 = 1701736302;
      *(_BYTE *)(v19 + 4) = 44;
      *(_QWORD *)(v18 + 32) += 5LL;
    }
LABEL_22:
    v21 = v71;
    if ( !v71 )
      goto LABEL_23;
LABEL_68:
    v54 = sub_CB6200(*a1, v70, v21);
    v55 = *(_BYTE **)(v54 + 32);
    if ( *(_BYTE **)(v54 + 24) == v55 )
    {
      sub_CB6200(v54, (unsigned __int8 *)",", 1u);
    }
    else
    {
      *v55 = 44;
      ++*(_QWORD *)(v54 + 32);
    }
    goto LABEL_23;
  }
  if ( v20 <= 6 )
  {
    sub_CB6200(v18, (unsigned __int8 *)"record,", 7u);
    goto LABEL_22;
  }
  *(_DWORD *)v19 = 1868785010;
  *(_WORD *)(v19 + 4) = 25714;
  *(_BYTE *)(v19 + 6) = 44;
  v21 = v71;
  *(_QWORD *)(v18 + 32) += 7LL;
  if ( v21 )
    goto LABEL_68;
LABEL_23:
  v22 = *a1;
  v23 = *(_QWORD *)(*a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v23) <= 5 )
  {
    sub_CB6200(v22, "label=", 6u);
  }
  else
  {
    *(_DWORD *)v23 = 1700946284;
    *(_WORD *)(v23 + 4) = 15724;
    *(_QWORD *)(v22 + 32) += 6LL;
  }
  if ( *((_BYTE *)a1 + 16) )
  {
    v24 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v24 == a2 + 48 )
      goto LABEL_109;
    if ( !v24 )
      goto LABEL_88;
    if ( (unsigned int)*(unsigned __int8 *)(v24 - 24) - 30 > 0xA || (v25 = sub_B46E30(v24 - 24)) == 0 )
    {
LABEL_109:
      v27 = 1;
    }
    else
    {
      v26 = 0;
      do
      {
        if ( v25 == ++v26 )
        {
          v27 = v25;
          goto LABEL_34;
        }
      }
      while ( v26 != 64 );
      v27 = 65;
    }
LABEL_34:
    v28 = *a1;
    v29 = *(__m128i **)(*a1 + 32);
    if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v29 <= 0x30u )
    {
      v28 = sub_CB6200(v28, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"", 0x31u);
      v31 = *(_QWORD *)(v28 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v28 + 24) - v31) > 0x2E )
        goto LABEL_36;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB60);
      v29[3].m128i_i8[0] = 34;
      *v29 = si128;
      v29[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB70);
      v29[2] = _mm_load_si128((const __m128i *)&xmmword_3F8CB80);
      v31 = *(_QWORD *)(v28 + 32) + 49LL;
      v32 = *(_QWORD *)(v28 + 24);
      *(_QWORD *)(v28 + 32) = v31;
      if ( (unsigned __int64)(v32 - v31) > 0x2E )
      {
LABEL_36:
        v33 = _mm_load_si128((const __m128i *)&xmmword_3F8CB90);
        qmemcpy((void *)(v31 + 32), "text\" colspan=\"", 15);
        *(__m128i *)v31 = v33;
        *(__m128i *)(v31 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F8CBA0);
        *(_QWORD *)(v28 + 32) += 47LL;
        goto LABEL_37;
      }
    }
    v28 = sub_CB6200(v28, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"", 0x2Fu);
LABEL_37:
    v34 = sub_CB59D0(v28, v27);
    v35 = *(_WORD **)(v34 + 32);
    if ( *(_QWORD *)(v34 + 24) - (_QWORD)v35 <= 1u )
    {
      sub_CB6200(v34, "\">", 2u);
    }
    else
    {
      *v35 = 15906;
      *(_QWORD *)(v34 + 32) += 2LL;
    }
    goto LABEL_39;
  }
  v58 = *a1;
  v59 = *(_WORD **)(*a1 + 32);
  if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v59 <= 1u )
  {
    sub_CB6200(v58, (unsigned __int8 *)"\"{", 2u);
  }
  else
  {
    *v59 = 31522;
    *(_QWORD *)(v58 + 32) += 2LL;
  }
LABEL_39:
  v36 = *a1;
  if ( *((_BYTE *)a1 + 16) )
  {
    v37 = (char *)sub_BD5D20(a2);
    if ( v37 )
    {
      v76 = (unsigned __int8 *)v78;
      sub_3157F50((__int64 *)&v76, v37, (__int64)&v37[v38]);
      v39 = v76;
      v40 = v77;
    }
    else
    {
      LOBYTE(v78[0]) = 0;
      v40 = 0;
      v76 = (unsigned __int8 *)v78;
      v39 = (unsigned __int8 *)v78;
      v77 = 0;
    }
    v41 = sub_CB6200(v36, v39, v40);
    v42 = *(_QWORD *)(v41 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v41 + 24) - v42) <= 4 )
    {
      sub_CB6200(v41, "</td>", 5u);
    }
    else
    {
      *(_DWORD *)v42 = 1685335868;
      *(_BYTE *)(v42 + 4) = 62;
      *(_QWORD *)(v41 + 32) += 5LL;
    }
    if ( v76 != (unsigned __int8 *)v78 )
      j_j___libc_free_0((unsigned __int64)v76);
  }
  else
  {
    v57 = (char *)sub_BD5D20(a2);
    if ( v57 )
    {
      v73 = &v75;
      sub_3157F50((__int64 *)&v73, v57, (__int64)&v57[v56]);
    }
    else
    {
      v75.m128i_i8[0] = 0;
      v73 = &v75;
      v74 = 0;
    }
    sub_C67200((__int64 *)&v76, (__int64)&v73);
    sub_CB6200(v36, v76, v77);
    if ( v76 != (unsigned __int8 *)v78 )
      j_j___libc_free_0((unsigned __int64)v76);
    if ( v73 != &v75 )
      j_j___libc_free_0((unsigned __int64)v73);
  }
  v78[4] = &v73;
  v75.m128i_i8[0] = 0;
  v73 = &v75;
  v78[3] = 0x100000000LL;
  v74 = 0;
  v77 = 0;
  memset(v78, 0, 24);
  v76 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5980((__int64)&v76, 0, 0, 0);
  if ( (unsigned __int8)sub_3159520((__int64)a1, (__int64)&v76, a2) )
  {
    if ( *((_BYTE *)a1 + 16) )
      goto LABEL_48;
    v62 = *a1;
    v63 = *(_BYTE **)(*a1 + 32);
    if ( *(_BYTE **)(*a1 + 24) == v63 )
    {
      sub_CB6200(v62, (unsigned __int8 *)"|", 1u);
    }
    else
    {
      *v63 = 124;
      ++*(_QWORD *)(v62 + 32);
    }
    v64 = *a1;
    if ( *((_BYTE *)a1 + 16) )
    {
LABEL_48:
      sub_CB6200(*a1, (unsigned __int8 *)v73, v74);
    }
    else
    {
      v65 = *(_BYTE **)(v64 + 32);
      if ( *(_BYTE **)(v64 + 24) == v65 )
      {
        v64 = sub_CB6200(v64, (unsigned __int8 *)"{", 1u);
      }
      else
      {
        *v65 = 123;
        ++*(_QWORD *)(v64 + 32);
      }
      v66 = sub_CB6200(v64, (unsigned __int8 *)v73, v74);
      v67 = *(_BYTE **)(v66 + 32);
      if ( *(_BYTE **)(v66 + 24) == v67 )
      {
        sub_CB6200(v66, (unsigned __int8 *)"}", 1u);
      }
      else
      {
        *v67 = 125;
        ++*(_QWORD *)(v66 + 32);
      }
    }
  }
  v43 = *a1;
  v44 = *(_WORD **)(*a1 + 32);
  v45 = *(_QWORD *)(*a1 + 24) - (_QWORD)v44;
  if ( *((_BYTE *)a1 + 16) )
  {
    if ( v45 <= 0xD )
    {
      sub_CB6200(v43, "</tr></table>>", 0xEu);
    }
    else
    {
      qmemcpy(v44, "</tr></table>>", 14);
      *(_QWORD *)(v43 + 32) += 14LL;
    }
  }
  else if ( v45 <= 1 )
  {
    sub_CB6200(v43, (unsigned __int8 *)"}\"", 2u);
  }
  else
  {
    *v44 = 8829;
    *(_QWORD *)(v43 + 32) += 2LL;
  }
  v46 = *a1;
  v47 = *(_QWORD *)(*a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v47) <= 2 )
  {
    sub_CB6200(v46, (unsigned __int8 *)"];\n", 3u);
  }
  else
  {
    *(_BYTE *)(v47 + 2) = 10;
    *(_WORD *)v47 = 15197;
    *(_QWORD *)(v46 + 32) += 3LL;
  }
  v48 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v48 == a2 + 48 )
    goto LABEL_61;
  if ( !v48 )
LABEL_88:
    BUG();
  v49 = v48 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v48 - 24) - 30 <= 0xA )
  {
    v50 = sub_B46E30(v49);
    if ( v50 )
    {
      v51 = 0;
      while ( 1 )
      {
        v52 = v51;
        v53 = v51++;
        sub_315C7D0(a1, a2, v53, v49, v52);
        if ( v51 == v50 )
          break;
        if ( v51 == 64 )
        {
          v60 = 64;
          do
          {
            v61 = v60++;
            sub_315C7D0(a1, a2, 64, v49, v61);
          }
          while ( v50 != v60 );
          break;
        }
      }
    }
  }
LABEL_61:
  v76 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5840((__int64)&v76);
  if ( v73 != &v75 )
    j_j___libc_free_0((unsigned __int64)v73);
  if ( v70 != v72 )
    j_j___libc_free_0((unsigned __int64)v70);
}
