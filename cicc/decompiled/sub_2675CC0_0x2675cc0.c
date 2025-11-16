// Function: sub_2675CC0
// Address: 0x2675cc0
//
__m128i *__fastcall sub_2675CC0(__m128i *a1, unsigned __int64 a2)
{
  char *v2; // r13
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  int v8; // eax
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rcx
  int v13; // eax
  char v14; // al
  unsigned __int64 v15; // rax
  const char *v16; // rsi
  __m128i *v17; // rax
  unsigned __int64 v18; // rcx
  __m128i *v19; // rax
  unsigned __int64 v20; // rcx
  __m128i *v21; // rax
  unsigned __int64 v22; // rcx
  __m128i *v23; // rax
  __int64 v24; // rcx
  unsigned __int64 v25; // rdx
  __m128i *v26; // rax
  __int64 v27; // rcx
  _OWORD *v28; // rdi
  unsigned __int64 v30; // r14
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rsi
  unsigned __int64 v33; // rcx
  int v34; // eax
  unsigned __int64 v35; // r9
  unsigned __int64 v36; // rdx
  unsigned __int64 v37; // rsi
  unsigned __int64 v38; // rcx
  int v39; // eax
  unsigned __int64 v40; // [rsp+20h] [rbp-260h]
  unsigned __int64 v41; // [rsp+28h] [rbp-258h]
  __int64 v42[2]; // [rsp+30h] [rbp-250h] BYREF
  __int64 v43; // [rsp+40h] [rbp-240h] BYREF
  unsigned __int64 v44[2]; // [rsp+50h] [rbp-230h] BYREF
  _BYTE v45[16]; // [rsp+60h] [rbp-220h] BYREF
  __m128i v46; // [rsp+70h] [rbp-210h] BYREF
  __int64 v47; // [rsp+80h] [rbp-200h] BYREF
  unsigned __int64 v48[2]; // [rsp+90h] [rbp-1F0h] BYREF
  _QWORD v49[2]; // [rsp+A0h] [rbp-1E0h] BYREF
  __m128i v50; // [rsp+B0h] [rbp-1D0h] BYREF
  __int64 v51; // [rsp+C0h] [rbp-1C0h] BYREF
  _BYTE *v52; // [rsp+D0h] [rbp-1B0h] BYREF
  int v53; // [rsp+D8h] [rbp-1A8h]
  _QWORD v54[2]; // [rsp+E0h] [rbp-1A0h] BYREF
  __m128i v55; // [rsp+F0h] [rbp-190h] BYREF
  __int64 v56; // [rsp+100h] [rbp-180h] BYREF
  unsigned __int64 v57[2]; // [rsp+110h] [rbp-170h] BYREF
  __m128i v58; // [rsp+120h] [rbp-160h] BYREF
  _BYTE *v59; // [rsp+130h] [rbp-150h] BYREF
  int v60; // [rsp+138h] [rbp-148h]
  _QWORD v61[2]; // [rsp+140h] [rbp-140h] BYREF
  __m128i v62; // [rsp+150h] [rbp-130h] BYREF
  __int64 v63; // [rsp+160h] [rbp-120h] BYREF
  unsigned __int64 v64[2]; // [rsp+170h] [rbp-110h] BYREF
  __m128i v65; // [rsp+180h] [rbp-100h] BYREF
  _BYTE *v66; // [rsp+190h] [rbp-F0h] BYREF
  int v67; // [rsp+198h] [rbp-E8h]
  _QWORD v68[2]; // [rsp+1A0h] [rbp-E0h] BYREF
  __m128i v69; // [rsp+1B0h] [rbp-D0h] BYREF
  __int64 v70; // [rsp+1C0h] [rbp-C0h] BYREF
  unsigned __int64 v71[2]; // [rsp+1D0h] [rbp-B0h] BYREF
  __m128i v72; // [rsp+1E0h] [rbp-A0h] BYREF
  _BYTE *v73; // [rsp+1F0h] [rbp-90h] BYREF
  int v74; // [rsp+1F8h] [rbp-88h]
  _QWORD v75[2]; // [rsp+200h] [rbp-80h] BYREF
  __m128i v76; // [rsp+210h] [rbp-70h] BYREF
  __int64 v77; // [rsp+220h] [rbp-60h] BYREF
  _OWORD *v78; // [rsp+230h] [rbp-50h] BYREF
  __int64 v79; // [rsp+238h] [rbp-48h]
  _OWORD v80[4]; // [rsp+240h] [rbp-40h] BYREF

  v2 = "yes";
  v4 = a2;
  if ( !*(_BYTE *)(a2 + 464) )
    v2 = "no";
  if ( *(_BYTE *)(a2 + 401) )
  {
    v5 = *(_QWORD *)(a2 + 448);
    if ( v5 <= 9 )
    {
      a2 = 1;
    }
    else if ( v5 <= 0x63 )
    {
      a2 = 2;
    }
    else if ( v5 <= 0x3E7 )
    {
      a2 = 3;
    }
    else if ( v5 <= 0x270F )
    {
      a2 = 4;
    }
    else
    {
      v6 = *(_QWORD *)(a2 + 448);
      LODWORD(a2) = 1;
      while ( 1 )
      {
        v7 = v6;
        v8 = a2;
        a2 = (unsigned int)(a2 + 4);
        v6 /= 0x2710u;
        if ( v7 <= 0x1869F )
          break;
        if ( v7 <= 0xF423F )
        {
          a2 = (unsigned int)(v8 + 5);
          break;
        }
        if ( v7 <= (unsigned __int64)&loc_98967F )
        {
          a2 = (unsigned int)(v8 + 6);
          break;
        }
        if ( v7 <= 0x5F5E0FF )
        {
          a2 = (unsigned int)(v8 + 7);
          break;
        }
      }
    }
    v73 = v75;
    sub_2240A50((__int64 *)&v73, a2, 0);
    sub_1249540(v73, v74, v5);
    if ( !*(_BYTE *)(v4 + 337) )
    {
LABEL_14:
      sub_26712D0((__int64 *)&v66, "<invalid>");
      if ( !*(_BYTE *)(v4 + 177) )
        goto LABEL_15;
LABEL_91:
      v35 = *(unsigned int *)(v4 + 224);
      if ( v35 <= 9 )
      {
        v37 = 1;
      }
      else if ( v35 <= 0x63 )
      {
        v37 = 2;
      }
      else if ( v35 <= 0x3E7 )
      {
        v37 = 3;
      }
      else if ( v35 <= 0x270F )
      {
        v37 = 4;
      }
      else
      {
        v36 = *(unsigned int *)(v4 + 224);
        LODWORD(v37) = 1;
        while ( 1 )
        {
          v38 = v36;
          v39 = v37;
          v37 = (unsigned int)(v37 + 4);
          v36 /= 0x2710u;
          if ( v38 <= 0x1869F )
            break;
          if ( v38 <= 0xF423F )
          {
            v37 = (unsigned int)(v39 + 5);
            break;
          }
          if ( v38 <= (unsigned __int64)&loc_98967F )
          {
            v37 = (unsigned int)(v39 + 6);
            break;
          }
          if ( v38 <= 0x5F5E0FF )
          {
            v37 = (unsigned int)(v39 + 7);
            break;
          }
        }
      }
      v41 = *(unsigned int *)(v4 + 224);
      v59 = v61;
      sub_2240A50((__int64 *)&v59, v37, 0);
      sub_1249540(v59, v60, v41);
      if ( *(_BYTE *)(v4 + 113) )
        goto LABEL_16;
LABEL_101:
      sub_26712D0((__int64 *)&v52, "<invalid>");
      goto LABEL_26;
    }
  }
  else
  {
    sub_26712D0((__int64 *)&v73, "<invalid>");
    if ( !*(_BYTE *)(a2 + 337) )
      goto LABEL_14;
  }
  v30 = *(unsigned int *)(v4 + 384);
  if ( v30 <= 9 )
  {
    v32 = 1;
  }
  else if ( v30 <= 0x63 )
  {
    v32 = 2;
  }
  else if ( v30 <= 0x3E7 )
  {
    v32 = 3;
  }
  else if ( v30 <= 0x270F )
  {
    v32 = 4;
  }
  else
  {
    v31 = *(unsigned int *)(v4 + 384);
    LODWORD(v32) = 1;
    while ( 1 )
    {
      v33 = v31;
      v34 = v32;
      v32 = (unsigned int)(v32 + 4);
      v31 /= 0x2710u;
      if ( v33 <= 0x1869F )
        break;
      if ( v33 <= 0xF423F )
      {
        v32 = (unsigned int)(v34 + 5);
        break;
      }
      if ( v33 <= (unsigned __int64)&loc_98967F )
      {
        v32 = (unsigned int)(v34 + 6);
        break;
      }
      if ( v33 <= 0x5F5E0FF )
      {
        v32 = (unsigned int)(v34 + 7);
        break;
      }
    }
  }
  v66 = v68;
  sub_2240A50((__int64 *)&v66, v32, 0);
  sub_1249540(v66, v67, v30);
  if ( *(_BYTE *)(v4 + 177) )
    goto LABEL_91;
LABEL_15:
  sub_26712D0((__int64 *)&v59, "<invalid>");
  if ( !*(_BYTE *)(v4 + 113) )
    goto LABEL_101;
LABEL_16:
  v9 = *(unsigned int *)(v4 + 160);
  if ( v9 <= 9 )
  {
    v11 = 1;
  }
  else if ( v9 <= 0x63 )
  {
    v11 = 2;
  }
  else if ( v9 <= 0x3E7 )
  {
    v11 = 3;
  }
  else if ( v9 <= 0x270F )
  {
    v11 = 4;
  }
  else
  {
    v10 = *(unsigned int *)(v4 + 160);
    LODWORD(v11) = 1;
    while ( 1 )
    {
      v12 = v10;
      v13 = v11;
      v11 = (unsigned int)(v11 + 4);
      v10 /= 0x2710u;
      if ( v12 <= 0x1869F )
        break;
      if ( v12 <= 0xF423F )
      {
        v11 = (unsigned int)(v13 + 5);
        break;
      }
      if ( v12 <= (unsigned __int64)&loc_98967F )
      {
        v11 = (unsigned int)(v13 + 6);
        break;
      }
      if ( v12 <= 0x5F5E0FF )
      {
        v11 = (unsigned int)(v13 + 7);
        break;
      }
    }
  }
  v40 = *(unsigned int *)(v4 + 160);
  v52 = v54;
  sub_2240A50((__int64 *)&v52, v11, 0);
  sub_1249540(v52, v53, v40);
LABEL_26:
  v49[0] = 0x203A7352502320LL;
  v48[0] = (unsigned __int64)v49;
  v14 = *(_BYTE *)(v4 + 241);
  v48[1] = 7;
  if ( *(_BYTE *)(v4 + 240) == v14 )
  {
    qmemcpy(v45, " [FIX]", 6);
    v44[0] = (unsigned __int64)v45;
    v15 = 6;
  }
  else
  {
    v44[0] = (unsigned __int64)v45;
    v15 = 0;
  }
  v44[1] = v15;
  v16 = "SPMD";
  v45[v15] = 0;
  if ( !*(_BYTE *)(v4 + 241) )
    v16 = "generic";
  sub_26712D0(v42, v16);
  sub_8FD5D0(&v46, (__int64)v42, v44);
  sub_8FD5D0(&v50, (__int64)&v46, v48);
  sub_8FD5D0(&v55, (__int64)&v50, &v52);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v55.m128i_i64[1]) <= 0xF )
    goto LABEL_136;
  v17 = (__m128i *)sub_2241490((unsigned __int64 *)&v55, ", #Unknown PRs: ", 0x10u);
  v57[0] = (unsigned __int64)&v58;
  if ( (__m128i *)v17->m128i_i64[0] == &v17[1] )
  {
    v58 = _mm_loadu_si128(v17 + 1);
  }
  else
  {
    v57[0] = v17->m128i_i64[0];
    v58.m128i_i64[0] = v17[1].m128i_i64[0];
  }
  v18 = v17->m128i_u64[1];
  v17[1].m128i_i8[0] = 0;
  v57[1] = v18;
  v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
  v17->m128i_i64[1] = 0;
  sub_8FD5D0(&v62, (__int64)v57, &v59);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v62.m128i_i64[1]) <= 0x14 )
    goto LABEL_136;
  v19 = (__m128i *)sub_2241490((unsigned __int64 *)&v62, ", #Reaching Kernels: ", 0x15u);
  v64[0] = (unsigned __int64)&v65;
  if ( (__m128i *)v19->m128i_i64[0] == &v19[1] )
  {
    v65 = _mm_loadu_si128(v19 + 1);
  }
  else
  {
    v64[0] = v19->m128i_i64[0];
    v65.m128i_i64[0] = v19[1].m128i_i64[0];
  }
  v20 = v19->m128i_u64[1];
  v19[1].m128i_i8[0] = 0;
  v64[1] = v20;
  v19->m128i_i64[0] = (__int64)v19[1].m128i_i64;
  v19->m128i_i64[1] = 0;
  sub_8FD5D0(&v69, (__int64)v64, &v66);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v69.m128i_i64[1]) <= 0xD )
    goto LABEL_136;
  v21 = (__m128i *)sub_2241490((unsigned __int64 *)&v69, ", #ParLevels: ", 0xEu);
  v71[0] = (unsigned __int64)&v72;
  if ( (__m128i *)v21->m128i_i64[0] == &v21[1] )
  {
    v72 = _mm_loadu_si128(v21 + 1);
  }
  else
  {
    v71[0] = v21->m128i_i64[0];
    v72.m128i_i64[0] = v21[1].m128i_i64[0];
  }
  v22 = v21->m128i_u64[1];
  v21[1].m128i_i8[0] = 0;
  v71[1] = v22;
  v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
  v21->m128i_i64[1] = 0;
  sub_8FD5D0(&v76, (__int64)v71, &v73);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v76.m128i_i64[1]) <= 0xC )
    goto LABEL_136;
  v23 = (__m128i *)sub_2241490((unsigned __int64 *)&v76, ", NestedPar: ", 0xDu);
  v78 = v80;
  if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
  {
    v80[0] = _mm_loadu_si128(v23 + 1);
  }
  else
  {
    v78 = (_OWORD *)v23->m128i_i64[0];
    *(_QWORD *)&v80[0] = v23[1].m128i_i64[0];
  }
  v24 = v23->m128i_i64[1];
  v23[1].m128i_i8[0] = 0;
  v79 = v24;
  v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
  v23->m128i_i64[1] = 0;
  v25 = strlen(v2);
  if ( v25 > 0x3FFFFFFFFFFFFFFFLL - v79 )
LABEL_136:
    sub_4262D8((__int64)"basic_string::append");
  v26 = (__m128i *)sub_2241490((unsigned __int64 *)&v78, v2, v25);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v26->m128i_i64[0] == &v26[1] )
  {
    a1[1] = _mm_loadu_si128(v26 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v26->m128i_i64[0];
    a1[1].m128i_i64[0] = v26[1].m128i_i64[0];
  }
  v27 = v26->m128i_i64[1];
  v26->m128i_i64[0] = (__int64)v26[1].m128i_i64;
  v28 = v78;
  v26->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v27;
  v26[1].m128i_i8[0] = 0;
  if ( v28 != v80 )
    j_j___libc_free_0((unsigned __int64)v28);
  if ( (__int64 *)v76.m128i_i64[0] != &v77 )
    j_j___libc_free_0(v76.m128i_u64[0]);
  if ( (__m128i *)v71[0] != &v72 )
    j_j___libc_free_0(v71[0]);
  if ( (__int64 *)v69.m128i_i64[0] != &v70 )
    j_j___libc_free_0(v69.m128i_u64[0]);
  if ( (__m128i *)v64[0] != &v65 )
    j_j___libc_free_0(v64[0]);
  if ( (__int64 *)v62.m128i_i64[0] != &v63 )
    j_j___libc_free_0(v62.m128i_u64[0]);
  if ( (__m128i *)v57[0] != &v58 )
    j_j___libc_free_0(v57[0]);
  if ( (__int64 *)v55.m128i_i64[0] != &v56 )
    j_j___libc_free_0(v55.m128i_u64[0]);
  if ( (__int64 *)v50.m128i_i64[0] != &v51 )
    j_j___libc_free_0(v50.m128i_u64[0]);
  if ( (__int64 *)v46.m128i_i64[0] != &v47 )
    j_j___libc_free_0(v46.m128i_u64[0]);
  if ( (__int64 *)v42[0] != &v43 )
    j_j___libc_free_0(v42[0]);
  if ( (_BYTE *)v44[0] != v45 )
    j_j___libc_free_0(v44[0]);
  if ( (_QWORD *)v48[0] != v49 )
    j_j___libc_free_0(v48[0]);
  if ( v52 != (_BYTE *)v54 )
    j_j___libc_free_0((unsigned __int64)v52);
  if ( v59 != (_BYTE *)v61 )
    j_j___libc_free_0((unsigned __int64)v59);
  if ( v66 != (_BYTE *)v68 )
    j_j___libc_free_0((unsigned __int64)v66);
  if ( v73 != (_BYTE *)v75 )
    j_j___libc_free_0((unsigned __int64)v73);
  return a1;
}
